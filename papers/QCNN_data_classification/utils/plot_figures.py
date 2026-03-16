#!/usr/bin/env python3
"""Run QConv sweeps, auto-resume past work, and plot accuracy/parameter trends."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from lib.models import QConvModel, build_quantum_kernels
except ModuleNotFoundError:  # Allow running as a package module.
    from ..lib.models import QConvModel, build_quantum_kernels

PCA_OPTIONS = (16,)
NB_KERNELS = (2, 3)
KERNEL_SIZES = (2, 3)
STRIDE_CHOICES = {
    1: (1, 2),
    2: (1, 2),
}
KERNEL_MODES = (12, 16)
PARAM_LABELS = {
    "kernel_modes": "Kernel modes",
    "stride": "Stride",
    "nb_kernels": "Number of kernels",
    "kernel_size": "Kernel size",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep qconv hyperparameters, run implementation.py, and plot results."
    )
    parser.add_argument(
        "--implementation",
        type=Path,
        default=Path("implementation.py"),
        help="Path to the implementation.py entrypoint.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        choices=["mnist", "fashionmnist", "kmnist"],
        help="Dataset to feed into implementation.py.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="Optimizer steps per run (default: 100 for faster sweeps).",
    )
    parser.add_argument(
        "--seeds", type=int, default=1, help="Number of seeds per run (default: 1)."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots"),
        help="Directory where aggregated data and figures are stored.",
    )
    return parser.parse_args()


def snapshot_run_dirs(results_dir: Path) -> dict[str, Path]:
    if not results_dir.exists():
        return {}
    return {path.name: path for path in results_dir.glob("run_*") if path.is_dir()}


def detect_new_run_dir(before: dict[str, Path], after: dict[str, Path]) -> Path:
    before_names = set(before)
    after_names = set(after)
    new_names = after_names - before_names
    if new_names:
        latest_name = sorted(new_names)[-1]
        return after[latest_name]

    before_mtime = max((p.stat().st_mtime for p in before.values()), default=-1.0)
    newest = max(after.values(), key=lambda p: p.stat().st_mtime, default=None)
    if newest and newest.stat().st_mtime > before_mtime:
        return newest
    raise RuntimeError(
        "Failed to locate new results directory after running implementation."
    )


def load_config_snapshot(run_dir: Path) -> dict[str, Any]:
    snapshot_path = run_dir / "config_snapshot.json"
    with open(snapshot_path, encoding="utf-8") as fh:
        return json.load(fh)


def build_config_key_from_values(
    dataset: str, steps: int, seeds: int, config: dict[str, int]
) -> tuple:
    return (
        dataset,
        config["pca_dim"],
        config["nb_kernels"],
        config["kernel_size"],
        config["stride"],
        config["kernel_modes"],
        steps,
        seeds,
        True,  # compare_classical
    )


def build_config_key_from_snapshot(snapshot: dict[str, Any]) -> tuple | None:
    required = [
        "dataset",
        "pca_dim",
        "nb_kernels",
        "kernel_size",
        "stride",
        "kernel_modes",
        "steps",
        "seeds",
    ]
    if not all(k in snapshot for k in required):
        return None
    if snapshot.get("model") != "qconv":
        return None
    if not snapshot.get("compare_classical"):
        return None
    return (
        snapshot["dataset"],
        int(snapshot["pca_dim"]),
        int(snapshot["nb_kernels"]),
        int(snapshot["kernel_size"]),
        int(snapshot["stride"]),
        int(snapshot["kernel_modes"]),
        int(snapshot["steps"]),
        int(snapshot["seeds"]),
        bool(snapshot.get("compare_classical")),
    )


def index_existing_runs(results_root: Path) -> dict[tuple, tuple[Path, dict[str, Any]]]:
    index: dict[tuple, tuple[Path, dict[str, Any]]] = {}
    if not results_root.exists():
        return index
    for run_dir in sorted(results_root.glob("run_*")):
        try:
            snapshot = load_config_snapshot(run_dir)
        except FileNotFoundError:
            continue
        key = build_config_key_from_snapshot(snapshot)
        if key is None:
            continue
        current = index.get(key)
        if current is None or run_dir.stat().st_mtime > current[0].stat().st_mtime:
            index[key] = (run_dir, snapshot)
    return index


def parse_run_summary(run_dir: Path) -> dict[str, dict[str, float]]:
    summary_path = run_dir / "run_summary.json"
    with open(summary_path, encoding="utf-8") as fh:
        data = json.load(fh)
    variants = {
        entry["variant"]: {
            "mean_accuracy": float(entry["mean_accuracy"]),
            "std_accuracy": float(entry["std_accuracy"]),
            "param_count": entry.get("param_count"),
        }
        for entry in data.get("variants", [])
    }
    return variants


def compute_param_counts(
    snapshot: dict[str, Any], cache: dict[tuple, dict[str, int]]
) -> dict[str, int]:
    key = (
        int(snapshot["pca_dim"]),
        int(snapshot["nb_kernels"]),
        int(snapshot["kernel_size"]),
        int(snapshot["stride"]),
        int(snapshot.get("kernel_modes") or snapshot["kernel_size"]),
        int(snapshot.get("n_photons", 4)),
        snapshot.get("state_pattern", "default"),
        bool(snapshot.get("reservoir_mode", False)),
        bool(snapshot.get("amplitudes", False)),
    )
    if key in cache:
        return cache[key]

    amplitudes = bool(snapshot.get("amplitudes", False))
    base_kwargs = {
        "input_dim": int(snapshot["pca_dim"]),
        "n_kernels": int(snapshot["nb_kernels"]),
        "kernel_size": int(snapshot["kernel_size"]),
        "stride": int(snapshot["stride"]),
        "amplitudes_encoding": amplitudes,
    }
    classical_model = QConvModel(bias=True, **base_kwargs)
    classical_params = sum(
        p.numel() for p in classical_model.parameters() if p.requires_grad
    )

    kernel_modes = int(snapshot.get("kernel_modes") or snapshot["kernel_size"])
    n_photons = int(snapshot.get("n_photons", 4))
    quantum_modules = build_quantum_kernels(
        n_kernels=int(snapshot["nb_kernels"]),
        kernel_size=int(snapshot["kernel_size"]),
        kernel_modes=kernel_modes,
        n_photons=n_photons,
        state_pattern=str(snapshot.get("state_pattern", "default")),
        reservoir_mode=bool(snapshot.get("reservoir_mode", False)),
        amplitudes_encoding=amplitudes,
        show_circuit=False,
    )
    quantum_model = QConvModel(
        bias=False, kernel_modules=quantum_modules, **base_kwargs
    )
    quantum_params = sum(
        p.numel() for p in quantum_model.parameters() if p.requires_grad
    )

    counts = {
        "qconv_quantum": quantum_params,
        "qconv_classical": classical_params,
        "qconv_classical_only": classical_params,
    }
    cache[key] = counts
    return counts


def ensure_variant_param_counts(
    variant_stats: dict[str, dict[str, float]],
    snapshot: dict[str, Any],
    cache: dict[tuple, dict[str, int]],
) -> None:
    missing = [
        name
        for name, stats in variant_stats.items()
        if stats.get("param_count") is None
    ]
    if not missing:
        return
    counts = compute_param_counts(snapshot, cache)
    for name in missing:
        if name in counts:
            variant_stats[name]["param_count"] = counts[name]


def run_configuration(
    impl_path: Path,
    config: dict[str, int],
    dataset: str,
    steps: int,
    seeds: int,
) -> Path:
    results_root = impl_path.parent / f"results-{dataset}"
    before = snapshot_run_dirs(results_root)
    cmd = [
        sys.executable,
        str(impl_path),
        "--model",
        "qconv",
        "--dataset",
        dataset,
        "--compare_classical",
        "--nb_kernels",
        str(config["nb_kernels"]),
        "--kernel_size",
        str(config["kernel_size"]),
        "--stride",
        str(config["stride"]),
        "--kernel_modes",
        str(config["kernel_modes"]),
        "--pca_dim",
        str(config["pca_dim"]),
        "--steps",
        str(steps),
        "--seeds",
        str(seeds),
    ]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(impl_path.parent),
    )
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        raise RuntimeError(f"implementation.py failed with exit code {proc.returncode}")
    after = snapshot_run_dirs(results_root)
    return detect_new_run_dir(before, after)


def generate_configs() -> Iterable[dict[str, int]]:
    for pca_dim in PCA_OPTIONS:
        for nb_kernels in NB_KERNELS:
            for kernel_size in KERNEL_SIZES:
                if kernel_size > pca_dim:
                    continue
                strides = STRIDE_CHOICES.get(kernel_size, (1,))
                for stride in strides:
                    if kernel_size == 1 and stride != 1:
                        continue
                    for kernel_modes in KERNEL_MODES:
                        if kernel_size > kernel_modes:
                            continue
                        yield {
                            "pca_dim": pca_dim,
                            "nb_kernels": nb_kernels,
                            "kernel_size": kernel_size,
                            "stride": stride,
                            "kernel_modes": kernel_modes,
                        }


def summarize_series(
    records: list[dict],
    pca_dim: int,
    param_key: str,
    variant: str,
    filters: dict[str, Any] | None = None,
) -> tuple[list[int], list[float], list[float]]:
    metric_values: dict[int, list[float]] = defaultdict(list)
    param_counts: dict[int, list[float]] = defaultdict(list)
    filters = filters or {}
    for entry in records:
        if entry["pca_dim"] != pca_dim or entry["variant"] != variant:
            continue
        if any(entry.get(key) != value for key, value in filters.items()):
            continue
        key = entry[param_key]
        metric_values[key].append(entry["mean_accuracy"])
        if entry["param_count"] is not None:
            param_counts[key].append(entry["param_count"])
    sorted_keys = sorted(metric_values.keys())
    accuracies = [float(np.mean(metric_values[k])) for k in sorted_keys]
    param_means = []
    for k in sorted_keys:
        if param_counts[k]:
            param_means.append(float(np.mean(param_counts[k])))
        else:
            param_means.append(np.nan)
    return sorted_keys, accuracies, param_means


def find_variant(records: list[dict], prefix: str, pca_dim: int) -> str | None:
    for entry in records:
        if entry["pca_dim"] == pca_dim and entry["variant"].startswith(prefix):
            return entry["variant"]
    return None


def plot_relationship(
    records: list[dict],
    pca_dim: int,
    param_key: str,
    variants: list[tuple[str, str]],
    title_suffix: str,
    output_path: Path,
    filters: dict[str, Any] | None = None,
) -> None:
    if not variants:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for variant_name, label in variants:
        xs, accs, params = summarize_series(
            records, pca_dim, param_key, variant_name, filters=filters
        )
        if not xs:
            continue
        axes[0].plot(xs, accs, marker="o", label=label)
        axes[1].plot(xs, params, marker="o", label=label)
    axes[0].set_xlabel(PARAM_LABELS[param_key])
    axes[0].set_ylabel("Mean accuracy")
    axes[0].set_title("Accuracy")
    axes[1].set_xlabel(PARAM_LABELS[param_key])
    axes[1].set_ylabel("Trainable parameters")
    axes[1].set_title("Parameters")
    axes[0].legend()
    axes[1].legend()
    fig.suptitle(f"PCA {pca_dim}: {title_suffix}")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def write_records(
    output_dir: Path, args: argparse.Namespace, records: list[dict]
) -> None:
    data_path = output_dir / "sweep_records.json"
    payload = {
        "updated_at": datetime.utcnow().isoformat(),
        "dataset": args.dataset,
        "steps": args.steps,
        "seeds": args.seeds,
        "records": records,
    }
    with open(data_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def main() -> None:
    args = parse_args()
    impl_path = args.implementation.resolve()
    if not impl_path.exists():
        raise FileNotFoundError(f"implementation.py not found at {impl_path}")
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    configs = list(generate_configs())
    total = len(configs)
    results_root = impl_path.parent / f"results-{args.dataset}"
    existing_runs = index_existing_runs(results_root)
    param_cache: dict[tuple, dict[str, int]] = {}
    records: list[dict] = []

    print(f"Planned {total} runs.")
    for idx, config in enumerate(configs, start=1):
        key = build_config_key_from_values(args.dataset, args.steps, args.seeds, config)
        cached = existing_runs.pop(key, None)
        if cached:
            run_dir, snapshot = cached
            reused = True
        else:
            reused = False
            run_dir = run_configuration(
                impl_path=impl_path,
                config=config,
                dataset=args.dataset,
                steps=args.steps,
                seeds=args.seeds,
            )
            snapshot = load_config_snapshot(run_dir)

        status = "reusing" if reused else "running"
        print(
            f"[{idx}/{total}] ({status}) PCA={config['pca_dim']} | kernels={config['nb_kernels']} "
            f"| kernel_size={config['kernel_size']} | stride={config['stride']} | modes={config['kernel_modes']}"
        )

        variant_stats = parse_run_summary(run_dir)
        ensure_variant_param_counts(variant_stats, snapshot, param_cache)
        for variant_name, stats in variant_stats.items():
            records.append(
                {
                    **config,
                    "variant": variant_name,
                    "mean_accuracy": stats["mean_accuracy"],
                    "std_accuracy": stats["std_accuracy"],
                    "param_count": stats.get("param_count"),
                    "run_dir": str(run_dir),
                }
            )
        write_records(output_dir, args, records)

    print(f"Wrote sweep data to {output_dir / 'sweep_records.json'}")

    for pca_dim in PCA_OPTIONS:
        quantum_variant = find_variant(records, "qconv_quantum", pca_dim)
        classical_variant = find_variant(records, "qconv_classical", pca_dim)

        if quantum_variant:
            plot_relationship(
                records,
                pca_dim,
                "kernel_modes",
                [(quantum_variant, "Quantum")],
                "Accuracy/parameters vs. kernel modes",
                output_dir / f"pca{pca_dim}_kernel_modes.png",
            )
        stride_variants = []
        if quantum_variant:
            stride_variants.append((quantum_variant, "Quantum"))
        if classical_variant:
            stride_variants.append((classical_variant, "Classical"))
        if stride_variants:
            plot_relationship(
                records,
                pca_dim,
                "stride",
                stride_variants,
                "Accuracy/parameters vs. stride",
                output_dir / f"pca{pca_dim}_stride.png",
            )
            plot_relationship(
                records,
                pca_dim,
                "nb_kernels",
                stride_variants,
                "Accuracy/parameters vs. number of kernels",
                output_dir / f"pca{pca_dim}_nb_kernels.png",
            )
            plot_relationship(
                records,
                pca_dim,
                "kernel_size",
                stride_variants,
                "Accuracy/parameters vs. kernel size",
                output_dir / f"pca{pca_dim}_kernel_size.png",
                filters={"stride": 2},
            )


if __name__ == "__main__":
    main()
