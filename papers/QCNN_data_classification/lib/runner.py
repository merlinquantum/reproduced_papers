# ruff: noqa: I001
from __future__ import annotations

import json
import logging
import statistics
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np
import torch.nn as nn
from lib.data import make_pca
from utils.circuit import required_input_params
from utils.training import train_once
import matplotlib
import matplotlib.pyplot as plt

from .models import QConvModel, SingleGI, build_quantum_kernels


def _angle_factor(scale: str) -> float:
    if scale == "pi":
        return float(np.pi)
    if scale == "2pi":
        return float(2 * np.pi)
    return 1.0


def _prepare_models(
    args: SimpleNamespace,
    input_dim: int,
    required_inputs: int,
) -> tuple[list[tuple[str, Callable[[], nn.Module]]], list[str]]:
    logs: list[str] = []
    builders: list[tuple[str, Callable[[], nn.Module]]] = []

    if args.model == "single":
        # Just a simple QuantumLayer on the data
        def build_single() -> nn.Module:
            return SingleGI(
                n_modes=args.n_modes,
                n_features=args.n_features,
                n_photons=args.n_photons,
                reservoir_mode=args.reservoir_mode,
                state_pattern=args.state_pattern,
                required_inputs=required_inputs,
                input_dim=input_dim,
            )

        builders.append(("single", build_single))
        return builders, logs

    # qconv path
    if args.kernel_size > args.pca_dim:
        raise ValueError("kernel_size cannot exceed the PCA dimension.")
    if args.stride <= 0:
        raise ValueError("stride must be a positive integer.")
    num_windows = 1 + (input_dim - args.kernel_size) // args.stride
    if num_windows <= 0:
        raise ValueError("qconv configuration results in zero sliding windows.")

    base_kwargs = {
        "input_dim": input_dim,
        "n_kernels": args.nb_kernels,
        "kernel_size": args.kernel_size,
        "stride": args.stride,
        "amplitudes_encoding": args.amplitude,
    }

    kernel_modes = args.kernel_modes or args.kernel_size
    n_photons = args.kernel_modes // 2

    def _build_quantum_modules() -> list[nn.Module]:
        return build_quantum_kernels(
            n_kernels=args.nb_kernels,
            kernel_size=args.kernel_size,
            kernel_modes=kernel_modes,
            n_photons=n_photons,
            state_pattern=args.state_pattern,
            reservoir_mode=args.reservoir_mode,
            amplitudes_encoding=args.amplitude,
        )

    # Classical variant (always available for comparison/logging)
    classical_sample = QConvModel(bias=True, **base_kwargs)
    classical_output_dim = classical_sample.output_features

    def build_classical() -> nn.Module:
        return QConvModel(bias=True, **base_kwargs)

    use_quantum = not args.conv_classical or args.compare_classical
    if args.conv_classical and not args.compare_classical:
        logs.append(
            f"classical: {args.nb_kernels} kernels → {classical_output_dim} dims"
        )
        builders.append(("qconv_classical_only", build_classical))
        return builders, logs

    quantum_modules = _build_quantum_modules()
    quantum_sample = QConvModel(
        bias=False, kernel_modules=quantum_modules, **base_kwargs
    )
    quantum_output_dim = quantum_sample.output_features
    logs.append(f"quantum: {args.nb_kernels} kernels → {quantum_output_dim} dims")

    def build_quantum() -> nn.Module:
        return QConvModel(
            bias=False,
            kernel_modules=_build_quantum_modules(),
            **base_kwargs,
        )

    builders.append(("qconv_quantum", build_quantum))

    if args.compare_classical or not use_quantum:
        logs.append(
            f"classical: {args.nb_kernels} kernels → {classical_output_dim} dims"
        )
        builders.append(("qconv_classical", build_classical))

    return builders, logs


def _plot_training_figures(
    run_dir: Path, variant_histories: dict[str, list[dict[str, list[float]]]]
) -> None:

    matplotlib.use("Agg")

    def _save_metric(metric_key: str, ylabel: str, title: str, filename: Path) -> None:
        metric_curves: dict[str, list[list[float]]] = {}
        for variant, histories in variant_histories.items():
            curves: list[list[float]] = []
            for entry in histories:
                curve_vals = entry.get(metric_key)
                if curve_vals:
                    curves.append(curve_vals)
            if curves:
                metric_curves[variant] = curves
        if not metric_curves:
            return

        plt.figure(figsize=(10, 6))
        color_cycle = plt.cm.get_cmap("tab10", len(metric_curves))
        for idx, (variant, curves) in enumerate(sorted(metric_curves.items())):
            color = color_cycle(idx % color_cycle.N)
            min_len = min(len(curve) for curve in curves)
            aligned = np.array([curve[:min_len] for curve in curves], dtype=float)
            steps = np.arange(1, min_len + 1)
            for curve in aligned:
                plt.plot(steps, curve, color=color, alpha=0.25, linewidth=0.8)
            mean_curve = aligned.mean(axis=0)
            plt.plot(
                steps, mean_curve, color=color, linewidth=2.0, label=f"{variant} mean"
            )
            if aligned.shape[0] > 1:
                std_curve = aligned.std(axis=0)
                plt.fill_between(
                    steps,
                    mean_curve - std_curve,
                    mean_curve + std_curve,
                    color=color,
                    alpha=0.1,
                )

        plt.xlabel("Training step")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.legend()
        filename.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()
        logging.info("[figures] Saved %s", filename.relative_to(run_dir))

    figures_dir = run_dir / "figures"
    _save_metric(
        "loss", "Loss", "Training loss per seed", figures_dir / "loss_curves.png"
    )
    _save_metric(
        "accuracy",
        "Accuracy",
        "Test accuracy per step",
        figures_dir / "accuracy_curves.png",
    )


def _to_namespace(cfg: dict[str, Any]) -> SimpleNamespace:
    # Keep config structure flexible; attributes map directly to keys.
    return SimpleNamespace(**cfg)


def train_and_evaluate(cfg: dict[str, Any], run_dir: Path) -> None:
    args = _to_namespace(cfg)

    angle_factor = _angle_factor(args.angle_scale)
    required_inputs = required_input_params(args.n_modes, args.n_features)
    print(f"Using args.dataset = {args.dataset}, PCA dim = {args.pca_dim}")
    (Ztr, ytr), (Zte, yte) = make_pca(args.pca_dim, dataset=args.dataset)
    input_dim = Ztr.shape[-1]

    if args.model == "single" and input_dim != required_inputs:
        raise ValueError(
            f"Single GI model requires feature dimension {required_inputs}, but PCA produced {input_dim}."
        )

    builders, conv_logs = _prepare_models(args, input_dim, required_inputs)

    dataset_pretty = {
        "mnist": "MNIST",
        "fashionmnist": "FashionMNIST",
        "kmnist": "KMNIST",
    }[args.dataset]
    print(
        f"{dataset_pretty} PCA-{args.pca_dim} ready: train {Ztr.shape}, test {Zte.shape} | angle={args.angle_scale}"
    )
    if conv_logs:
        print("Convolution configurations:")
        for log_entry in conv_logs:
            print(f"  - {log_entry}")

    comparison_results = []
    variant_seed_details = {}
    variant_histories: dict[str, list[dict[str, list[float]]]] = {}

    config_snapshot_path = run_dir / "config_snapshot.json"
    variant_param_counts: dict[str, int] = {}
    for name, builder in builders:
        print(
            f"\n=== Evaluating {name} ({args.seeds} seed{'s' if args.seeds > 1 else ''}) ==="
        )
        variant_accs = []
        seed_records = []
        for s in range(args.seeds):
            print(f"[Seed {s + 1}/{args.seeds}]")
            model = builder()
            param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Number of trainable parameters = {param_count}")
            if name not in variant_param_counts:
                variant_param_counts[name] = param_count
            acc, loss_history, accuracy_history = train_once(
                Ztr,
                ytr,
                Zte,
                yte,
                steps=args.steps,
                batch=args.batch,
                opt_name=args.opt,
                lr=args.lr,
                momentum=args.momentum,
                angle_factor=angle_factor,
                seed=1235 + s,
                model=model,
                track_metrics=args.figures,
            )
            print(f"  Test accuracy: {acc * 100:.2f}%")
            variant_accs.append(acc)
            variant_dir = run_dir / name
            variant_dir.mkdir(parents=True, exist_ok=True)
            seed_index = s + 1
            seed_value = 1235 + s
            seed_fname = variant_dir / f"seed_{seed_index:02d}.json"
            seed_data = {
                "seed_index": seed_index,
                "seed_value": seed_value,
                "test_accuracy": float(acc),
                "loss_history": [float(x) for x in loss_history],
            }
            if args.figures:
                seed_data["accuracy_history"] = [float(x) for x in accuracy_history]
            with open(seed_fname, "w", encoding="utf-8") as fh:
                json.dump(seed_data, fh, indent=2)
            seed_records.append(
                {
                    "seed_index": seed_index,
                    "seed_value": seed_value,
                    "test_accuracy": float(acc),
                    "curve_file": seed_fname.name,
                }
            )
            if args.figures:
                variant_histories.setdefault(name, []).append(
                    {
                        "loss": [float(x) for x in loss_history],
                        "accuracy": [float(x) for x in accuracy_history],
                    }
                )
        mean = statistics.mean(variant_accs)
        std = statistics.stdev(variant_accs) if len(variant_accs) > 1 else 0.0
        print(f"→ Summary for {name}: mean {mean * 100:.2f}% ± {std * 100:.2f}%")
        variant_dir = run_dir / name
        summary = {
            "variant": name,
            "mean_accuracy": float(mean),
            "std_accuracy": float(std),
            "seeds": seed_records,
            "summary_file": "summary.json",
            "param_count": variant_param_counts.get(name),
        }
        with open(variant_dir / "summary.json", "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)

        comparison_results.append((name, variant_accs, mean, std))
        variant_seed_details[name] = seed_records

    run_summary = {
        "run_dir": str(run_dir),
        "timestamp": run_dir.name,
        "model": args.model,
        "pca_dim": args.pca_dim,
        "config_snapshot": config_snapshot_path.name
        if config_snapshot_path.exists()
        else None,
        "variants": [],
        "convolution_logs": conv_logs,
    }
    for name, _accs, mean, std in comparison_results:
        run_summary["variants"].append(
            {
                "variant": name,
                "mean_accuracy": float(mean),
                "std_accuracy": float(std),
                "seeds": variant_seed_details.get(name, []),
                "summary_file": f"{name}/summary.json",
                "param_count": variant_param_counts.get(name),
            }
        )

    with open(run_dir / "run_summary.json", "w", encoding="utf-8") as fh:
        json.dump(run_summary, fh, indent=2)

    if args.figures and variant_histories:
        _plot_training_figures(run_dir, variant_histories)

    if len(comparison_results) > 1:
        print("\n=== Overall Comparison ===")
        for name, accs, mean, std in comparison_results:
            acc_line = ", ".join(f"{a * 100:.2f}%" for a in accs)
            print(f"{name}: [{acc_line}] → mean {mean * 100:.2f}% ± {std * 100:.2f}%")
    elif comparison_results:
        name, accs, mean, std = comparison_results[0]
        print("\n=== Summary ===")
        print("Accuracies:", ", ".join(f"{a * 100:.2f}%" for a in accs))
        print(f"Mean ± Std: {mean * 100:.2f}% ± {std * 100:.2f}%")


__all__ = ["train_and_evaluate"]
