from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# This script lives inside the QRNN paper folder.
QRNN_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = QRNN_DIR.parents[1]

# Allow running this script without installing the repo as a package.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runtime_lib import run_from_project  # noqa: E402


@dataclass(frozen=True)
class RunResult:
    dataset: str
    model: str
    sweep: str
    config_path: str
    run_dir: str
    seconds: float
    final_train_loss: float | None
    final_train_accuracy: float | None
    final_val_loss: float | None
    final_val_accuracy: float | None
    final_test_loss: float | None
    final_test_accuracy: float | None
    best_val_loss: float | None
    error: str | None = None


def _read_accuracy(metrics: dict[str, Any]) -> dict[str, float | None]:
    acc = metrics.get("accuracy")
    if not isinstance(acc, dict):
        acc = {}
    return {
        "train": metrics.get("final_train_accuracy", acc.get("train")),
        "val": metrics.get("final_val_accuracy", acc.get("val")),
        "test": metrics.get("final_test_accuracy", acc.get("test")),
    }


def _label_for_result(result: RunResult) -> str:
    if result.sweep:
        return f"{result.model} | {result.sweep}"
    return result.model


def _maybe_generate_plots(
    results: list[RunResult],
    *,
    run_group: Path,
    datasets: Iterable[str],
) -> None:
    try:
        from papers.shared.time_series import plot_metrics as shared_plot_metrics
        from papers.shared.time_series import (
            plot_predictions as shared_plot_predictions,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[plots] Skipped (import error): {type(exc).__name__}: {exc}")
        return

    for ds_name in datasets:
        ds_results = [
            r for r in results if r.dataset == ds_name and r.error is None and r.run_dir
        ]
        if not ds_results:
            continue

        run_dirs = [Path(r.run_dir) for r in ds_results]
        labels = [_label_for_result(r) for r in ds_results]

        try:
            metrics_path = run_group / f"{ds_name}__metrics_overlay.png"
            shared_plot_metrics.plot_runs(
                run_dirs,
                out_path=metrics_path,
                show=False,
                include_val=True,
                labels=labels,
            )
            print(f"[plots] Saved metrics overlay to {metrics_path}")
        except Exception as exc:  # noqa: BLE001
            print(
                f"[plots] Failed to plot metrics for {ds_name}: {type(exc).__name__}: {exc}"
            )

        try:
            pred_path = run_group / f"{ds_name}__predictions_overlay.png"
            shared_plot_predictions.plot_runs(
                run_dirs, out_path=pred_path, labels=labels
            )
            print(f"[plots] Saved predictions overlay to {pred_path}")
        except Exception as exc:  # noqa: BLE001
            print(
                f"[plots] Failed to plot predictions for {ds_name}: {type(exc).__name__}: {exc}"
            )


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _dump_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _infer_feature_dim(dataset_cfg: dict[str, Any]) -> int:
    gen = dataset_cfg.get("generator")
    if isinstance(gen, dict):
        feat = gen.get("feature_dim")
        if feat is not None:
            return int(feat)
    cols = dataset_cfg.get("feature_columns")
    if cols:
        return int(len(cols))
    feat = dataset_cfg.get("feature_dim")
    if feat is not None:
        return int(feat)
    raise ValueError("Unable to infer dataset feature dimension")


def _grid(sweep: dict[str, Iterable[Any]]) -> list[dict[str, Any]]:
    keys = list(sweep)
    values = [list(sweep[k]) for k in keys]
    combos = []
    for prod in itertools.product(*values):
        combos.append(dict(zip(keys, prod)))
    return combos


def _sweep_tag(params: dict[str, Any]) -> str:
    parts = []
    for k in sorted(params):
        parts.append(f"{k}={params[k]}")
    return ",".join(parts)


def _make_config(
    *,
    base_dataset_cfg: dict[str, Any],
    base_training_cfg: dict[str, Any],
    description: str,
    dtype: str | None,
    model_name: str,
    model_params: dict[str, Any],
    epochs: int,
) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "description": description,
        "dataset": base_dataset_cfg,
        "model": {"name": model_name, "params": model_params},
        "training": dict(base_training_cfg),
    }
    if dtype is not None:
        cfg["dtype"] = dtype

    cfg["training"]["epochs"] = int(epochs)

    # Keep early stopping off for deterministic short sweeps.
    early = cfg["training"].get("early_stopping")
    if isinstance(early, dict):
        early = dict(early)
        early["enabled"] = False
        cfg["training"]["early_stopping"] = early

    return cfg


def _read_metrics(run_dir: Path) -> dict[str, Any]:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        return {}
    return _load_json(metrics_path)


def _run_one(
    *,
    config_path: Path,
    outdir: Path,
    seed: int,
    device: str | None,
) -> tuple[Path, float]:
    argv = [
        "--config",
        str(config_path),
        "--outdir",
        str(outdir),
        "--seed",
        str(seed),
    ]
    if device is not None:
        argv.extend(["--device", device])

    start = time.time()
    run_dir = run_from_project(QRNN_DIR, argv)
    seconds = time.time() - start
    return run_dir, seconds


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare QRNN models (rnn_baseline, gate_pqrnn, photonic_qrnn) on sin_tiny, airline_small, and weather_small. "
            "Runs a small parameter sweep with fixed epochs and exports a CSV summary."
        )
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of epochs per run (default: 5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device override passed to the runtime (e.g. cpu, mps)",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=str(QRNN_DIR / "outdir" / "qrnn_compare"),
        help="Base output directory (default: papers/QRNN/outdir/qrnn_compare)",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Optional cap on number of runs (for quick debugging)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned runs without executing",
    )
    parser.add_argument(
        "--plots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Generate overlay plots (metrics + predictions) at the end of the sweep "
            "and save them next to summary.csv (default: enabled)"
        ),
    )

    parser.add_argument(
        "--feature-normalization",
        type=str,
        default="minmax_-1_1",
        help=(
            "Feature normalization scheme used by the QRNN dataloaders. "
            "Choices: standard (default), none, minmax_0_1, minmax_-1_1. "
            "Use minmax_-1_1 when enabling --input-encoding=arccos."
        ),
    )
    parser.add_argument(
        "--input-encoding",
        type=str,
        default="arccos",
        help=(
            "Input-to-angle encoding applied inside gate_pqrnn and photonic_qrnn. "
            "Choices: identity (default), arccos."
        ),
    )

    args = parser.parse_args(argv)

    epochs = int(args.epochs)
    seed = int(args.seed)
    base_outdir = Path(args.outdir).expanduser().resolve()
    feature_norm = str(args.feature_normalization)
    input_encoding = str(args.input_encoding)

    # Dataset baselines (we reuse dataset + training blocks from the existing configs).
    sin_base = _load_json(QRNN_DIR / "configs" / "sin_tiny_rnn.json")
    airline_base = _load_json(QRNN_DIR / "configs" / "airline_small_rnn.json")
    weather_base = _load_json(QRNN_DIR / "configs" / "weather_strong_rnn.json")

    # Keep weather runs reasonably small for the quantum models.
    # (Gate pQRNN scales poorly with feature_dim due to density-matrix simulation.)
    weather_base = dict(weather_base)
    weather_dataset = dict(weather_base.get("dataset") or {})
    weather_dataset.update(
        {
            "sequence_length": 12,
            "prediction_horizon": int(
                weather_dataset.get("prediction_horizon", 1) or 1
            ),
            "max_rows": int(weather_dataset.get("max_rows") or 1200),
            "batch_size": 16,
            # Use a reduced feature set so feature_dim stays small.
            # These columns exist after the szeged_weather preprocess.
            "feature_columns": ["min_temperature", "avg_humidity", "month"],
        }
    )
    weather_base["dataset"] = weather_dataset

    datasets = {
        "sin_tiny": sin_base,
        "airline_small": airline_base,
        "weather_small": weather_base,
    }

    planned: list[tuple[str, str, dict[str, Any], dict[str, Any], str | None]] = []

    for ds_name, base in datasets.items():
        dataset_cfg = base.get("dataset") or {}
        training_cfg = base.get("training") or {}
        feat_dim = _infer_feature_dim(dataset_cfg)

        # RNN baseline sweep.
        for p in _grid({"hidden_dim": [16, 32]}):
            model_params = dict(base.get("model", {}).get("params", {}) or {})
            model_params.update(
                {
                    "cell_type": "rnn",
                    "hidden_dim": int(p["hidden_dim"]),
                    "layers": int(model_params.get("layers", 1)),
                    "dropout": float(model_params.get("dropout", 0.0)),
                    "input_dropout": float(model_params.get("input_dropout", 0.0)),
                    "bidirectional": bool(model_params.get("bidirectional", False)),
                }
            )
            planned.append(
                (
                    ds_name,
                    "rnn_baseline",
                    model_params,
                    training_cfg,
                    base.get("dtype"),
                )
            )

        # Gate-based pQRNN (PennyLane) sweep.
        # Align naming with photonic QRNN: use kd/kh.
        # Constraint: kd must be >= feature dimension.
        gate_kd = feat_dim
        for p in _grid({"kh": [1, 2], "depth": [1]}):
            model_params = {
                "kd": int(gate_kd),
                "kh": int(p["kh"]),
                "depth": int(p["depth"]),
                "entangling": "cb",
                "entangling_wrap": True,
            }
            # Gate-based runs are more stable in float64.
            planned.append(
                (ds_name, "gate_pqrnn", model_params, training_cfg, "float64")
            )

        # Photonic QRNN sweep.
        # Must satisfy feat_dim <= 2*kd. We keep kd=1 for these tiny benchmarks.
        kd = 1
        if feat_dim > 2 * kd:
            kd = (feat_dim + 1) // 2
        for p in _grid({"kh": [1, 2], "depth": [1]}):
            model_params = {
                "kd": int(kd),
                "kh": int(p["kh"]),
                "depth": int(p["depth"]),
                "shots": 0,
                "measurement_space": "dual_rail",
            }
            planned.append(
                (
                    ds_name,
                    "photonic_qrnn",
                    model_params,
                    training_cfg,
                    base.get("dtype"),
                )
            )

    if args.max_runs is not None:
        planned = planned[: int(args.max_runs)]

    run_group = base_outdir / f"compare_seed{seed}_e{epochs}"
    configs_dir = run_group / "sweep_configs"
    results_csv = run_group / "summary.csv"

    if args.dry_run:
        for ds_name, model_name, model_params, _training_cfg, dtype in planned:
            print(ds_name, model_name, dtype, _sweep_tag(model_params))
        print(f"Planned runs: {len(planned)}")
        return 0

    run_group.mkdir(parents=True, exist_ok=True)

    results: list[RunResult] = []

    for idx, (ds_name, model_name, model_params, training_cfg, dtype) in enumerate(
        planned, start=1
    ):
        sweep = _sweep_tag(model_params)
        ds_base = datasets[ds_name]
        dataset_cfg = dict(ds_base.get("dataset") or {})
        dataset_cfg["feature_normalization"] = feature_norm

        if model_name in {"gate_pqrnn", "photonic_qrnn"}:
            model_params = dict(model_params)
            model_params["input_encoding"] = input_encoding

        desc = f"sweep compare | dataset={ds_name} model={model_name} | {sweep}"
        cfg = _make_config(
            base_dataset_cfg=dataset_cfg,
            base_training_cfg=training_cfg,
            description=desc,
            dtype=dtype,
            model_name=model_name,
            model_params=model_params,
            epochs=epochs,
        )

        config_path = configs_dir / f"{ds_name}__{model_name}__{idx:03d}.json"
        _dump_json(config_path, cfg)

        outdir = run_group / ds_name / model_name / f"run_{idx:03d}"
        print(f"[{idx}/{len(planned)}] {ds_name} | {model_name} | {sweep}")

        try:
            run_dir, seconds = _run_one(
                config_path=config_path,
                outdir=outdir,
                seed=seed,
                device=args.device,
            )
            metrics = _read_metrics(run_dir)
            acc = _read_accuracy(metrics)
            results.append(
                RunResult(
                    dataset=ds_name,
                    model=model_name,
                    sweep=sweep,
                    config_path=str(config_path),
                    run_dir=str(run_dir),
                    seconds=float(seconds),
                    final_train_loss=metrics.get("final_train_loss"),
                    final_train_accuracy=acc["train"],
                    final_val_loss=metrics.get("final_val_loss"),
                    final_val_accuracy=acc["val"],
                    final_test_loss=metrics.get("final_test_loss"),
                    final_test_accuracy=acc["test"],
                    best_val_loss=metrics.get("best_val_loss"),
                    error=None,
                )
            )
        except Exception as exc:  # noqa: BLE001
            results.append(
                RunResult(
                    dataset=ds_name,
                    model=model_name,
                    sweep=sweep,
                    config_path=str(config_path),
                    run_dir="",
                    seconds=0.0,
                    final_train_loss=None,
                    final_train_accuracy=None,
                    final_val_loss=None,
                    final_val_accuracy=None,
                    final_test_loss=None,
                    final_test_accuracy=None,
                    best_val_loss=None,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )

        # Append to CSV after each run for resilience.
        write_header = not results_csv.exists()
        with results_csv.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "dataset",
                    "model",
                    "sweep",
                    "seconds",
                    "final_train_loss",
                    "final_train_accuracy",
                    "final_val_loss",
                    "final_val_accuracy",
                    "final_test_loss",
                    "final_test_accuracy",
                    "best_val_loss",
                    "error",
                    "config_path",
                    "run_dir",
                ],
            )
            if write_header:
                writer.writeheader()
            last = results[-1]
            writer.writerow(
                {
                    "dataset": last.dataset,
                    "model": last.model,
                    "sweep": last.sweep,
                    "seconds": f"{last.seconds:.3f}",
                    "final_train_loss": last.final_train_loss,
                    "final_train_accuracy": last.final_train_accuracy,
                    "final_val_loss": last.final_val_loss,
                    "final_val_accuracy": last.final_val_accuracy,
                    "final_test_loss": last.final_test_loss,
                    "final_test_accuracy": last.final_test_accuracy,
                    "best_val_loss": last.best_val_loss,
                    "error": last.error,
                    "config_path": last.config_path,
                    "run_dir": last.run_dir,
                }
            )

    print(f"Done. Summary: {results_csv}")

    if args.plots:
        _maybe_generate_plots(
            results,
            run_group=run_group,
            datasets=datasets.keys(),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
