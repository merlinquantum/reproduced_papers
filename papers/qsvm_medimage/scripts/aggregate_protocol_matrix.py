#!/usr/bin/env python3
"""Aggregate paired protocol-sensitivity comparisons."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROTOCOLS = {
    "legacy_leak__legacy_trace": {
        "preprocessing_protocol": "legacy_train_plus_heldout",
        "trace_protocol": "legacy_square_only",
        "kernel_normalization": "none",
        "classical_dir": "legacy",
    },
    "train_only__legacy_trace": {
        "preprocessing_protocol": "train_only",
        "trace_protocol": "legacy_square_only",
        "kernel_normalization": "none",
        "classical_dir": "train_only",
    },
    "legacy_leak__train_trace": {
        "preprocessing_protocol": "legacy_train_plus_heldout",
        "trace_protocol": "train_trace",
        "kernel_normalization": "train_trace",
        "classical_dir": "legacy",
    },
    "train_only__train_trace": {
        "preprocessing_protocol": "train_only",
        "trace_protocol": "train_trace",
        "kernel_normalization": "train_trace",
        "classical_dir": "train_only",
    },
}

BASELINES = {
    "linear_c1": ("linear_c1", "linear"),
    "rbf_tuned": ("rbf_tuned", "rbf"),
}


def summarize_deltas(model_f1, baseline_f1, tolerance=1e-12):
    """Summarize paired deltas from the model-minus-baseline perspective."""
    model_values = np.asarray(model_f1, dtype=float)
    baseline_values = np.asarray(baseline_f1, dtype=float)
    if model_values.shape != baseline_values.shape:
        raise ValueError("model_f1 and baseline_f1 must have matching shapes")
    if model_values.ndim != 1 or model_values.size == 0:
        raise ValueError("paired F1 inputs must be non-empty one-dimensional arrays")
    if not np.isfinite(model_values).all() or not np.isfinite(baseline_values).all():
        raise ValueError("paired F1 inputs must contain only finite values")
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")

    deltas = model_values - baseline_values
    ties = np.isclose(deltas, 0.0, atol=tolerance, rtol=0.0)
    return {
        "mean_delta": float(deltas.mean()),
        "wins": int(np.sum((deltas > 0.0) & ~ties)),
        "ties": int(np.sum(ties)),
        "losses": int(np.sum((deltas < 0.0) & ~ties)),
    }


def _select_metrics_row(metrics_path):
    metrics = pd.read_csv(metrics_path)
    if metrics.empty:
        raise ValueError(f"No metrics rows in {metrics_path}")
    if "alpha_sweep" in metrics.columns:
        best = metrics[metrics["alpha_sweep"] == "best"]
        if not best.empty:
            return best.iloc[-1]
    return metrics.iloc[-1]


def _load_model_rows(protocol_root, protocol_id, model):
    model_dir = protocol_root / ("qsvm" if model == "qsvm" else "merlin")
    metric_paths = sorted(model_dir.rglob("metrics_summary.csv"))
    if not metric_paths:
        raise FileNotFoundError(f"No metrics_summary.csv under {model_dir}")

    expected = PROTOCOLS[protocol_id]
    rows = []
    for metrics_path in metric_paths:
        info_path = metrics_path.parent / "dataset_info.json"
        if not info_path.is_file():
            raise FileNotFoundError(f"Missing {info_path}")
        info = json.loads(info_path.read_text())
        metrics = _select_metrics_row(metrics_path)

        if model == "qsvm":
            seed = int(info["seed"])
            q = int(info["qubits"])
            trace_protocol = str(metrics["trace_protocol"])
            kernel_normalization = ""
        else:
            seed = int(metrics["seed"])
            q = int(metrics["pca_dim"])
            trace_protocol = ""
            kernel_normalization = str(metrics["kernel_normalization"])

        preprocessing_protocol = str(metrics["preprocessing_protocol"])
        if preprocessing_protocol != expected["preprocessing_protocol"]:
            raise ValueError(
                f"{metrics_path} has preprocessing_protocol="
                f"{preprocessing_protocol}, expected "
                f"{expected['preprocessing_protocol']}"
            )
        if model == "qsvm" and trace_protocol != expected["trace_protocol"]:
            raise ValueError(
                f"{metrics_path} has trace_protocol={trace_protocol}, "
                f"expected {expected['trace_protocol']}"
            )
        if (
            model == "merlin_fidelity"
            and kernel_normalization != expected["kernel_normalization"]
        ):
            raise ValueError(
                f"{metrics_path} has kernel_normalization={kernel_normalization}, "
                f"expected {expected['kernel_normalization']}"
            )

        rows.append(
            {
                "protocol_id": protocol_id,
                "preprocessing_protocol": preprocessing_protocol,
                "trace_protocol": trace_protocol,
                "kernel_normalization": kernel_normalization,
                "seed": seed,
                "q": q,
                "model": model,
                "model_f1": float(metrics["test_minority_f1"]),
            }
        )

    frame = pd.DataFrame(rows)
    if frame.duplicated(["seed", "q"]).any():
        raise ValueError(f"Duplicate (seed, q) rows under {model_dir}")
    return frame


def _load_baseline_rows(result_root, classical_dir, baseline, kernel):
    metrics_path = (
        result_root / "classical" / classical_dir / baseline / "metrics_summary.csv"
    )
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Missing {metrics_path}")
    metrics = pd.read_csv(metrics_path)
    metrics = metrics[metrics["kernel"] == kernel].copy()
    if metrics.empty:
        raise ValueError(f"No kernel={kernel} rows in {metrics_path}")
    if metrics.duplicated(["seed", "pca_dim"]).any():
        raise ValueError(f"Duplicate (seed, q) rows in {metrics_path}")
    return metrics.rename(
        columns={
            "pca_dim": "q",
            "test_minority_f1": "baseline_f1",
        }
    )[["seed", "q", "baseline_f1"]]


def _pair_model_and_baseline(model_rows, baseline_rows, baseline):
    model_keys = set(map(tuple, model_rows[["seed", "q"]].to_numpy()))
    baseline_keys = set(map(tuple, baseline_rows[["seed", "q"]].to_numpy()))
    if model_keys != baseline_keys:
        missing_baseline = sorted(model_keys - baseline_keys)
        missing_model = sorted(baseline_keys - model_keys)
        raise ValueError(
            f"Unpaired rows for baseline={baseline}: "
            f"missing_baseline={missing_baseline}, missing_model={missing_model}"
        )
    paired = model_rows.merge(
        baseline_rows,
        on=["seed", "q"],
        how="inner",
        validate="one_to_one",
    )
    paired["baseline"] = baseline
    paired["delta_f1"] = paired["model_f1"] - paired["baseline_f1"]
    return paired


def aggregate_protocols(result_root, tolerance=1e-12):
    """Load and aggregate every protocol directory under ``result_root``."""
    per_seed_frames = []
    protocols_root = result_root / "protocols"
    protocol_dirs = sorted(path for path in protocols_root.iterdir() if path.is_dir())
    if not protocol_dirs:
        raise FileNotFoundError(f"No protocol directories under {protocols_root}")

    for protocol_root in protocol_dirs:
        protocol_id = protocol_root.name
        if protocol_id not in PROTOCOLS:
            raise ValueError(f"Unsupported protocol directory: {protocol_id}")
        protocol = PROTOCOLS[protocol_id]
        for model in ("qsvm", "merlin_fidelity"):
            model_rows = _load_model_rows(protocol_root, protocol_id, model)
            for baseline, (baseline_dir, kernel) in BASELINES.items():
                baseline_rows = _load_baseline_rows(
                    result_root,
                    protocol["classical_dir"],
                    baseline_dir,
                    kernel,
                )
                per_seed_frames.append(
                    _pair_model_and_baseline(model_rows, baseline_rows, baseline)
                )

    per_seed = pd.concat(per_seed_frames, ignore_index=True).sort_values(
        ["protocol_id", "q", "model", "baseline", "seed"]
    )

    summary_rows = []
    group_columns = [
        "protocol_id",
        "preprocessing_protocol",
        "trace_protocol",
        "kernel_normalization",
        "q",
        "model",
        "baseline",
    ]
    for keys, group in per_seed.groupby(group_columns, dropna=False, sort=True):
        comparison = summarize_deltas(
            group["model_f1"],
            group["baseline_f1"],
            tolerance=tolerance,
        )
        row = dict(zip(group_columns, keys))
        row.update(
            {
                "mean_f1": float(group["model_f1"].mean()),
                "std_f1": float(group["model_f1"].std(ddof=1)),
                "baseline_mean_f1": float(group["baseline_f1"].mean()),
                "baseline_std_f1": float(group["baseline_f1"].std(ddof=1)),
                "delta_f1": comparison["mean_delta"],
                "wins": comparison["wins"],
                "ties": comparison["ties"],
                "losses": comparison["losses"],
                "seeds": int(group["seed"].nunique()),
            }
        )
        summary_rows.append(row)

    return per_seed, pd.DataFrame(summary_rows)


def _format_markdown(summary, tolerance):
    columns = [
        "protocol_id",
        "preprocessing_protocol",
        "trace_protocol",
        "kernel_normalization",
        "q",
        "model",
        "baseline",
        "mean_f1",
        "std_f1",
        "baseline_mean_f1",
        "baseline_std_f1",
        "delta_f1",
        "wins",
        "ties",
        "losses",
        "seeds",
    ]
    lines = [
        "# Protocol sensitivity summary",
        "",
        "Perspective: `delta_f1 = model F1 - baseline F1`; wins and losses "
        "use the same first-named-model perspective.",
        "",
        f"W/T/L is counted seed by seed with `np.isclose(..., atol={tolerance}, "
        "rtol=0)`.",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in summary[columns].iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, (float, np.floating)):
                values.append("" if pd.isna(value) else f"{value:.6f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result_root", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=1e-12)
    args = parser.parse_args()

    per_seed, summary = aggregate_protocols(args.result_root, args.tolerance)
    per_seed_path = args.result_root / "protocol_results_per_seed.csv"
    summary_path = args.result_root / "protocol_summary.csv"
    markdown_path = args.result_root / "protocol_summary.md"
    per_seed.to_csv(per_seed_path, index=False)
    summary.to_csv(summary_path, index=False)
    markdown_path.write_text(_format_markdown(summary, args.tolerance))
    print(f"Saved: {per_seed_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {markdown_path}")


if __name__ == "__main__":
    main()
