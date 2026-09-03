"""Build the README result tables programmatically from run artefacts.

Usage
-----
    python utils/make_tables.py --table2 outdir/run_YYYYMMDD-HHMMSS \
        [--instance-sweep outdir/run_...] [--photonic outdir/run_...] \
        [--feature-selection outdir/run_...] --out results

Every number in the README comes from this script reading `metrics.json` /
`sweep_summary.json`; nothing is transcribed by hand.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

# Paper Table II, transcribed once from arXiv:2505.13933v2 for side-by-side
# comparison. Keys are (horizon, model).
PAPER_TABLE2 = {
    1: {"HAR": (0.1476, 2.0431), "HARX": (0.1508, 2.2436), "AR1": (0.1304, 1.7279),
        "AR3": (0.1178, 1.5893), "ARMAX": (0.1145, 1.6196), "LSTM": (0.1295, 1.7909),
        "LSTMX": (0.1185, 1.7571), "RC": (0.1441, 2.1011), "RCX": (0.1089, 1.6480),
        "QR1": (0.1050, 1.4427), "QR2": (0.1030, 1.4004)},
    5: {"HAR": (0.2143, 2.9041), "HARX": (0.2934, 4.5800), "AR1": (0.2642, 3.4136),
        "AR3": (0.2134, 2.8369), "ARMAX": (0.2134, 3.0703), "LSTM": (0.1831, 2.4600),
        "LSTMX": (0.2200, 3.4512), "RC": (0.1528, 2.0551), "RCX": (0.1667, 2.4605),
        "QR1": (0.1556, 2.1518), "QR2": (0.1663, 2.2332)},
}
PAPER_MCS = {
    1: {"ARMAX": 0.4406, "LSTMX": 0.4406, "RCX": 0.6086, "QR1": 0.7603, "QR2": 1.0000,
        "HAR": 0.0004, "HARX": 0.0004, "AR1": 0.0065, "AR3": 0.0936, "LSTM": 0.0221,
        "RC": 0.0084},
    5: {"HAR": 0.1291, "HARX": 0.0044, "AR1": 0.0938, "AR3": 0.1302, "ARMAX": 0.1302,
        "LSTM": 0.1291, "LSTMX": 0.0925, "RC": 1.0000, "RCX": 0.6333, "QR1": 0.7642,
        "QR2": 0.6333},
}


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def table2(run_dir: Path) -> dict[int, pd.DataFrame]:
    """Reproduced-vs-paper comparison table per horizon."""
    metrics = _load(run_dir / "metrics.json")
    out = {}
    for horizon_key, payload in metrics["horizons"].items():
        horizon = int(horizon_key)
        rows = []
        for name, scores in payload["models"].items():
            paper = PAPER_TABLE2[horizon].get(name, (None, None))
            rows.append({
                "model": name,
                "mse": scores["mse"],
                "paper_mse": paper[0],
                "qlike": scores["qlike"],
                "paper_qlike": paper[1],
                "mcs_p_mse": scores["mcs_p_mse"],
                "paper_mcs_p": PAPER_MCS[horizon].get(name),
                "hit_rate": scores["hit_rate"],
                "n": scores["n_observations"],
            })
        out[horizon] = pd.DataFrame(rows).sort_values("mse").reset_index(drop=True)
    return out


def instance_sweep(run_dir: Path) -> pd.DataFrame:
    """Per-variant distribution over reservoir instances."""
    summary = _load(run_dir / "sweep_summary.json")
    rows = []
    for variant, entry in summary["per_variant"].items():
        rows.append({
            "variant": variant,
            "n_completed": entry["n_completed"],
            "n_expected": entry["n_expected"],
            "mse_mean": entry["test_mse"]["mean"],
            "mse_std": entry["test_mse"]["std"],
            "mse_median": entry["test_mse"]["median"],
            "mse_min": entry["test_mse"]["min"],
            "mse_max": entry["test_mse"]["max"],
            "mse_q05": entry["test_mse"]["q05"],
            "best_on_test_instance": entry["best_on_test"]["instance"],
            "best_on_test_mse": entry["best_on_test"]["test_mse"],
            "val_selected_instance": entry["selected_on_validation"]["instance"],
            "val_selected_test_mse": entry["selected_on_validation"]["test_mse"],
            "val_selected_test_qlike": entry["selected_on_validation"]["test_qlike"],
        })
    return pd.DataFrame(rows)


def photonic(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Photonic per-variant summary, per-scale breakdown and hardware metadata."""
    summary = _load(run_dir / "sweep_summary.json")
    variant_rows, scale_rows = [], []
    for variant, entry in summary["per_variant"].items():
        row = {
            "variant": variant,
            "n_completed": entry["n_completed"],
            "n_expected": entry["n_expected"],
            "mse_mean": entry["test_mse"]["mean"],
            "mse_std": entry["test_mse"]["std"],
            "mse_min": entry["test_mse"]["min"],
            "best_on_test_mse": entry["best_on_test"]["test_mse"],
            "best_on_test_scale_divisor": entry["best_on_test"]["encoding_scale_divisor"],
            "val_selected_scale_divisor":
                entry["selected_on_validation"]["encoding_scale_divisor"],
            "val_selected_test_mse": entry["selected_on_validation"]["test_mse"],
            "val_selected_test_qlike": entry["selected_on_validation"]["test_qlike"],
            "mean_wall_clock_s": entry["mean_wall_clock_seconds"],
        }
        for key, value in entry.items():
            if key.startswith("test_mse_S"):
                row[f"{key}_at_val_selected"] = value["at_validation_selected"]
        variant_rows.append(row)
        for scale, block in summary["per_variant_scale"][variant].items():
            scale_rows.append({"variant": variant, "encoding_scale": scale, **block})
    return pd.DataFrame(variant_rows), pd.DataFrame(scale_rows), summary["hardware"]


def feature_selection(run_dir: Path) -> pd.DataFrame:
    """Greedy forward-selection path with both scoring splits."""
    summary = _load(run_dir / "sweep_summary.json")
    rows = []
    for variant, path in summary["paths"].items():
        for step in path:
            rows.append({
                "variant": variant, "k": step["step"], "added": step["added"],
                "features": ",".join(step["features"]),
                "test_mse": step["test_mse"], "test_qlike": step["test_qlike"],
                "validation_mse": step["validation_mse"],
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table2", type=Path, required=True)
    parser.add_argument("--instance-sweep", type=Path)
    parser.add_argument("--photonic", type=Path)
    parser.add_argument("--feature-selection", type=Path)
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    for horizon, frame in table2(args.table2).items():
        frame.to_csv(args.out / f"table2_S{horizon}.csv", index=False)
        print(f"=== Table II, S={horizon} ===")
        print(frame.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    if args.instance_sweep:
        frame = instance_sweep(args.instance_sweep)
        frame.to_csv(args.out / "instance_sweep_summary.csv", index=False)
        print("=== Reservoir-instance sweep ===")
        print(frame.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    if args.photonic:
        variants, scales, hardware = photonic(args.photonic)
        variants.to_csv(args.out / "photonic_summary.csv", index=False)
        scales.to_csv(args.out / "photonic_by_scale.csv", index=False)
        with (args.out / "photonic_hardware.json").open("w", encoding="utf-8") as handle:
            json.dump(hardware, handle, indent=2, sort_keys=True)
        print("=== Photonic (MerLin) summary ===")
        print(variants.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        print(scales.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    if args.feature_selection:
        frame = feature_selection(args.feature_selection)
        frame.to_csv(args.out / "feature_selection_path.csv", index=False)
        print("=== Forward feature selection ===")
        print(frame.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
