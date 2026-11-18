"""Build the Landsat comparison tables (E5, E10) from structured run artifacts.

Usage (from the paper folder)::

    python utils/summarize_landsat.py --readouts-run outdir/run_A --classical-run outdir/run_B \
        --label n16 --scope partial

Writes ``results/landsat_<label>_summary.json`` and ``results/landsat_<label>_summary.md``.
Every number comes from ``metrics.json``; aggregation is mean and std over all recorded
(split, reservoir, shot) seeds for QELM readouts, and over splits for classical baselines
(each split's value is itself the mean over model initialisations).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

PAPER_DIR = Path(__file__).resolve().parents[1]

# Paper Fig. 4 / text (124 qubits, F1 macro). "scaled" values are stated in the text;
# "unit" values and classical lines were read off the plot (about +-0.003).
PAPER_F1_MACRO = {
    "w1_x_scaled": 0.782,
    "w1_pauli_scaled": 0.800,
    "w1_et_scaled": 0.812,
    "w2_x_scaled": 0.789,
    "w2_pauli_scaled": 0.789,
    "w2_et_scaled": 0.801,
    "w2_etcut_scaled": 0.830,
    "w1_x_unit": 0.783,
    "w1_pauli_unit": 0.790,
    "w1_et_unit": 0.783,
    "w1_etcut_unit": 0.789,
    "w2_x_unit": 0.750,
    "w2_pauli_unit": 0.750,
    "w2_et_unit": 0.749,
    "w2_etcut_unit": 0.792,
}
PAPER_CLASSICAL_F1_MACRO = {
    "random_forest": 0.840,
    "knn": 0.834,
    "mlp": 0.830,
    "svm_rbf": 0.809,
    "naive_bayes": 0.782,
    "decision_tree": 0.776,
    "qda": 0.716,
    "adaboost": 0.699,
    "ridge": 0.608,
}
METRICS = (
    "f1_macro",
    "accuracy",
    "f1_weighted",
    "precision_macro",
    "precision_weighted",
)


def _mean_std(values) -> dict:
    array = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError("Non-finite metric in included runs")
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "n": int(array.size),
    }


def _require_completed(run_dir: Path) -> dict:
    status = json.loads((run_dir / "run_status.json").read_text())
    if status["status"] != "COMPLETED":
        raise ValueError(f"{run_dir} is {status['status']}, not COMPLETED")
    return json.loads((run_dir / status["metrics_path"]).read_text())


def summarise_readouts(metrics: dict) -> dict:
    """Headline readouts plus the cutoff, shots and train-size curves."""
    records = metrics["records"]
    summary: dict = {
        "readouts": {},
        "cutoff_sweep": {},
        "shots_sweep": {},
        "train_size_sweep": {},
    }
    for name in sorted({r["readout"] for r in records if r["analysis"] == "readouts"}):
        items = [
            r for r in records if r["analysis"] == "readouts" and r["readout"] == name
        ]
        entry = {metric: _mean_std([r[metric] for r in items]) for metric in METRICS}
        entry["n_features"] = _mean_std([r["n_features"] for r in items])
        if "n_eigentasks_retained" in items[0]:
            entry["n_eigentasks_retained"] = _mean_std(
                [r["n_eigentasks_retained"] for r in items]
            )
            entry["n_eigentasks_total"] = _mean_std(
                [r["n_eigentasks_total"] for r in items]
            )
        entry["paper_f1_macro"] = PAPER_F1_MACRO.get(name)
        summary["readouts"][name] = entry
    for analysis, key in (
        ("cutoff_sweep", "nsr_cutoff"),
        ("shots_sweep", "shots"),
        ("train_size_sweep", "train_size"),
    ):
        items = [r for r in records if r["analysis"] == analysis]
        for readout in sorted({r["readout"] for r in items}):
            curve = {}
            for value in sorted({r[key] for r in items if r["readout"] == readout}):
                group = [
                    r for r in items if r["readout"] == readout and r[key] == value
                ]
                point = {"f1_macro": _mean_std([r["f1_macro"] for r in group])}
                if analysis == "cutoff_sweep":
                    point["n_eigentasks_retained"] = _mean_std(
                        [r["n_eigentasks_retained"] for r in group]
                    )
                    point["n_eigentasks_total"] = _mean_std(
                        [r["n_eigentasks_total"] for r in group]
                    )
                    point["rec_cut"] = _mean_std([r["rec_cut"] for r in group])
                    point["rec_total"] = _mean_std([r["rec_total"] for r in group])
                    point["empty_readout_fraction"] = float(
                        np.mean([r["empty_readout"] for r in group])
                    )
                curve[str(value)] = point
            summary[analysis][readout] = curve
    return summary


# Paired comparisons: (A, B) differences F1(A) - F1(B) on identical (split, reservoir, shot)
# draws, matching the paper's readout claims (C3) and scaling statements.
PAIRED_COMPARISONS = (
    ("w1_pauli_scaled", "w1_x_scaled"),
    ("w1_et_scaled", "w1_pauli_scaled"),
    ("w2_et_scaled", "w2_pauli_scaled"),
    ("w2_etcut_scaled", "w2_et_scaled"),
    ("w2_etcut_scaled", "w1_et_scaled"),
    ("w1_pauli_scaled", "w1_pauli_unit"),
    ("w2_pauli_scaled", "w2_pauli_unit"),
    ("w2_et_scaled", "w2_et_unit"),
    ("w2_etcut_scaled", "w2_etcut_unit"),
    ("w2_pauli_unit", "w1_pauli_unit"),
)


def paired_differences(metrics: dict) -> dict:
    """Mean, std, and sign counts of paired F1-macro differences between readouts."""
    by_key: dict = {}
    for record in metrics["records"]:
        if record["analysis"] != "readouts":
            continue
        key = (record["split_seed"], record["reservoir_seed"], record["shot_seed"])
        by_key.setdefault(key, {})[record["readout"]] = record["f1_macro"]
    result = {}
    for first, second in PAIRED_COMPARISONS:
        diffs = np.array(
            [
                v[first] - v[second]
                for v in by_key.values()
                if first in v and second in v
            ]
        )
        if diffs.size == 0:
            continue
        stderr = (
            diffs.std(ddof=1) / np.sqrt(diffs.size) if diffs.size > 1 else float("nan")
        )
        result[f"{first} - {second}"] = {
            "mean": float(diffs.mean()),
            "std": float(diffs.std()),
            "stderr": float(stderr),
            "n": int(diffs.size),
            "n_positive": int(np.sum(diffs > 0)),
            "n_negative": int(np.sum(diffs < 0)),
        }
    return result


def summarise_classical(metrics: dict) -> dict:
    """Per-baseline mean and std over splits."""
    splits = list(metrics["splits"].values())
    summary: dict = {"paper_baselines": {}, "matched_elm": {}, "tuned": {}}
    for name in splits[0]["paper_baselines"]:
        summary["paper_baselines"][name] = {
            metric: _mean_std(
                [s["paper_baselines"][name][f"{metric}_mean"] for s in splits]
            )
            for metric in METRICS
        }
        summary["paper_baselines"][name]["n_inits"] = splits[0]["paper_baselines"][
            name
        ]["n_inits"]
        summary["paper_baselines"][name]["paper_f1_macro"] = (
            PAPER_CLASSICAL_F1_MACRO.get(name)
        )
    summary["ridge_degree2"] = {
        metric: _mean_std([s["ridge_degree2"][metric] for s in splits])
        for metric in METRICS
    }
    for label in splits[0]["matched_elm"]:
        summary["matched_elm"][label] = {
            metric: _mean_std(
                [s["matched_elm"][label][f"{metric}_mean"] for s in splits]
            )
            for metric in METRICS
        }
    for name in splits[0]["tuned"]:
        summary["tuned"][name] = {
            metric: _mean_std([s["tuned"][name][metric] for s in splits])
            for metric in METRICS
        }
        summary["tuned"][name]["selected_params"] = [
            s["tuned"][name]["params"] for s in splits
        ]
    return summary


def _fmt(entry: dict) -> str:
    return f"{entry['mean']:.3f} ± {entry['std']:.3f}"


def to_markdown(summary: dict) -> str:
    lines = [f"# Landsat summary ({summary['label']}, scope: {summary['scope']})", ""]
    lines += [f"Sources: {', '.join(summary['sources'].values())}", ""]
    if "qelm" in summary:
        lines += [
            "## QELM readouts (F1 macro, mean ± std over seeds)",
            "",
            "| Readout | F1 macro | Accuracy | Features | Retained ET | Paper F1 (124 q) |",
            "|---|---|---|---|---|---|",
        ]
        for name, entry in summary["qelm"]["readouts"].items():
            retained = (
                _fmt(entry["n_eigentasks_retained"])
                if "n_eigentasks_retained" in entry
                else ""
            )
            paper = (
                ""
                if entry["paper_f1_macro"] is None
                else f"{entry['paper_f1_macro']:.3f}"
            )
            lines.append(
                f"| {name} | {_fmt(entry['f1_macro'])} | {_fmt(entry['accuracy'])} | "
                f"{entry['n_features']['mean']:.0f} | {retained} | {paper} |"
            )
        lines.append("")
        lines += [
            "## Paired F1-macro differences (same split, reservoir and shot draw)",
            "",
            "| A - B | mean | std | stderr | n | A > B | A < B |",
            "|---|---|---|---|---|---|---|",
        ]
        for label, entry in summary["qelm"]["paired_differences"].items():
            lines.append(
                f"| {label} | {entry['mean']:+.4f} | {entry['std']:.4f} | "
                f"{entry['stderr']:.4f} | {entry['n']} | {entry['n_positive']} | "
                f"{entry['n_negative']} |"
            )
        lines.append("")
        for analysis in ("shots_sweep", "train_size_sweep", "cutoff_sweep"):
            for readout, curve in summary["qelm"][analysis].items():
                lines += [
                    f"## {analysis}: {readout}",
                    "",
                    "| value | F1 macro | extra |",
                    "|---|---|---|",
                ]
                for value, point in curve.items():
                    extra = ""
                    if "n_eigentasks_retained" in point:
                        extra = (
                            f"retained {point['n_eigentasks_retained']['mean']:.1f}/"
                            f"{point['n_eigentasks_total']['mean']:.0f}, REC "
                            f"{point['rec_cut']['mean']:.1f}/{point['rec_total']['mean']:.1f}, "
                            f"empty {point['empty_readout_fraction']:.2f}"
                        )
                    lines.append(f"| {value} | {_fmt(point['f1_macro'])} | {extra} |")
                lines.append("")
    if "classical" in summary:
        lines += [
            "## Classical baselines (F1 macro, mean ± std over splits)",
            "",
            "| Model | F1 macro | Accuracy | Paper F1 |",
            "|---|---|---|---|",
        ]
        for name, entry in summary["classical"]["paper_baselines"].items():
            paper = (
                ""
                if entry["paper_f1_macro"] is None
                else f"{entry['paper_f1_macro']:.3f}"
            )
            lines.append(
                f"| {name} (Table A2) | {_fmt(entry['f1_macro'])} | {_fmt(entry['accuracy'])} | {paper} |"
            )
        lines.append(
            f"| ridge_degree2 | {_fmt(summary['classical']['ridge_degree2']['f1_macro'])} | "
            f"{_fmt(summary['classical']['ridge_degree2']['accuracy'])} | |"
        )
        for label, entry in summary["classical"]["matched_elm"].items():
            lines.append(
                f"| elm {label} | {_fmt(entry['f1_macro'])} | {_fmt(entry['accuracy'])} | |"
            )
        for name, entry in summary["classical"]["tuned"].items():
            lines.append(
                f"| tuned {name} | {_fmt(entry['f1_macro'])} | {_fmt(entry['accuracy'])} | |"
            )
        lines.append("")
    return "\n".join(lines)


# Settings that must agree for runs to be merged; realisation selection, probability
# reuse, device and chunk size may differ.
MERGE_IGNORED_EXPERIMENT_KEYS = (
    "realisations",
    "probability_sources",
    "save_probabilities",
)


def _comparable_config(run_dir: Path) -> dict:
    cfg = json.loads((run_dir / "config_snapshot.json").read_text())
    reservoir = {
        k: v
        for k, v in cfg["reservoir"].items()
        if k not in ("chunk_size",) and not k.startswith("_")
    }
    experiment = {
        k: v
        for k, v in cfg["landsat_readouts"].items()
        if k not in MERGE_IGNORED_EXPERIMENT_KEYS and not k.startswith("_")
    }
    landsat = {"input_scaling": "minmax"} | {
        k: v for k, v in cfg["dataset"]["landsat"].items() if not k.startswith("_")
    }
    hyper = {k: v for k, v in cfg["hyperparameters"].items() if not k.startswith("_")}
    return {
        "reservoir": reservoir,
        "experiment": experiment,
        "landsat": landsat,
        "hyperparameters": hyper,
        "dtype": cfg["dtype"],
        "readout": cfg["readout"],
    }


def merge_readout_runs(run_dirs: list[Path]) -> dict:
    """Concatenate the records of comparable runs, rejecting duplicated evaluations.

    Raises
    ------
    ValueError
        If configurations differ beyond realisation selection, or a
        (analysis, readout, split, reservoir, shot, sweep value) record appears twice.
    """
    reference = _comparable_config(run_dirs[0])
    records: list[dict] = []
    for run_dir in run_dirs:
        if _comparable_config(run_dir) != reference:
            raise ValueError(f"{run_dir} is not comparable with {run_dirs[0]}")
        records.extend(_require_completed(run_dir)["records"])
    keys = [
        (
            r["analysis"],
            r["readout"],
            r["split_seed"],
            r["reservoir_seed"],
            r.get("shot_seed"),
            r.get("nsr_cutoff"),
            r.get("shots"),
            r.get("train_size"),
            r.get("resample"),
        )
        for r in records
    ]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate evaluations across merged runs")
    return {"records": records}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--readouts-run",
        type=Path,
        nargs="+",
        help="landsat_readouts run directories; several are merged after a comparability check",
    )
    parser.add_argument(
        "--classical-run", type=Path, help="landsat_classical run directory"
    )
    parser.add_argument("--label", required=True, help="Output name suffix, e.g. n16")
    parser.add_argument(
        "--scope", required=True, help="Result-scope label, e.g. partial or preliminary"
    )
    parser.add_argument("--outdir", type=Path, default=PAPER_DIR / "results")
    args = parser.parse_args()
    if args.readouts_run is None and args.classical_run is None:
        parser.error("give --readouts-run and/or --classical-run")

    summary: dict = {"label": args.label, "scope": args.scope, "sources": {}}
    if args.readouts_run is not None:
        readout_metrics = merge_readout_runs(args.readouts_run)
        summary["qelm"] = summarise_readouts(readout_metrics)
        summary["qelm"]["paired_differences"] = paired_differences(readout_metrics)
        summary["qelm"]["n_realisations"] = len(
            {(r["split_seed"], r["reservoir_seed"]) for r in readout_metrics["records"]}
        )
        summary["sources"]["readouts"] = ", ".join(r.name for r in args.readouts_run)
    if args.classical_run is not None:
        summary["classical"] = summarise_classical(
            _require_completed(args.classical_run)
        )
        summary["sources"]["classical"] = args.classical_run.name
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / f"landsat_{args.label}_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    (args.outdir / f"landsat_{args.label}_summary.md").write_text(
        to_markdown(summary) + "\n"
    )
    print(to_markdown(summary))


if __name__ == "__main__":
    main()
