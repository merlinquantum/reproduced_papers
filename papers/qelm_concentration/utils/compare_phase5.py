"""Phase 5 fair comparison: QELM (qubit and MerLin) against matched classical baselines.

Usage (from the paper folder)::

    python utils/compare_phase5.py \
        --landsat-qelm LABEL::MACHINE::RUN[+RUN] ... \
        --landsat-classical MACHINE::RUN ... \
        --narma LABEL::MACHINE::RUN ... \
        --shots-pairs PHOTONIC_LABEL::QUBIT_LABEL ...

Implements the frozen protocol of ``LOG.md`` section ``Phase 5 Comparison Plan``:

- Landsat: for each QELM run, the primary readouts ``w1_et_scaled`` and ``w2_et_scaled`` are
  paired with the classical run of the same ``input_scaling`` at the same readout dimension
  (random Fourier features with feature noise 0.01 as the primary baseline). The 9 QELM
  evaluations of a split (3 reservoirs x 3 shot seeds) and the 3 ELM inits are averaged per
  split; differences are paired over the 5 splits.
- NARMA: cumulative test R^2 over n = 1..L under 1e5 shots, averaged over the 3 reservoirs of
  a sequence, paired over the 5 sequences with the baselines of the same run.
- Decision rule: "clearly above" when the mean paired difference exceeds the tie tolerance
  (0.01 F1; 1 % of the baseline for NARMA) and 2 standard errors; otherwise "no clear
  separation".

Every run must be COMPLETED, every included metric finite, and every expected record present
exactly once. Writes ``results/phase5_comparison.{json,md}`` and
``results/phase5_comparison.png``; exits non-zero on a validation failure.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PAPER_DIR = Path(__file__).resolve().parents[1]
PRIMARY_READOUTS = ("w1_et_scaled", "w2_et_scaled")
PRIMARY_BASELINE = ("fourier", 0.01)
SECONDARY_BASELINES = (("fourier", 0.0), ("tanh", 0.01), ("tanh", 0.0))
LANDSAT_TIE = 0.01
NARMA_RELATIVE_TIE = 0.01
NARMA_SHOT_CONDITIONS = ("shots_1e5", "shadow_1e5_shots")
NARMA_BASELINES = ("matched_elm", "raw_window", "degree2_window", "esn")
# Paper-reported values (Fig. 4, read off the plot to +-0.003; 124 qubits on ibm_quebec).
PAPER_LANDSAT = {
    "QELM w2 ET-cut scaled (124 q, 930 features)": 0.830,
    "QELM w1 ET scaled (124 q, 372 features)": 0.812,
    "RF (untuned)": 0.840,
    "kNN (untuned)": 0.834,
    "MLP (untuned)": 0.830,
    "RBF-SVM (untuned)": 0.809,
    "Ridge on raw features": 0.608,
}


class ValidationError(RuntimeError):
    """A run or metric violates the frozen protocol."""


def _load_run(run_dir: Path) -> tuple[dict, dict, dict]:
    status = json.loads((run_dir / "run_status.json").read_text())
    if status["status"] != "COMPLETED":
        raise ValidationError(f"{run_dir} is {status['status']}")
    metrics = json.loads((run_dir / status["metrics_path"]).read_text())
    config = json.loads((run_dir / status["config_path"]).read_text())
    return status, metrics, config


def _input_scaling(config: dict) -> tuple[str, bool]:
    """Input scaling of a run and whether it was inferred.

    Snapshots written before the ``input_scaling`` option existed used min-max scaling, the
    only behaviour at the time (same rule as ``landsat_task.load_saved_probabilities``).
    """
    landsat = config["dataset"]["landsat"]
    if "input_scaling" in landsat:
        return landsat["input_scaling"], False
    return "minmax", True


def _seconds(status: dict) -> float:
    parse = lambda stamp: datetime.fromisoformat(stamp.replace("Z", "+00:00"))  # noqa: E731
    return (parse(status["completed_at"]) - parse(status["started_at"])).total_seconds()


def _spec(text: str, n_parts: int) -> list[str]:
    parts = text.split("::")
    if len(parts) != n_parts:
        raise ValueError(f"Expected {n_parts} '::'-separated fields in {text!r}")
    return parts


def _paired(a: dict, b: dict, tie: float) -> dict:
    keys = sorted(a)
    if keys != sorted(b):
        raise ValidationError(f"Unpaired keys: {keys} vs {sorted(b)}")
    diff = np.array([a[k] - b[k] for k in keys])
    stderr = float(diff.std(ddof=1) / np.sqrt(len(diff))) if len(diff) > 1 else np.inf
    mean = float(diff.mean())
    if mean > tie and mean > 2 * stderr:
        verdict = "QELM clearly above"
    elif -mean > tie and -mean > 2 * stderr:
        verdict = "baseline clearly above"
    else:
        verdict = "no clear separation"
    return {
        "n": len(diff),
        "mean_difference": mean,
        "stderr": stderr,
        "n_positive": int((diff > 0).sum()),
        "tie_tolerance": tie,
        "verdict": verdict,
    }


def _mean_std(values) -> dict:
    values = np.asarray(list(values), dtype=float)
    return {"mean": float(values.mean()), "std": float(values.std()), "n": len(values)}


def landsat_qelm_system(label: str, machine: str, run_dirs: list[Path]) -> dict:
    """Per-split means of the primary readouts and the shots sweep of one QELM system."""
    seen = set()
    per_split = defaultdict(lambda: defaultdict(list))
    shots = defaultdict(lambda: defaultdict(list))
    dims, scalings, seconds, kinds = {}, set(), 0.0, set()
    any_inferred = False
    for run_dir in run_dirs:
        status, metrics, config = _load_run(run_dir)
        seconds += _seconds(status)
        scaling, inferred = _input_scaling(config)
        scalings.add(scaling)
        any_inferred = any_inferred or inferred
        kinds.add("MerLin adaptation" if "hardware" in metrics else "qubit QELM")
        for record in metrics["records"]:
            if record["analysis"] == "shots_sweep":
                shots[record["shots"]][record["split_seed"]].append(record["f1_macro"])
                continue
            if (
                record["analysis"] != "readouts"
                or record["readout"] not in PRIMARY_READOUTS
            ):
                continue
            key = (
                record["readout"],
                record["split_seed"],
                record["reservoir_seed"],
                record["shot_seed"],
            )
            if key in seen:
                raise ValidationError(f"{label}: duplicate record {key}")
            if not np.isfinite(record["f1_macro"]):
                raise ValidationError(f"{label}: non-finite F1 for {key}")
            seen.add(key)
            dims[record["readout"]] = int(record["n_features"])
            per_split[record["readout"]][record["split_seed"]].append(
                record["f1_macro"]
            )
    if len(scalings) != 1 or len(kinds) != 1:
        raise ValidationError(f"{label}: mixed scalings {scalings} or kinds {kinds}")
    readouts = {}
    for readout in PRIMARY_READOUTS:
        counts = {split: len(v) for split, v in per_split[readout].items()}
        if len(counts) != 5 or set(counts.values()) != {9}:
            raise ValidationError(
                f"{label} {readout}: expected 5 splits x 9, got {counts}"
            )
        split_means = {s: float(np.mean(v)) for s, v in per_split[readout].items()}
        readouts[readout] = {
            "n_features": dims[readout],
            "runs": sum(counts.values()),
            "expected_runs": 45,
            "split_means": split_means,
            "f1_macro": _mean_std(split_means.values()),
        }
    shots_curve = {
        int(n): {s: float(np.mean(v)) for s, v in by_split.items()}
        for n, by_split in sorted(shots.items())
    }
    return {
        "label": label,
        "kind": kinds.pop(),
        "machine": machine,
        "input_scaling": scalings.pop(),
        "input_scaling_inferred": any_inferred,
        "runs": [str(r) for r in run_dirs],
        "wall_clock_s": seconds,
        "readouts": readouts,
        "shots_curve": shots_curve,
    }


def landsat_classical_system(machine: str, run_dir: Path) -> dict:
    """Per-split ELM means by (family, dimension, noise), plus the reference models."""
    status, metrics, config = _load_run(run_dir)
    splits = metrics["splits"]
    if sorted(splits) != ["0", "1", "2", "3", "4"]:
        raise ValidationError(f"{run_dir}: splits {sorted(splits)}")
    elm_cfg = config["landsat_classical"]["matched_elm"]
    elm = {}
    for family in elm_cfg["families"]:
        for dim in elm_cfg["n_features"]:
            for noise in elm_cfg["feature_noise_std"]:
                label = f"{family}|n_features={dim}|noise_std={noise}"
                split_means = {
                    int(s): splits[s]["matched_elm"][label]["f1_macro_mean"]
                    for s in splits
                }
                if not all(np.isfinite(v) for v in split_means.values()):
                    raise ValidationError(f"{run_dir}: non-finite {label}")
                elm[(family, int(dim), float(noise))] = {
                    "split_means": split_means,
                    "f1_macro": _mean_std(split_means.values()),
                    "n_inits": int(elm_cfg["n_inits"]),
                }
    references = {
        f"{name} (untuned, raw)": _mean_std(
            splits[s]["paper_baselines"][name]["f1_macro_mean"] for s in splits
        )
        for name in config["landsat_classical"]["paper_baselines"]
    }
    references.update(
        {
            f"{name} (tuned, raw)": _mean_std(
                splits[s]["tuned"][name]["f1_macro"] for s in splits
            )
            for name in config["landsat_classical"]["tuned_baselines"]["grids"]
        }
    )
    references["ridge degree-2 (702 features)"] = _mean_std(
        splits[s]["ridge_degree2"]["f1_macro"] for s in splits
    )
    return {
        "machine": machine,
        "run": str(run_dir),
        "input_scaling": _input_scaling(config)[0],
        "input_scaling_inferred": _input_scaling(config)[1],
        "wall_clock_s": _seconds(status),
        "elm": elm,
        "references": references,
    }


def compare_landsat(qelm_systems: list[dict], classical: dict[str, dict]) -> list[dict]:
    rows = []
    for system in qelm_systems:
        baseline = classical.get(system["input_scaling"])
        if baseline is None:
            raise ValidationError(f"No classical run with {system['input_scaling']}")
        for readout, entry in system["readouts"].items():
            dim = entry["n_features"]
            comparisons = {}
            for family, noise in (PRIMARY_BASELINE, *SECONDARY_BASELINES):
                key = (family, dim, noise)
                if key not in baseline["elm"]:
                    raise ValidationError(f"Classical run lacks matched baseline {key}")
                comparisons[f"{family}|noise={noise}"] = {
                    "baseline_f1_macro": baseline["elm"][key]["f1_macro"],
                    "paired": _paired(
                        entry["split_means"],
                        baseline["elm"][key]["split_means"],
                        LANDSAT_TIE,
                    ),
                }
            rows.append(
                {
                    "system": system["label"],
                    "kind": system["kind"],
                    "readout": readout,
                    "n_features": dim,
                    "input_scaling": system["input_scaling"],
                    "qelm_f1_macro": entry["f1_macro"],
                    "runs": entry["runs"],
                    "expected_runs": entry["expected_runs"],
                    "primary": comparisons[
                        f"{PRIMARY_BASELINE[0]}|noise={PRIMARY_BASELINE[1]}"
                    ],
                    "comparisons": comparisons,
                    "status": "COMMENSURABLE",
                }
            )
    return rows


def compare_shots(qelm_systems: list[dict]) -> list[dict]:
    """Photonic vs qubit shots curves at matched size and scaling (partially commensurable)."""
    rows = []
    merlin = [s for s in qelm_systems if s["kind"] == "MerLin adaptation"]
    for photonic in merlin:
        partner = photonic.get("shots_partner")
        if partner is None:
            continue
        qubit = next(s for s in qelm_systems if s["label"] == partner)
        if qubit["input_scaling"] != photonic["input_scaling"]:
            raise ValidationError(
                f"Shots pair {photonic['label']}/{partner} scaling differs"
            )
        for shots in sorted(set(photonic["shots_curve"]) & set(qubit["shots_curve"])):
            rows.append(
                {
                    "photonic": photonic["label"],
                    "qubit": partner,
                    "shots": shots,
                    "photonic_f1": _mean_std(photonic["shots_curve"][shots].values()),
                    "qubit_f1": _mean_std(qubit["shots_curve"][shots].values()),
                    "paired": _paired(
                        photonic["shots_curve"][shots],
                        qubit["shots_curve"][shots],
                        LANDSAT_TIE,
                    ),
                    "status": "PARTIALLY_COMMENSURABLE",
                }
            )
    return rows


def narma_system(label: str, machine: str, run_dir: Path) -> dict:
    status, metrics, _ = _load_run(run_dir)
    qelm = defaultdict(list)
    baselines = defaultdict(dict)
    n_features = {}
    for record in metrics["records"]:
        if not np.isfinite(record["cumulative_r2"]):
            raise ValidationError(f"{label}: non-finite cumulative R^2")
        if record["model"] == "qelm":
            if record["condition"] in NARMA_SHOT_CONDITIONS:
                qelm[record["sequence_seed"]].append(record["cumulative_r2"])
                n_features["qelm"] = record["n_features"]
        elif record["model"] in NARMA_BASELINES:
            if record["sequence_seed"] in baselines[record["model"]]:
                raise ValidationError(f"{label}: duplicate {record['model']} record")
            baselines[record["model"]][record["sequence_seed"]] = record[
                "cumulative_r2"
            ]
            n_features[record["model"]] = record["n_features"]
    counts = {seq: len(v) for seq, v in qelm.items()}
    if len(counts) != 5 or set(counts.values()) != {3}:
        raise ValidationError(
            f"{label}: expected 5 sequences x 3 reservoirs, got {counts}"
        )
    qelm_means = {seq: float(np.mean(v)) for seq, v in qelm.items()}
    comparisons = {}
    for name in NARMA_BASELINES:
        reference = float(np.mean(list(baselines[name].values())))
        comparisons[name] = {
            "n_features": int(n_features[name]),
            "cumulative_r2": _mean_std(baselines[name].values()),
            "paired": _paired(
                qelm_means, baselines[name], NARMA_RELATIVE_TIE * abs(reference)
            ),
            "status": "COMMENSURABLE" if name != "esn" else "REFERENCE (true memory)",
        }
    return {
        "label": label,
        "kind": "MerLin adaptation" if "hardware" in metrics else "qubit QELM",
        "machine": machine,
        "run": str(run_dir),
        "window": int(metrics["window"]),
        "wall_clock_s": _seconds(status),
        "qelm_n_features": int(n_features["qelm"]),
        "qelm_runs": sum(counts.values()),
        "qelm_cumulative_r2": _mean_std(qelm_means.values()),
        "comparisons": comparisons,
    }


def _fmt(stat: dict) -> str:
    return f"{stat['mean']:.3f} ± {stat['std']:.3f}"


def write_markdown(report: dict, path: Path) -> None:
    lines = [
        "# Phase 5 fair comparison (programmatic; scope `partial`, simulated)",
        "",
        "Generated by `utils/compare_phase5.py` from structured run artifacts. Values are "
        "mean ± std over the 5 split (Landsat) or sequence (NARMA) means. Paired "
        "differences are QELM − baseline, mean ± standard error. Decision rule: clearly "
        "above when the difference exceeds the tie tolerance and 2 standard errors.",
        "",
        "## Landsat F1 macro at matched readout dimension",
        "",
        "Primary baseline: random Fourier features, same dimension, same ridge, feature "
        "noise 0.01, same input scaling and splits.",
        "",
        "| System | Role | Provenance | Scope | Successful / expected runs | Readout | Features "
        "| Scaling | QELM F1 | Matched RFF F1 | Paired difference | Splits QELM > RFF | "
        "Verdict | Status |",
        "|---|---|---|---|---:|---|---:|---|---:|---:|---:|---:|---|---|",
    ]
    for row in report["landsat"]:
        primary = row["primary"]
        role = (
            "adaptation" if row["kind"] == "MerLin adaptation" else "reproduced method"
        )
        lines.append(
            f"| {row['system']} | {role} | local execution | partial | {row['runs']} / "
            f"{row['expected_runs']} | {row['readout']} | {row['n_features']} | "
            f"{row['input_scaling']} | {_fmt(row['qelm_f1_macro'])} | "
            f"{_fmt(primary['baseline_f1_macro'])} | {primary['paired']['mean_difference']:+.3f} "
            f"± {primary['paired']['stderr']:.3f} | {primary['paired']['n_positive']}/5 | "
            f"{primary['paired']['verdict']} | {row['status']} |"
        )
    lines += [
        "",
        "Secondary baselines (paired difference QELM − baseline, verdict):",
        "",
        "| System | Readout | "
        + " | ".join(f"{f} noise {n}" for f, n in SECONDARY_BASELINES)
        + " |",
        "|---|---|" + "---|" * len(SECONDARY_BASELINES),
    ]
    for row in report["landsat"]:
        cells = []
        for family, noise in SECONDARY_BASELINES:
            comp = row["comparisons"][f"{family}|noise={noise}"]
            cells.append(
                f"{comp['baseline_f1_macro']['mean']:.3f}; "
                f"{comp['paired']['mean_difference']:+.3f} ({comp['paired']['verdict']})"
            )
        lines.append(
            f"| {row['system']} | {row['readout']} | " + " | ".join(cells) + " |"
        )
    lines += [
        "",
        "## Landsat references (not dimension-matched; same splits)",
        "",
        "| Model | "
        + " | ".join(f"{c['input_scaling']} run" for c in report["landsat_classical"])
        + " | Paper (Fig. 4) |",
        "|---|" + "---:|" * (len(report["landsat_classical"]) + 1),
    ]
    names = list(report["landsat_classical"][0]["references"])
    for name in names:
        cells = [_fmt(c["references"][name]) for c in report["landsat_classical"]]
        lines.append(f"| {name} | " + " | ".join(cells) + " | |")
    for name, value in PAPER_LANDSAT.items():
        lines.append(
            f"| {name} | "
            + " | ".join("—" for _ in report["landsat_classical"])
            + f" | {value:.3f} (paper-reported) |"
        )
    lines += [
        "",
        "## Photonic vs qubit QELM at fixed shots (PARTIALLY_COMMENSURABLE)",
        "",
        "Same splits, scaling and readout pipeline; the photonic w1 click readout vs the "
        "qubit w1 Pauli readout of the shots sweep. Feature counts and physical resources "
        "differ, so this is not a matched-capacity comparison.",
        "",
        "| Photonic | Qubit | Shots | Photonic F1 | Qubit F1 | Paired difference | Verdict |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in report["shots"]:
        lines.append(
            f"| {row['photonic']} | {row['qubit']} | {row['shots']:g} | "
            f"{_fmt(row['photonic_f1'])} | {_fmt(row['qubit_f1'])} | "
            f"{row['paired']['mean_difference']:+.3f} ± {row['paired']['stderr']:.3f} | "
            f"{row['paired']['verdict']} |"
        )
    lines += [
        "",
        "## NARMA cumulative test R² over n = 1..L (QELM under 1e5 shots)",
        "",
        "| System | Role | Window L | QELM features | Successful / expected runs | QELM | "
        + " | ".join(NARMA_BASELINES)
        + " |",
        "|---|---|---:|---:|---:|---:|" + "---|" * len(NARMA_BASELINES),
    ]
    for system in report["narma"]:
        cells = []
        for name in NARMA_BASELINES:
            comp = system["comparisons"][name]
            cells.append(
                f"{_fmt(comp['cumulative_r2'])} (M = {comp['n_features']}); "
                f"{comp['paired']['mean_difference']:+.2f} ± {comp['paired']['stderr']:.2f}, "
                f"{comp['paired']['verdict']}"
            )
        role = (
            "adaptation"
            if system["kind"] == "MerLin adaptation"
            else "reproduced method"
        )
        lines.append(
            f"| {system['label']} | {role} | {system['window']} | {system['qelm_n_features']} "
            f"| {system['qelm_runs']} / 15 | {_fmt(system['qelm_cumulative_r2'])} | "
            + " | ".join(cells)
            + " |"
        )
    lines += [
        "",
        "ESN-400 has true memory and is an upper reference, not a matched baseline.",
        "",
        "## Wall-clock time",
        "",
        "| Run | Workload | Machine | Wall-clock (s) |",
        "|---|---|---|---:|",
    ]
    for entry in report["wall_clock"]:
        lines.append(
            f"| `{entry['run']}` | {entry['workload']} | {entry['machine']} | "
            f"{entry['seconds']:.0f} |"
        )
    lines.append("")
    path.write_text("\n".join(lines))


def plot(report: dict, path: Path) -> None:
    fig, (ax_landsat, ax_narma) = plt.subplots(
        1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1.4, 1]}
    )
    for classical, style in zip(report["landsat_classical"], ("-", "--")):
        for family, colour in (("fourier", "#1f77b4"), ("tanh", "#7f7f7f")):
            points = sorted(
                (dim, entry["f1_macro"])
                for (fam, dim, noise), entry in classical["elm"].items()
                if fam == family and noise == 0.01
            )
            dims = [d for d, _ in points]
            means = np.array([s["mean"] for _, s in points])
            stds = np.array([s["std"] for _, s in points])
            ax_landsat.plot(
                dims,
                means,
                style,
                color=colour,
                label=f"{family} ELM, noise 0.01 ({classical['input_scaling']})",
            )
            ax_landsat.fill_between(
                dims, means - stds, means + stds, color=colour, alpha=0.1
            )
    systems = list(dict.fromkeys(row["system"] for row in report["landsat"]))
    colours = dict(zip(systems, plt.cm.tab10(np.arange(len(systems)) + 2)))
    for row in report["landsat"]:
        stat = row["qelm_f1_macro"]
        first_readout = row["readout"] == PRIMARY_READOUTS[0]
        ax_landsat.errorbar(
            row["n_features"],
            stat["mean"],
            yerr=stat["std"],
            fmt="s" if row["kind"] == "MerLin adaptation" else "o",
            color=colours[row["system"]],
            mfc="none" if row["input_scaling"] == "minmax" else None,
            capsize=3,
            label=f"{row['system']} ({row['kind']})" if first_readout else None,
        )
        ax_landsat.annotate(
            row["readout"][:2],
            (row["n_features"], stat["mean"]),
            textcoords="offset points",
            xytext=(4, 2),
            fontsize=7,
            color=colours[row["system"]],
        )
    ax_landsat.axhline(
        PAPER_LANDSAT["QELM w2 ET-cut scaled (124 q, 930 features)"],
        color="k",
        linestyle=":",
        linewidth=1,
        label="paper QELM 124 q, 930 features (paper-reported)",
    )
    ax_landsat.set_xscale("log")
    ax_landsat.set_xlabel("readout dimension (features into the same ridge)")
    ax_landsat.set_ylabel("test F1 macro (mean ± std over 5 splits)")
    ax_landsat.set_title(
        "Landsat, matched dimension (partial, simulated)\n"
        "o: qubit QELM, s: MerLin adaptation; hollow = min-max inputs; w1/w2 = readout",
        fontsize=9,
    )
    ax_landsat.legend(fontsize=6, loc="lower right", ncol=2)

    names = ["QELM", *NARMA_BASELINES]
    width = 0.8 / len(report["narma"])
    for index, system in enumerate(report["narma"]):
        values = [system["qelm_cumulative_r2"]] + [
            system["comparisons"][name]["cumulative_r2"] for name in NARMA_BASELINES
        ]
        means = [v["mean"] / system["window"] for v in values]
        stds = [v["std"] / system["window"] for v in values]
        ax_narma.bar(
            np.arange(len(names)) + index * width,
            means,
            width,
            yerr=stds,
            capsize=2,
            label=f"{system['label']} (L = {system['window']})",
        )
    ax_narma.set_xticks(np.arange(len(names)) + 0.4 - width / 2)
    ax_narma.set_xticklabels(
        ["QELM 1e5 shots", "matched ELM", "raw window", "degree-2", "ESN (ref.)"],
        fontsize=8,
    )
    ax_narma.set_ylabel("cumulative test R² / L (mean ± std over 5 sequences)")
    ax_narma.set_ylim(0.5, 1.02)
    ax_narma.set_title("NARMA n = 1..L (partial, simulated)", fontsize=9)
    ax_narma.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--landsat-qelm", nargs="+", required=True)
    parser.add_argument("--landsat-classical", nargs="+", required=True)
    parser.add_argument("--narma", nargs="+", required=True)
    parser.add_argument(
        "--shots-pairs",
        nargs="*",
        default=[],
        help="PHOTONIC_LABEL::QUBIT_LABEL pairs for the fixed-shots comparison",
    )
    parser.add_argument("--output-dir", type=Path, default=PAPER_DIR / "results")
    args = parser.parse_args()

    try:
        qelm_systems = []
        for text in args.landsat_qelm:
            label, machine, runs = _spec(text, 3)
            qelm_systems.append(
                landsat_qelm_system(label, machine, [Path(r) for r in runs.split("+")])
            )
        for text in args.shots_pairs:
            photonic, qubit = _spec(text, 2)
            next(s for s in qelm_systems if s["label"] == photonic)["shots_partner"] = (
                qubit
            )
        classical = {}
        for text in args.landsat_classical:
            machine, run = _spec(text, 2)
            system = landsat_classical_system(machine, Path(run))
            if system["input_scaling"] in classical:
                raise ValidationError(
                    f"Two classical runs with {system['input_scaling']}"
                )
            classical[system["input_scaling"]] = system
        narma = []
        for text in args.narma:
            label, machine, run = _spec(text, 3)
            narma.append(narma_system(label, machine, Path(run)))
        landsat_rows = compare_landsat(qelm_systems, classical)
        shots_rows = compare_shots(qelm_systems)
    except ValidationError as error:
        print(f"VALIDATION FAILED: {error}")
        sys.exit(1)

    wall_clock = [
        {
            "run": "+".join(s["runs"]),
            "workload": f"Landsat {s['kind']} {s['label']} (all readouts and sweeps)",
            "machine": s["machine"],
            "seconds": s["wall_clock_s"],
        }
        for s in qelm_systems
    ]
    wall_clock += [
        {
            "run": c["run"],
            "workload": f"Landsat classical baselines ({c['input_scaling']})",
            "machine": c["machine"],
            "seconds": c["wall_clock_s"],
        }
        for c in classical.values()
    ]
    wall_clock += [
        {
            "run": s["run"],
            "workload": f"NARMA {s['kind']} {s['label']} (QELM 3 conditions + baselines)",
            "machine": s["machine"],
            "seconds": s["wall_clock_s"],
        }
        for s in narma
    ]
    classical_out = [
        {
            **{k: v for k, v in c.items() if k != "elm"},
            "elm": {
                f"{fam}|n_features={dim}|noise_std={noise}": entry
                for (fam, dim, noise), entry in c["elm"].items()
            },
        }
        for c in classical.values()
    ]
    report = {
        "protocol": "LOG.md, Phase 5 Comparison Plan — 2026-10-28T14:23Z (+ amendment 14:25Z)",
        "landsat": landsat_rows,
        "landsat_classical": classical_out,
        "shots": shots_rows,
        "narma": narma,
        "wall_clock": wall_clock,
        "paper_reported": PAPER_LANDSAT,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "phase5_comparison.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    report_for_markdown = {**report, "landsat_classical": list(classical.values())}
    write_markdown(report_for_markdown, args.output_dir / "phase5_comparison.md")
    plot(report_for_markdown, args.output_dir / "phase5_comparison.png")
    for row in landsat_rows:
        print(
            f"{row['system']} {row['readout']} ({row['n_features']}): "
            f"{row['qelm_f1_macro']['mean']:.4f} vs RFF "
            f"{row['primary']['baseline_f1_macro']['mean']:.4f} -> "
            f"{row['primary']['paired']['verdict']}"
        )
    for system in narma:
        print(
            f"NARMA {system['label']}: {system['qelm_cumulative_r2']['mean']:.3f}; "
            + ", ".join(
                f"{n} {c['cumulative_r2']['mean']:.3f} ({c['paired']['verdict']})"
                for n, c in system["comparisons"].items()
            )
        )


if __name__ == "__main__":
    main()
