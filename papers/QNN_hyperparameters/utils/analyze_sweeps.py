"""Generate the predeclared result tables of a sweep from its structured artifacts.

Usage (from the repository root)::

    python papers/QNN_hyperparameters/utils/analyze_sweeps.py mechanism \
        papers/QNN_hyperparameters/outdir/sweeps/E-B_mechanism

Writes ``analysis_<kind>.json`` and CSV tables into the sweep directory and a
copy into ``papers/QNN_hyperparameters/results/``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR.parent))

from QNN_hyperparameters.lib.reanalysis import (  # noqa: E402
    load_released_results,
    paired_factor_tests,
    unpaired_factor_tests,
)
from QNN_hyperparameters.lib.sweep_analysis import (  # noqa: E402
    baseline_summary,
    cell_means,
    grid_agreement,
    grid_as_released,
    load_sweep,
    mechanism_effects,
    noise_effect,
    photonic_effects,
    qnn_reference,
)

RELEASED_RESULTS = (
    PROJECT_DIR.parents[1]
    / "data"
    / "QNN_hyperparameters"
    / "sources"
    / "author_repo"
    / "reports"
    / "results"
)


def analyze_mechanism(sweep_dir: Path) -> dict:
    """E-B tables: coverage, cell means and paired scaling / init effects."""

    frame, coverage = load_sweep(sweep_dir)
    if not coverage["complete"]:
        raise SystemExit(f"PARTIAL SWEEP, refusing an unqualified analysis: {coverage}")
    tables = {
        "cell_means_val": cell_means(frame, ["data", "scaling", "init"]),
        "cell_means_test": cell_means(
            frame, ["data", "scaling", "init"], metric="test_accuracy"
        ),
        "effects_val": mechanism_effects(frame, metric="val_accuracy"),
        "effects_test": mechanism_effects(frame, metric="test_accuracy"),
    }
    return {"coverage": coverage, "verdict": "COMPLETE SWEEP", "tables": tables}


def analyze_baselines(sweep_dir: Path) -> dict:
    """E-F tables: classical accuracy per cell, next to the released QNN grid."""

    frame, coverage = load_sweep(sweep_dir)
    if not coverage["complete"]:
        raise SystemExit(f"PARTIAL SWEEP, refusing an unqualified analysis: {coverage}")
    summary = baseline_summary(frame)
    reference = qnn_reference(load_released_results(RELEASED_RESULTS))
    comparison = summary.merge(reference, on=["data", "scaling"], how="left")
    comparison["val_minus_qnn_best"] = comparison["val_mean"] - comparison["qnn_best"]
    return {
        "coverage": coverage,
        "verdict": "COMPLETE SWEEP",
        "tables": {"baselines_vs_qnn": comparison},
    }


def _author_tests(results: pd.DataFrame) -> list[dict]:
    """The author notebook tests (as in E-A2), one row per p-value."""

    rows = []
    for factor, levels, better in (
        ("init", ["beta", "uniform"], "beta"),
        ("optimizer", ["cobyla", "spsa", "neldermead"], None),
    ):
        for test in paired_factor_tests(results, factor, levels, better):
            if "friedman_p" in test:
                rows.append(
                    {"factor": factor, "test": "friedman", "p": test["friedman_p"]}
                )
            for name, pair in test["pairwise"].items():
                rows.append(
                    {
                        "factor": factor,
                        "test": f"wilcoxon {name}",
                        "p": pair["p_value"],
                        "mean_diff": pair["mean_diff"],
                        "n": pair["n_pairs"],
                    }
                )
    for factor in ("ansatz", "featuremap", "entanglement", "entanglement_featuremap"):
        for test in unpaired_factor_tests(results, factor):
            rows.append(
                {
                    "factor": factor,
                    "test": test["test"],
                    "p": test["p_value"],
                    "means": json.dumps(test["means"]),
                }
            )
    return rows


def analyze_grid(sweep_dir: Path) -> dict:
    """E-C tables: agreement with the released Rice grid and the author tests."""

    frame, coverage = load_sweep(sweep_dir)
    if not coverage["complete"]:
        raise SystemExit(f"PARTIAL SWEEP, refusing an unqualified analysis: {coverage}")
    released = load_released_results(RELEASED_RESULTS)
    ours = grid_as_released(frame, released)
    reference = released[
        (released["dataset"] == "rice")
        & ~released["noisy"]
        & released["init"].isin(ours["init"].unique())
    ]
    tests = pd.DataFrame(_author_tests(ours)).merge(
        pd.DataFrame(_author_tests(reference)),
        on=["factor", "test"],
        how="outer",
        suffixes=("_ours", "_released"),
    )
    return {
        "coverage": coverage,
        "verdict": "COMPLETE SWEEP",
        "tables": {
            "agreement": grid_agreement(ours, released),
            "author_tests": tests,
            "per_config": ours,
        },
    }


def analyze_noisy(sweep_dir: Path, noiseless_dir: Path) -> dict:
    """E-D tables: agreement with the released noisy Rice runs and noise effect.

    ``noiseless_dir`` is the noiseless sweep run with the same PCA sign
    convention (E-C for E-D, E-C2 for E-D2).
    """

    frame, coverage = load_sweep(sweep_dir)
    noiseless_frame, noiseless_coverage = load_sweep(noiseless_dir)
    if not (coverage["complete"] and noiseless_coverage["complete"]):
        raise SystemExit(
            f"PARTIAL SWEEP, refusing an unqualified analysis: {coverage} {noiseless_coverage}"
        )
    released = load_released_results(RELEASED_RESULTS)
    noisy = grid_as_released(frame, released, noisy=True)
    noiseless = grid_as_released(noiseless_frame, released)
    return {
        "coverage": coverage,
        "verdict": "COMPLETE SWEEP",
        "tables": {
            "agreement": grid_agreement(noisy, released, noisy=True),
            "noise_effect": noise_effect(noisy, noiseless, released),
            "per_config": noisy,
        },
    }


def analyze_photonic(sweep_dir: Path) -> dict:
    """E-P tables (Phase 4 photonic-native adaptation): cell means and effects."""

    frame, coverage = load_sweep(sweep_dir)
    if not coverage["complete"]:
        raise SystemExit(f"PARTIAL SWEEP, refusing an unqualified analysis: {coverage}")
    return {
        "coverage": coverage,
        "verdict": "COMPLETE SWEEP",
        "tables": {
            "cell_means_val": cell_means(frame, ["data", "scaling", "init"]),
            "cell_means_photons_val": cell_means(
                frame, ["data", "scaling", "photons", "depth"]
            ),
            "effects_val": photonic_effects(frame, metric="val_accuracy"),
            "effects_test": photonic_effects(frame, metric="test_accuracy"),
        },
    }


ANALYSES = {
    "mechanism": analyze_mechanism,
    "baselines": analyze_baselines,
    "grid": analyze_grid,
    "noisy": analyze_noisy,
    "photonic": analyze_photonic,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("kind", choices=sorted(ANALYSES))
    parser.add_argument("sweep_dir", type=Path)
    parser.add_argument(
        "--noiseless",
        type=Path,
        help="noisy only: the noiseless sweep with the same PCA sign convention",
    )
    args = parser.parse_args()

    if args.kind == "noisy":
        if args.noiseless is None:
            parser.error("noisy needs --noiseless <sweep_dir>")
        analysis = analyze_noisy(args.sweep_dir, args.noiseless)
    else:
        analysis = ANALYSES[args.kind](args.sweep_dir)
    results_dir = PROJECT_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    summary = {
        "coverage": analysis["coverage"],
        "verdict": analysis["verdict"],
        "tables": {},
    }
    for name, table in analysis["tables"].items():
        filename = f"{args.sweep_dir.name}_{name}.csv"
        for directory in (args.sweep_dir, results_dir):
            table.to_csv(directory / filename, index=False)
        summary["tables"][name] = filename
    text = json.dumps(summary, indent=2)
    for directory in (args.sweep_dir, results_dir):
        (directory / f"{args.sweep_dir.name}_analysis_{args.kind}.json").write_text(
            text, encoding="utf-8"
        )
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
