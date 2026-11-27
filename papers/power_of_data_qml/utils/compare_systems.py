"""Phase 5 comparison table: paper method vs paper baseline vs MerLin adaptation.

Usage (from the paper folder)::

    python utils/compare_systems.py --gate results/study_extended.json.gz \
        --photonic results/study_photonic_extended.json.gz \
        --training-size results/training_size.json.gz --output results/comparison_extended

Implements the frozen protocol in LOG.md (Phase 5 Comparison Plan): CV protocol
for verdicts (paper protocol shown for comparability), mean over 10 seeds,
sample std, paired difference to the best classical model with a 95% t-interval,
and "better" only when that interval excludes 0 (otherwise TIE). Run coverage is
checked from the per-row seed lists; tuning budgets are counted from the
candidate builders; wall-clock comes from the per-record timings.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parents[1]
sys.path[:0] = [str(REPO_ROOT), str(PROJECT_DIR)]

from lib import models  # noqa: E402
from lib.result_io import dump_json, load_json  # noqa: E402

EXPECTED_SEEDS = list(range(10))
QUANTUM = {
    "quantum_kernel": "QK (fidelity)",
    "projected_quantum_kernel": "PQK (projected)",
}
SCOPE = {
    "gate": "partial (n ≤ 20 vs paper ≤ 30)",
    "photonic": "MerLin adaptation, simulated, k = floor(m/2) photons (n ≤ 18)",
    "photonic_capped": "MerLin adaptation, simulated, photons capped at k ≤ 8 (n = 17-20; separate regime)",
}
TASKS = [
    # (claim, component, train size, metric direction, sizes per family)
    ("C1/C10 quantum dataset (MAE)", "quantum_dataset", 600, "min"),
    ("C5/C6 PQK-engineered (accuracy)", "engineered_pqk", 600, "max"),
    ("C5/C6 PQK-engineered (accuracy)", "engineered_pqk", 100, "max"),
    ("C7 QK-engineered (accuracy)", "engineered_qk", 600, "max"),
]
SIZES = {
    "gate": (8, 12, 16, 18, 20),
    "photonic": (8, 12, 16, 18),
    "photonic_capped": (17, 18, 19, 20),
}


def tuning_budget() -> dict:
    """Number of hyperparameter configurations per family (paper grids)."""
    counts = {}
    for kind in ("regression", "classification"):
        keys = {
            "gaussian_kernel": [
                (f"rbf_{s}", {}) for s in models.PAPER_GRIDS["svm_gamma_scales"]
            ],
            "linear_kernel": [("linear", {})],
            "quantum_kernel": [("qk", {})],
            "projected_quantum_kernel": [
                (f"pqk_{s}", {}) for s in models.PAPER_GRIDS["svm_gamma_scales"]
            ],
        }
        per = {
            f: len(
                models.kernel_family_candidates(
                    f, kind, k, grid_name="paper", svr_epsilon=0.1
                )
            )
            for f, k in keys.items()
        }
        for c in models.feature_family_candidates(
            kind, grid_name="paper", random_state=0, mlp_max_iter=1
        ):
            per[c.family] = per.get(c.family, 0) + 1
        per["best_classical (6 paper families)"] = sum(
            per[f] for f in models.CLASSICAL_FAMILIES
        )
        counts[kind] = per
    return counts


def index(rows):
    return {
        (
            r["component"],
            r["quantity"],
            r["embedding"],
            r["n"],
            r["train_size"],
            r["protocol"],
        ): r
        for r in rows
    }


def unit_seconds(run_dirs):
    times = {}
    for run in run_dirs:
        for line in (Path(run) / "records.jsonl").read_text().splitlines():
            record = json.loads(line)
            times.setdefault((record.get("embedding"), record["n"]), []).append(
                record["elapsed_seconds"]
            )
    return {key: statistics.mean(values) for key, values in times.items()}


def run_seconds_by_machine(run_dirs):
    """Sum run wall-clock seconds, split by the device in each run's config snapshot."""
    totals = {}
    for run in run_dirs:
        device = json.loads((Path(run) / "config_snapshot.json").read_text())["device"]
        machine = "remote" if device.startswith("cuda") else "local"
        totals[machine] = (
            totals.get(machine, 0.0)
            + json.loads((Path(run) / "metrics.json").read_text())["elapsed_seconds"]
        )
    return totals


def verdict(direction, difference):
    """Quantum vs best classical under the frozen paired-CI rule."""
    if difference is None:
        return "N/A"
    low, high = (
        difference["mean"] - difference["ci95_half_width"],
        difference["mean"] + difference["ci95_half_width"],
    )
    quantum_better = high < 0 if direction == "min" else low > 0
    classical_better = low > 0 if direction == "min" else high < 0
    return (
        "QUANTUM BETTER"
        if quantum_better
        else "CLASSICAL BETTER"
        if classical_better
        else "TIE"
    )


def coverage(row):
    seeds = sorted(row["seeds"])
    if seeds != EXPECTED_SEEDS:
        raise ValueError(f"Incomplete or duplicate seeds {seeds} for {row}")
    return f"{len(seeds)}/{len(EXPECTED_SEEDS)}"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gate", required=True, type=Path)
    parser.add_argument("--photonic", required=True, type=Path)
    parser.add_argument(
        "--photonic-capped",
        type=Path,
        default=None,
        help="Optional aggregate of the capped-photon regime (reported as its own family)",
    )
    parser.add_argument("--training-size", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    sources = {
        "gate": load_json(args.gate),
        "photonic": load_json(args.photonic),
    }
    if args.photonic_capped is not None:
        sources["photonic_capped"] = load_json(args.photonic_capped)
    for name, aggregate in sources.items():
        if aggregate["coverage"] != "COMPLETE":
            raise ValueError(f"{name} aggregate coverage is {aggregate['coverage']}")
    budget = tuning_budget()
    seconds = {name: unit_seconds(agg["runs"]) for name, agg in sources.items()}
    rows_out = []
    for family, aggregate in sources.items():
        idx = index(aggregate["summary"])
        embeddings = sorted(
            {r["embedding"] for r in aggregate["summary"] if r["embedding"]}
        )
        for claim, component, train_size, direction in TASKS:
            kind = "regression" if component == "quantum_dataset" else "classification"
            for embedding in embeddings:
                for n in SIZES[family]:
                    for protocol in ("cv", "paper"):
                        best = idx.get(
                            (
                                component,
                                "best_classical",
                                embedding,
                                n,
                                train_size,
                                protocol,
                            )
                        )
                        if best is None:
                            continue
                        base = {
                            "claim": claim,
                            "family": family,
                            "embedding": embedding,
                            "n": n,
                            "N": train_size,
                            "protocol": protocol,
                            "scope": SCOPE[family],
                            "metric_direction": direction,
                            "unit_wall_clock_s": seconds[family].get((embedding, n)),
                        }
                        rows_out.append(
                            {
                                **base,
                                "system": "best classical (paper baseline, 6 families)",
                                "role": "paper baseline",
                                "provenance": "local",
                                "runs": coverage(best),
                                "mean": best["mean"],
                                "std": best["std"],
                                "paper_value": best.get("paper_value"),
                                "tuning_configs": budget[kind][
                                    "best_classical (6 paper families)"
                                ],
                                "difference_vs_best": None,
                                "verdict": "reference",
                            }
                        )
                        extension = idx.get(
                            (
                                component,
                                "best_classical_with_extension",
                                embedding,
                                n,
                                train_size,
                                protocol,
                            )
                        )
                        if extension is not None:
                            rows_out.append(
                                {
                                    **base,
                                    "system": "best classical + polynomial kernel (extension baseline)",
                                    "role": "extension baseline",
                                    "provenance": "local",
                                    "runs": coverage(extension),
                                    "mean": extension["mean"],
                                    "std": extension["std"],
                                    "paper_value": None,
                                    "tuning_configs": None,
                                    "difference_vs_best": None,
                                    "verdict": "reference",
                                }
                            )
                        for quantity, label in QUANTUM.items():
                            row = idx.get(
                                (
                                    component,
                                    quantity,
                                    embedding,
                                    n,
                                    train_size,
                                    protocol,
                                )
                            )
                            if row is None:
                                continue
                            difference = idx.get(
                                (
                                    component,
                                    f"{quantity}_minus_best_classical",
                                    embedding,
                                    n,
                                    train_size,
                                    protocol,
                                )
                            )
                            rows_out.append(
                                {
                                    **base,
                                    "system": f"{label} [{'MerLin adaptation' if family.startswith('photonic') else 'paper method'}]",
                                    "role": "MerLin adaptation"
                                    if family.startswith("photonic")
                                    else "paper method",
                                    "provenance": "local",
                                    "runs": coverage(row),
                                    "mean": row["mean"],
                                    "std": row["std"],
                                    "paper_value": row.get("paper_value"),
                                    "tuning_configs": budget[kind][quantity],
                                    "difference_vs_best": None
                                    if difference is None
                                    else {
                                        "mean": difference["mean"],
                                        "ci95": difference["ci95_half_width"],
                                    },
                                    "verdict": verdict(direction, difference),
                                }
                            )
    # Dataset (C) (dress vs shirt, natural labels): one record per (seed, n) with
    # every embedding's QK/PQK as families; classification error, lower is better.
    for family, aggregate in sources.items():
        idx = index(aggregate["summary"])
        embeddings = sorted(
            {r["embedding"] for r in aggregate["summary"] if r["embedding"]}
        )
        for n in SIZES[family]:
            for protocol in ("cv", "paper"):
                best = idx.get(
                    ("classical_dataset", "best_classical", None, n, 600, protocol)
                )
                if best is None:
                    continue
                base = {
                    "claim": "C10 Dataset (C) natural labels (error)",
                    "family": family,
                    "n": n,
                    "N": 600,
                    "protocol": protocol,
                    "scope": SCOPE[family],
                    "metric_direction": "min",
                    "unit_wall_clock_s": seconds[family].get((None, n)),
                }
                rows_out.append(
                    {
                        **base,
                        "embedding": "-",
                        "system": "best classical (paper baseline, 6 families)",
                        "role": "paper baseline",
                        "provenance": "local",
                        "runs": coverage(best),
                        "mean": best["mean"],
                        "std": best["std"],
                        "paper_value": best.get("paper_value"),
                        "tuning_configs": budget["classification"][
                            "best_classical (6 paper families)"
                        ],
                        "difference_vs_best": None,
                        "verdict": "reference",
                    }
                )
                for embedding in embeddings:
                    for quantity, label in QUANTUM.items():
                        name = f"{quantity}_{embedding}"
                        row = idx.get(
                            ("classical_dataset", name, None, n, 600, protocol)
                        )
                        if row is None:
                            continue
                        difference = idx.get(
                            (
                                "classical_dataset",
                                f"{name}_minus_best_classical",
                                None,
                                n,
                                600,
                                protocol,
                            )
                        )
                        rows_out.append(
                            {
                                **base,
                                "embedding": embedding,
                                "system": f"{label} [{'MerLin adaptation' if family.startswith('photonic') else 'paper method'}]",
                                "role": "MerLin adaptation"
                                if family.startswith("photonic")
                                else "paper method",
                                "provenance": "local",
                                "runs": coverage(row),
                                "mean": row["mean"],
                                "std": row["std"],
                                "paper_value": row.get("paper_value"),
                                "tuning_configs": budget["classification"][quantity],
                                "difference_vs_best": None
                                if difference is None
                                else {
                                    "mean": difference["mean"],
                                    "ci95": difference["ci95_half_width"],
                                },
                                "verdict": verdict("min", difference),
                            }
                        )
    training = load_json(args.training_size)
    tidx = index(training["summary"])
    for n in (10, 12):
        for size in (100, 1600):
            for quantity, label in QUANTUM.items():
                difference = tidx.get(
                    (
                        "training_size",
                        f"{quantity}_minus_best_classical",
                        "E2",
                        n,
                        size,
                        "paper",
                    )
                )
                row = tidx.get(("training_size", quantity, "E2", n, size, "paper"))
                rows_out.append(
                    {
                        "claim": "C8 training size (MAE)",
                        "family": "gate",
                        "embedding": "E2",
                        "n": n,
                        "N": size,
                        "protocol": "paper",
                        "scope": "paper-scale (n=10-13)",
                        "metric_direction": "min",
                        "system": f"{label} [paper method]",
                        "role": "paper method",
                        "provenance": "local",
                        "runs": coverage(row),
                        "mean": row["mean"],
                        "std": row["std"],
                        "paper_value": None,
                        "tuning_configs": budget["regression"][quantity],
                        "unit_wall_clock_s": None,
                        "difference_vs_best": {
                            "mean": difference["mean"],
                            "ci95": difference["ci95_half_width"],
                        },
                        "verdict": verdict("min", difference),
                    }
                )

    commensurability = [
        {
            "pair": "paper method (gate QK/PQK) vs paper baseline (best classical)",
            "status": "COMMENSURABLE",
            "why": "same units, paired seeds, subsample, split, N, test points, metric and selection protocol",
        },
        {
            "pair": "MerLin adaptation (photonic FQK/PQK) vs best classical in the photonic runs",
            "status": "COMMENSURABLE",
            "why": "same pipeline and pairing; the classical models are trained on the same photonic labels",
        },
        {
            "pair": "capped-photon regime (k ≤ 8) vs k = floor(m/2) regime",
            "status": "PARTIALLY_COMMENSURABLE",
            "why": "identical setup for m ≤ 17; for m ≥ 18 the photon number, Fock space and therefore labels differ, "
            "so only paired differences to each regime's own best classical model are compared",
        },
        {
            "pair": "MerLin adaptation vs paper method (gate)",
            "status": "PARTIALLY_COMMENSURABLE",
            "why": "same inputs, classical suite, protocol and seeds, but different labels (photonic vs gate QNN observable; "
            "engineered labels built from different kernels), so raw metrics are not directly comparable; compare paired "
            "differences to each family's own best classical model",
        },
        {
            "pair": "local results vs paper-reported values",
            "status": "PARTIALLY_COMMENSURABLE",
            "why": "same protocol family but n ≤ 16 vs ≤ 30, inferred conventions (input scale), regenerated labels; "
            "paper values come from test-set selection (compare with protocol 'paper')",
        },
        {
            "pair": "any local result vs paper-reported MerLin/photonic values",
            "status": "N/A",
            "why": "the paper has no photonic experiments",
        },
    ]
    hardware = {
        "machines": {
            "local": "arm64 Apple M5 (10 cores: 4P + 6E), macOS, CPU, float64, "
            "joblib n_jobs = 8 (model fitting)",
            "remote": "gpu-reproduced-papers-1: NVIDIA L4 (23 GB), CUDA, float32 (complex64) "
            "simulation, joblib n_jobs = 8; used for gate n = 18, 20 and photonic n >= 15 "
            "(and the capped regime)",
        },
        "simulation": "NumPy/torch statevector (gate), MerLin 0.2.3 FOCK (photonic)",
        "workload_per_unit": "simulate 800 states and fit every candidate of every family for all requested components "
        "(both protocols where CV applies). Unit wall-clock was measured on the remote machine for gate n >= 18 "
        "and photonic n >= 15",
        "total_wall_clock_s": {
            name: run_seconds_by_machine(agg["runs"]) for name, agg in sources.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dump_json(
        {
            "rows": rows_out,
            "commensurability": commensurability,
            "tuning_budget": budget,
            "hardware": hardware,
        },
        args.output.with_name(args.output.name + ".json.gz"),
        default=str,
    )

    def fmt(x, d=3):
        return "—" if x is None else f"{x:.{d}f}"

    lines = [
        "# Phase 5 comparison table (generated by utils/compare_systems.py)",
        "",
        "Decision rule (frozen): quantum − best classical paired 95% CI must exclude 0, otherwise TIE. "
        "Verdicts use protocol `cv`; `paper` rows are for comparison with paper values only.",
        "",
        "| Claim | Family | Emb | n | N | Protocol | System | Role | Runs | Mean | Std | Paper | Δ vs best [95% CI] | Verdict | Tuning configs | Unit wall-clock (s) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows_out:
        diff = r["difference_vs_best"]
        diff_text = (
            "—" if diff is None else f"{diff['mean']:+.4f} [±{diff['ci95']:.4f}]"
        )
        lines.append(
            f"| {r['claim']} | {r['family']} | {r['embedding']} | {r['n']} | {r['N']} | {r['protocol']} | {r['system']} | "
            f"{r['role']} | {r['runs']} | {fmt(r['mean'], 4)} | {fmt(r['std'], 4)} | {fmt(r['paper_value'], 4)} | "
            f"{diff_text} | {r['verdict']} | {r['tuning_configs'] or '—'} | {fmt(r['unit_wall_clock_s'], 1)} |"
        )
    lines += ["", "## Commensurability", "", "| Pair | Status | Why |", "|---|---|---|"]
    lines += [f"| {c['pair']} | {c['status']} | {c['why']} |" for c in commensurability]
    lines += [
        "",
        "## Hardware and timing",
        "",
        "```json",
        json.dumps(hardware, indent=2),
        "```",
        "",
        "## Tuning budget (hyperparameter configurations per family)",
        "",
        "```json",
        json.dumps(budget, indent=2),
        "```",
    ]
    args.output.with_suffix(".md").write_text("\n".join(lines) + "\n")
    cv_rows = [
        r for r in rows_out if r["protocol"] in ("cv",) or r["claim"].startswith("C8")
    ]
    tally = {}
    for r in cv_rows:
        if r["verdict"] in ("reference", "N/A"):
            continue
        tally.setdefault((r["claim"], r["family"], r["system"]), []).append(
            f"{r['embedding']}@{r['n']}:{r['verdict']}"
        )
    for key, values in tally.items():
        print(" | ".join(map(str, key)), "->", ", ".join(values))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
