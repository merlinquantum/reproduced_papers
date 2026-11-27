"""Generate ``results/results_summary.md`` from the structured artifacts.

Writes ``results/results_summary.md`` with:

1. the paper-item table (paper value, reproduced value, delta, metric / trend
   agreement, claim support, seeds, comment) for Tables IV-VII (headline),
   Tables C (per-cell agreement by cell type), Table VIII (C5) and the noise side check of
   cross-architecture comparison, plus the fair-baseline rows;
2. the sweep index: every evidence sweep (reproduction, MerLin, fair baselines) with its
   purpose, candidates, successful / expected runs, completeness, machine, code
   hash, declared selection rule and the result recomputed here from the raw
   ``metrics.json`` files, and the path of its full candidate table
   (``summary.json``, written by the sweep coordinator).

Inputs: ``results/data/headline_statistics.json``, ``results/data/<architecture>_vs_paper.csv``,
``results/data/fair_comparison.json``, ``results/data/merlin_comparison.json``,
``results/data/noise_distribution_comparison.json`` and ``outdir/sweep_*``.

Usage (from the repository root)::

    python papers/distributed_qrc/utils/documentation_tables.py papers/distributed_qrc/outdir
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

PAPER_DIR = Path(__file__).resolve().parents[1]
ARCHS = {"srsr": "E1", "mrsr": "E2", "srmr": "E3", "mrmr": "E4"}
TABLES = {
    "srsr": "IV / C1-C3",
    "mrsr": "V / C4-C6",
    "srmr": "VI / C7-C9",
    "mrmr": "VII / C10-C12",
}
# Evidence sweeps (superseded first-pass sweeps without the -R2 suffix are excluded).
SWEEPS = {
    "P1-R6R7-R2": (
        "Choice of the neuron observable and of the reservoir update order",
        "min mean eval RMSE over quantum_ideal|linear and quantum_ideal|quantum_kernel_ideal; 1 % ties -> first listed",
        ["quantum_ideal|linear", "quantum_ideal|quantum_kernel_ideal"],
    ),
    "E1-R2": (
        "Paper grid, SRSR (Table IV)",
        "no selection; seed aggregation",
        None,
    ),
    "E2-R2": ("Paper grid, MRSR (Table V)", "no selection; seed aggregation", None),
    "E3-R2": ("Paper grid, SRMR (Table VI)", "no selection; seed aggregation", None),
    "E4-R2": (
        "Paper grid, MRMR (Table VII)",
        "no selection; seed aggregation",
        None,
    ),
    "P1-ROBUST-R2": (
        "Robustness to normalization (per split) and data artifacts (interpolated)",
        "no selection; materiality rule (results/data/robustness_verdicts.json)",
        None,
    ),
    "GAP-SHOTS-R2": (
        "Do finite shots explain the paper's noisy degradation?",
        "no selection; degradation-ratio categories (results/data/robustness_verdicts.json)",
        None,
    ),
    "GAP-NOISE-SCALE-R2": (
        "Which noise scale matches the paper's noisy degradation?",
        "closest scale by min sum |ratio - paper|; 1 % ties -> smaller (results/data/robustness_verdicts.json)",
        None,
    ),
    "GAP-INTERCEPT-R2": (
        "Does the missing intercept of Eq. 5 explain the linear-readout gap?",
        "no selection; >= 13/24 cells rule (results/data/robustness_verdicts.json)",
        None,
    ),
    "GAP-CLASSICAL-SRSR-R2": (
        "Classical-baseline seed spread, SRSR (20 seeds)",
        "no selection; paper value inside the 20-seed range in >= 5/8",
        None,
    ),
    "GAP-CLASSICAL-MRSR-R2": (
        "Classical-baseline seed spread, MRSR (20 seeds)",
        "as above",
        None,
    ),
    "GAP-CLASSICAL-SRMR-R2": (
        "Classical-baseline seed spread, SRMR (20 seeds)",
        "as above",
        None,
    ),
    "GAP-CLASSICAL-MRMR-R2": (
        "Classical-baseline seed spread, MRMR (20 seeds)",
        "as above",
        None,
    ),
    "XCHECK-E3-R2": (
        "Laptop CPU vs NVIDIA L4 equivalence, SRMR 10/2",
        "no selection; max |difference| (results/checks/local_vs_gpu_crosscheck.json)",
        None,
    ),
    "XCHECK-E4-R2": ("Laptop CPU vs NVIDIA L4 equivalence, MRMR 2x5", "as above", None),
    "P1-MERLIN-SCALES": (
        "MerLin encoding-scale selection",
        "min mean eval RMSE over photonic|linear and photonic|photonic_kernel; 1 % ties -> first listed",
        ["photonic|linear", "photonic|photonic_kernel"],
    ),
    "E1-MERLIN": (
        "MerLin grid, SRSR",
        "no selection; preservation tolerance (results/data/merlin_comparison.json)",
        None,
    ),
    "E2-MERLIN": ("MerLin grid, MRSR", "as above", None),
    "E3-MERLIN": ("MerLin grid, SRMR", "as above", None),
    "E4-MERLIN": ("MerLin grid, MRMR", "as above", None),
    "B1B3-RAW": (
        "No-reservoir control",
        "no sweep selection; per-cell γ/λ on eval (results/data/fair_comparison.json)",
        None,
    ),
    "E1-CLASSICAL": (
        "Fair baselines, SRSR",
        "no sweep selection; ESN, γ, λ selected on eval per cell",
        None,
    ),
    "E2-CLASSICAL": ("Fair baselines, MRSR", "as above", None),
    "E3-CLASSICAL": ("Fair baselines, SRMR", "as above", None),
    "E4-CLASSICAL": ("Fair baselines, MRMR", "as above", None),
    "C6-SRSR": (
        "Noise side check (Sec. III-B5), SRSR, noise ×10",
        "no selection; decision rule declared before the runs",
        None,
    ),
    "C6-MRSR": (
        "Noise side check (Sec. III-B5), MRSR, noise ×10",
        "as above",
        None,
    ),
    "C6-MRMR": (
        "Noise side check (Sec. III-B5), MRMR, noise ×10",
        "as above",
        None,
    ),
}


def total_neurons(config: str) -> int:
    """Total reservoir neurons of a configuration label: "10", "10/2" or "2x5"."""

    if "x" in config:
        k, n = config.split("x")
        return int(k) * int(n)
    return int(config.split("/")[0])


def esc(text: str) -> str:
    return text.replace("|", "\\|")


def selection_result(sweep_dir: Path, cells: list[str]) -> str:
    """Recompute a declared selection: mean eval RMSE over seeds and cells; 1 % ties -> first listed."""

    status = json.loads((sweep_dir / "sweep_status.json").read_text())
    order, scores = [], {}
    for run in status["runs"]:
        if run["status"] != "COMPLETED":
            continue
        key = json.dumps(run["candidate"], sort_keys=True)
        if key not in scores:
            order.append(key)
            scores[key] = []
        metrics = json.loads((sweep_dir / run["metrics_path"]).read_text())
        by_cell = {
            f"{c['reservoir_type']}|{c['readout_type']}": c["metrics"]["eval"]["rmse"]
            for c in metrics["cells"]
        }
        scores[key].append(np.mean([by_cell[c] for c in cells]))
    means = {k: float(np.mean(v)) for k, v in scores.items()}
    best = min(means.values())
    tied = [k for k in order if means[k] <= best * 1.01]
    table = "; ".join(f"{k} {means[k]:.6f}" for k in order)
    return f"selected {tied[0]} (tied within 1 %: {', '.join(tied)}). All candidates: {table}"


def main() -> int:
    outdir = Path(sys.argv[1])
    head = json.loads(
        (PAPER_DIR / "results" / "data" / "headline_statistics.json").read_text()
    )
    fair = json.loads(
        (PAPER_DIR / "results" / "data" / "fair_comparison.json").read_text()
    )
    noise = json.loads(
        (
            PAPER_DIR / "results" / "data" / "noise_distribution_comparison.json"
        ).read_text()
    )
    lines = [
        "# Results summary (generated by `utils/documentation_tables.py`)",
        "",
        "Test metrics on the normalized scale. Ours: mean over seeds 0-2 (full data, full-n kernels, simulation; `paper-accurate` protocol). Paper values: as printed in the paper's tables (one value per cell; no seeds or spread reported).",
        "",
        "## Paper items",
        "",
        "| Paper item | Claim tested | Paper value | Reproduced value | Delta | Metric agreement | Trend agreement | Claim support | Seeds | Comment |",
        "|---|---|---:|---:|---:|---|---|---|---:|---|",
    ]
    for arch in ARCHS:
        p = head["paper"][arch]["mae"]
        o = head["ours"][arch]["paper_rule"]["mae"]
        for stat in ("average", "peak"):
            pv, ov, o_std = p[stat], o[f"{stat}_mean"], o[f"{stat}_std"]
            lines.append(
                f"| Table {TABLES[arch].split(' / ')[0]} ({arch.upper()}): {stat} MAE reduction, paper rule vs the paper baseline | C1 | {pv:.1%} | {ov:.1%} ± {o_std:.1%} | {ov - pv:+.1%} | {'within 1 std' if abs(ov - pv) <= o_std else 'outside 1 std'} | yes (large reductions vs the paper baseline) | supported vs the paper baseline only; unsupported vs fair classical (fair baselines) | 3 | the paper baseline is seed-unstable; paper value recomputed from its tables{' (reported 25.6 %)' if arch == 'mrsr' and stat == 'average' else ''} |"
            )
    for arch in ARCHS:
        rows = list(
            csv.DictReader(
                open(PAPER_DIR / "results" / "data" / f"{arch}_vs_paper.csv")
            )
        )
        groups: dict[str, list[dict]] = {}
        for r in rows:
            groups.setdefault(f"{r['reservoir']}|{r['readout']}", []).append(r)
        for cell in (
            "quantum_ideal|quantum_kernel_ideal",
            "classical|quantum_kernel_ideal",
            "classical|linear",
            "quantum_ideal|linear",
            "quantum_noisy|quantum_kernel_noisy",
        ):
            g = groups[cell]
            rel = np.array([float(r["mae_rel"]) for r in g])
            ours = np.array([float(r["mae_ours"]) for r in g])
            paper = np.array([float(r["mae_paper"]) for r in g])
            within = int(np.sum(np.abs(rel) <= 0.05))
            lines.append(
                f"| Tables {TABLES[arch].split(' / ')[1]} ({arch.upper()}): `{esc(cell)}` over 8 configurations | {'C1, C2, C3' if 'quantum_kernel' in cell else 'C1'}{', C4' if 'noisy' in cell else ''} | {paper.min():.4f}–{paper.max():.4f} | {ours.min():.4f}–{ours.max():.4f} | median {np.median(rel):+.1%}, max abs {np.abs(rel).max():.1%} | {within}/8 within 5 % | {'yes' if within >= 6 else 'partial'} | see claim rows | 3 | per-cell values: `results/data/{arch}_vs_paper.csv` |"
            )
    c5 = fair["C5_best_model"]
    lines.append(
        f"| Table VIII / Table V: best overall model, MRSR 3x20 classical reservoir + quantum kernel | C5 | 0.0113 | {c5['hybrid_3x20']['mean']:.4f} ± {c5['hybrid_3x20']['std']:.4f} | {c5['hybrid_3x20']['mean'] - 0.0113:+.4f} | no | no (not the best model in our runs) | unsupported | 3 | ties with the fair classical models ({c5['fair_classical_3x20']['mean']:.4f} ± {c5['fair_classical_3x20']['std']:.4f}) |"
    )
    same_sign = sum(
        v["same_sign"] for k, v in noise.items() if k.startswith("mrsr-vs-srsr@")
    )
    within = sum(
        abs(v["ours_diff_in_pooled_std"]) <= 1
        for k, v in noise.items()
        if k.startswith("mrsr-vs-srsr@")
    )
    c6 = fair["noise_side_check_x10"]
    lines.append(
        f"| Sec. III-B: noisy fully quantum MRSR vs SRSR at equal N (×1, paper mapping) | side check (not a tracked claim) | MRSR better at 8/8 | same sign as paper at {same_sign}/8, abs Δ ≤ 1 pooled std at {within}/8 | — | no | no | unsupported | 3 | `results/data/noise_distribution_comparison.json` |"
    )
    lines.append(
        f"| Sec. III-B, controlled (fair baselines): MRSR vs SRSR at noise ×10 | side check (not a tracked claim) | MRSR better at 8/8 | paper mapping {c6['paper']['mrsr_vs_srsr']['better']}/8; same device {c6['ibm_brisbane']['mrsr_vs_srsr']['better']}/8 | — | N/A (noise model differs) | only under the paper's device mapping | unsupported as a distribution effect | 3 | `results/data/fair_comparison.json` → `noise_side_check_x10` |"
    )
    for arch in ARCHS:
        a = fair["architectures"][arch]
        cl = a["claims"]["C1_vs_fair_classical"]["counts"]
        h = a["headline_vs_best_classical"]["mae"]
        lines.append(
            f"| Fair baselines ({arch.upper()}): eval-selected quantum vs eval-selected best classical (the paper baseline and the additional baselines) | C1 | N/A (no fair baseline in the paper) | {h['average']['mean']:+.1%} ± {h['average']['std']:.1%} average MAE reduction; better in {cl['better']}/8, tie {cl['tie']}/8, worse {cl['worse']}/8 | — | N/A | N/A | unsupported | 3 | `results/fair_comparison_table.md` |"
        )

    lines += [
        "",
        "## Headline reductions, configuration by configuration",
        "",
        "Reduction = 1 - MAE(best quantum/hybrid cell) / MAE(paper baseline), with the paper baseline = `classical|linear` and the best cell taken among the 8 quantum or hybrid cells by highest test R2 (our reading of the paper's rule, which the paper does not state). Paper: computed from its Tables C. Ours: the best cell is chosen per seed (listed per seed when it differs); paper baseline MAE, best MAE and reduction are means over seeds 0-2. Averaging a reduction column over the 8 configurations gives the headline average of the README. The headline peak is the maximum over configurations taken per run (one run for the paper, each seed for us) and then averaged over seeds, so our peak can exceed the maximum of the column below when the seeds peak at different configurations (MRSR: 57.9 % vs 47.6 %).",
        "",
        "| Architecture | Configuration | Paper: baseline MAE | Paper: best cell | Paper: best MAE | Paper: reduction | Ours: baseline MAE | Ours: best cell | Ours: best MAE | Ours: reduction |",
        "|---|---|---:|---|---:|---:|---:|---|---:|---:|",
    ]
    for arch in ARCHS:
        paper_rows = head["paper"][arch]["mae"]["per_config"]
        seed_rows = [
            s["per_config"] for s in head["ours"][arch]["paper_rule"]["mae"]["per_seed"]
        ]
        for config in sorted(paper_rows, key=total_neurons):
            pr = paper_rows[config]
            ours = [s[config] for s in seed_rows]
            cells = [o["best_cell"] for o in ours]
            cell_text = cells[0] if len(set(cells)) == 1 else " / ".join(cells)
            lines.append(
                f"| {arch.upper()} | {config} | {pr['baseline']:.4f} | `{esc(pr['best_cell'])}` | {pr['best']:.4f} | {pr['reduction']:.1%} "
                f"| {np.mean([o['baseline'] for o in ours]):.4f} | `{esc(cell_text)}` | {np.mean([o['best'] for o in ours]):.4f} | {np.mean([o['reduction'] for o in ours]):.1%} |"
            )

    lines += [
        "",
        "## Sweep index",
        "",
        "Every evidence sweep. The full candidate-by-cell table of each sweep (mean and std over seeds of every split and metric) is its coordinator's `summary.json`. The suffix `-R2` marks the second, fully pre-planned run of the reproduction; an earlier pass was superseded and is not evidence.",
        "",
        "| Sweep | Purpose | Candidates | Successful / expected runs | Completeness | Machine | Code (`lib_sha256`) | Declared rule | Result (recomputed from raw metrics) | Full table |",
        "|---|---|---:|---:|---|---|---|---|---|---|",
    ]
    for sweep_id, (purpose, rule, cells) in SWEEPS.items():
        sweep_dir = outdir / f"sweep_{sweep_id}"
        status = json.loads((sweep_dir / "sweep_status.json").read_text())
        candidates = {
            json.dumps(r["candidate"], sort_keys=True) for r in status["runs"]
        }
        done = sum(r["status"] == "COMPLETED" for r in status["runs"])
        complete = (
            "COMPLETE"
            if status["status"] == "COMPLETED" and done == status["expected_runs"]
            else "PARTIAL"
        )
        machine = {"cuda:0": "NVIDIA L4"}.get(status.get("device"), "local CPU")
        result = selection_result(sweep_dir, cells) if cells else "see rule artifact"
        lines.append(
            f"| `{sweep_id}` | {purpose} | {len(candidates)} | {done} / {status['expected_runs']} | {complete} | {machine} | {status['code'].get('lib_sha256', '?')[:8]}… | {esc(rule)} | {esc(result)} | `outdir/sweep_{sweep_id}/summary.json` |"
        )
    path = PAPER_DIR / "results" / "results_summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {os.path.relpath(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
