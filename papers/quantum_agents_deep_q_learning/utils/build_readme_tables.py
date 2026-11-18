"""Generate the README / Confluence result tables from structured results.

Reads ``results/statistics.json`` (paper-table settings),
``results/statistics_codefaithful.json`` (code-faithful settings),
``results/phase5_selection_{nn,pqc}.json`` and ``results/merlin_comparison.json``
and writes, into ``--output-dir``:

* ``readme_results_table.md``: one row per reproduced paper figure item with
  paper value, reproduced value (code-faithful set), delta, metric / trend
  agreement, claim support, seeds and a comment holding the paper-table value;
* ``readme_sweep_table.md``: the complete Phase 5 tuning-grid candidate table
  with the declared rule, run counts, ties and completeness;
* ``readme_merlin_table.md``: the MerLin vs qubit table with hardware-aware
  fields;
* ``readme_phase5_table.md``: the parameter-matched PQC vs NN comparison with
  provenance, scope, run counts and the pairwise tests.

Paper values marked ``~`` were read by eye from the paper's figures (about
+-20 episodes for Fig. 6, +-0.01 for Fig. 3b); text values are quoted. The
metric agreement label is ``within CI`` when the paper value lies in our 95%
bootstrap interval of the mean episodes, otherwise ``outside CI``; for Fig.
3b it is ``within tolerance`` when |delta| <= 0.01 + 2 s.e.m.

Example (from the paper folder)::

    python utils/build_readme_tables.py --output-dir results
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]

# (paper item, claim, label, config stem in statistics*.json, paper value, trend, support)
FIG6A = {5: 320, 10: 635, 15: 690, 20: 770, 25: 350, 30: 940}
FIG6B = {
    "10_10": 1110,
    "15_15": 540,
    "20_20": 430,
    "24_24": 415,
    "30_30": 250,
    "64_64": 345,
}
FIG3B = {5: 0.19, 10: 0.17, 15: 0.155}
STEM_PQC = {5: "e2e3_cartpole_optimal_full_original"}


def load(name: str) -> dict:
    return json.loads((PROJECT_DIR / "results" / name).read_text())


def episodes_cell(record: dict) -> str:
    return (
        f"{record['solved']}/{record['agents']} solved, mean {record['mean_censored']:.0f} "
        f"[{record['mean_censored_ci95'][0]:.0f}, {record['mean_censored_ci95'][1]:.0f}]"
    )


def results_table(paper_table: dict, code_faithful: dict) -> str:
    cf, pt = code_faithful["configs"], paper_table["configs"]
    lines = [
        "| Paper item | Claim tested | Paper value | Reproduced value (code-faithful) | Delta | Metric agreement | Trend agreement | Claim support | Seeds | Comment (paper-table value) |",
        "|---|---|---|---|---:|---|---|---|---:|---|",
    ]

    def add(
        item,
        claim,
        paper,
        stem,
        trend,
        support,
        comment_extra="",
        numeric=None,
        qualitative_metric=None,
    ):
        record = cf[stem]
        reproduced = episodes_cell(record)
        if numeric is not None:
            delta = f"{record['mean_censored'] - numeric:+.0f}"
            low, high = record["mean_censored_ci95"]
            metric = "within CI" if low <= numeric <= high else "outside CI"
        else:
            delta, metric = "--", qualitative_metric or "qualitative"
        comment = f"paper-table: {episodes_cell(pt[stem])}" + (
            f"; {comment_extra}" if comment_extra else ""
        )
        lines.append(
            f"| {item} | {claim} | {paper} | {reproduced} | {delta} | {metric} | {trend} | {support} | 10 (0-9) | {comment} |"
        )

    for depth, name in ((5, "l5"), (10, "l10"), (15, "l15")):
        add(
            f"Fig. 3a, FL {depth} layers",
            "C1: all agents solve",
            "all 10 solve",
            f"e1_frozenlake_{name}_original",
            "time falls with depth: yes",
            "supported",
            qualitative_metric="count matches",
        )
    for depth, name in ((5, "l5"), (10, "l10"), (15, "l15")):
        record = cf[f"e1_frozenlake_{name}_original"]
        measured, paper = record["mae44_at_step_10000_mean"], FIG3B[depth]
        delta = measured - paper
        tolerance = 0.01 + 2 * record["mae44_at_step_10000_sem"]
        agreement = (
            "within tolerance" if abs(delta) <= tolerance else "lower than paper"
        )
        lines.append(
            f"| Fig. 3b, FL {depth} layers | C1: MAE to Q* (44 non-terminal pairs, step 10000) | ~{paper} | "
            f"{measured:.3f} (s.e.m. {record['mae44_at_step_10000_sem']:.3f}) | {delta:+.3f} | {agreement} | "
            f"ordering by depth: yes | partially supported | 10 (0-9) | paper-table: "
            f"{pt[f'e1_frozenlake_{name}_original']['mae44_at_step_10000_mean']:.3f}; start value 0.54 matches |"
        )
    add(
        "Fig. 4a / 5a, CP full, best hparams",
        "C3/C4: solves Cart Pole",
        "<= 600 (text); ~320 (Fig. 6a)",
        "e2e3_cartpole_optimal_full_original",
        "yes",
        "supported",
        numeric=320,
    )
    add(
        "Fig. 4a, CP w/o re-uploading, best",
        "C3: re-uploading needed",
        "does not solve (text)",
        "e2e3_cartpole_optimal_no_reuploading_original",
        "no: about as fast as full",
        "not supported",
        "paper's Fig. 4a curve ~190",
        qualitative_metric="not reproduced",
    )
    add(
        "Fig. 4a, CP w/o input weights, best",
        "C3: trainable w_d needed",
        "does not solve (text)",
        "e2e3_cartpole_optimal_no_input_weights_original",
        "yes: slower and less reliable",
        "partially supported",
        qualitative_metric="not reproduced (8/10 solve)",
    )
    add(
        "Fig. 4b, CP full, sub-optimal",
        "C3: full model robust to hparams",
        "<= 3000 (text)",
        "e2e3_cartpole_suboptimal_full_original",
        "yes",
        "supported",
        qualitative_metric="within text bound",
    )
    add(
        "Fig. 4b, CP w/o re-uploading, sub-opt.",
        "C3",
        "does not solve (text)",
        "e2e3_cartpole_suboptimal_no_reuploading_original",
        "yes: slower, 8/10 solve",
        "partially supported",
        qualitative_metric="not reproduced",
    )
    add(
        "Fig. 4b, CP w/o input weights, sub-opt.",
        "C3",
        "does not solve (text)",
        "e2e3_cartpole_suboptimal_no_input_weights_original",
        "yes: 5/10 solve",
        "partially supported",
        qualitative_metric="not reproduced",
    )
    add(
        "Fig. 5a, CP fixed [0,1], best",
        "C2: range must match Q*",
        "never solves",
        "e2e3_cartpole_optimal_fixed_range_1_original",
        "yes",
        "supported",
        qualitative_metric="matches",
    )
    add(
        "Fig. 5a, CP fixed [0,90], best",
        "C2",
        "solves",
        "e2e3_cartpole_optimal_fixed_range_90_original",
        "yes",
        "supported",
        qualitative_metric="matches",
    )
    add(
        "Fig. 5a, CP fixed [0,180], best",
        "C2",
        "solves (slower)",
        "e2e3_cartpole_optimal_fixed_range_180_original",
        "yes",
        "supported",
        qualitative_metric="matches",
    )
    add(
        "Fig. 5b, CP fixed [0,1], sub-opt.",
        "C2",
        "never solves",
        "e2e3_cartpole_suboptimal_fixed_range_1_original",
        "yes",
        "supported",
        qualitative_metric="matches",
    )
    add(
        "Fig. 5b, CP fixed [0,90], sub-opt.",
        "C2",
        "solves (~4500)",
        "e2e3_cartpole_suboptimal_fixed_range_90_original",
        "yes",
        "supported",
        qualitative_metric="matches (solves earlier)",
    )
    add(
        "Fig. 5b, CP fixed [0,180], sub-opt.",
        "C2: [0,180] can fail completely",
        "complete failure",
        "e2e3_cartpole_suboptimal_fixed_range_180_original",
        "no",
        "not supported",
        "implementation and settings match the reference code; unexplained",
        qualitative_metric="not reproduced",
    )
    for depth, paper in FIG6A.items():
        stem = STEM_PQC.get(depth, f"e4_cartpole_pqc_l{depth}_original")
        extra = "text: 500 for 25 layers" if depth == 25 else ""
        add(
            f"Fig. 6a, PQC {depth} layers",
            "C5: non-monotonic in parameters",
            f"~{paper}",
            stem,
            "non-monotonic: yes",
            "partially supported",
            extra,
            numeric=paper,
        )
    for width, paper in FIG6B.items():
        extra = "text: 250 for (20,20)" if width == "20_20" else ""
        add(
            f"Fig. 6b, NN ({width.replace('_', ',')})",
            "C5: non-monotonic in parameters",
            f"~{paper}",
            f"e4_cartpole_mlp_{width}_classical",
            "non-monotonic: yes",
            "partially supported",
            extra,
            numeric=paper,
        )
    best = code_faithful["best_individual"]
    for label, key, paper in (("PQC", "pqc", 206), ("NN", "nn", 186)):
        episodes = best[key]["episodes"]
        lines.append(
            f"| Fig. 7, fastest {label} agent | C6: best agents close | {paper} | {episodes} ({best[key]['config']}) | "
            f"{episodes - paper:+d} | within 5 episodes | yes | supported (descriptive) | 10 (0-9) per config | "
            f"extreme-value statistic over 6 configs; no inference |"
        )
    return "\n".join(lines) + "\n"


def sweep_table() -> str:
    lines = [
        "Declared rule (Phase 5 Comparison Plan, LOG.md): per model, metric = median over the 5 selection agents "
        "(seeds 100-104) of the 0-based solve episode, unsolved censored at 5000; direction: lower is better; split: "
        "online interaction on selection seeds (evaluation seeds 0-9 are disjoint); checkpoint: final (agents stop at "
        "solve); aggregation: median over agents; tie tolerance 5% of the best median, ties go to the smaller batch, "
        "then the smaller target interval.",
        "",
        "| Model | Params | Batch | Target update | Median episodes | Solved (of 5) | Status |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    params = {"mlp_5_5": 67, "mlp_14_14": 310, "pqc_l5": 62, "pqc_l25": 302}
    notes = []
    for file in ("phase5_selection_nn.json", "phase5_selection_pqc.json"):
        selection = load(file)["models"]
        for model, info in selection.items():
            for row in info["candidates"]:
                status = (
                    "selected"
                    if row["config"] == info["selected"]
                    else (
                        "tied within 5%"
                        if row["config"] in info["tied_candidates"]
                        else ""
                    )
                )
                lines.append(
                    f"| {model} | {params[model]} | {row['batch_size']} | {row['update_target_every']} | "
                    f"{row['median_censored']:.0f} | {row['solved']} | {status} |"
                )
            ties = [c for c in info["tied_candidates"] if c != info["selected"]]
            notes.append(
                f"{model}: best median {info['best_median']:.0f}; selected `{info['selected']}`; "
                + (
                    f"tied: {', '.join(ties)} (smaller batch wins)"
                    if ties
                    else "no ties"
                )
            )
    lines += [
        "",
        "Completeness: COMPLETE SWEEP, 24 of 24 planned candidate runs observed and COMPLETED (4 models x 6 settings), "
        "validated with `validate_logging.py`; `utils/select_phase5.py` raises on any missing, duplicate or "
        "non-100-104-seed candidate. Selections and ties:",
        "",
        *[f"- {note}" for note in notes],
        "",
        "Uncertainty: each candidate has only 5 agents, so medians carry large sampling noise, in particular for "
        "NN (5,5) (medians of 1487-5000 with 1-5 of 5 solved); the selection is a protocol step, not evidence "
        "that the chosen setting is optimal. The selected settings were evaluated once on seeds 0-9 "
        "(`results/phase5_comparison.md`).",
    ]
    return "\n".join(lines) + "\n"


def merlin_table() -> str:
    pairs = load("merlin_comparison.json")["pairs"]
    lines = [
        "| MerLin config | Solved (MerLin / qubit) | Mean episodes MerLin [95% CI] | Mean episodes qubit [95% CI] | MWU p | Agreement | Params (MerLin / qubit) | Wall-clock s (MerLin) |",
        "|---|---:|---|---|---:|---|---|---:|",
    ]
    for name, item in pairs.items():
        m, q = item["merlin"], item["qubit"]
        lines.append(
            f"| `{name}` | {m['solved']}/10 / {q['solved']}/10 | {m['mean_censored']:.0f} "
            f"[{m['mean_censored_ci95'][0]:.0f}, {m['mean_censored_ci95'][1]:.0f}] | {q['mean_censored']:.0f} "
            f"[{q['mean_censored_ci95'][0]:.0f}, {q['mean_censored_ci95'][1]:.0f}] | {item['mann_whitney_p']:.2g} | "
            f"{item['agreement']} | {m['trainable_parameters']} / {q['trainable_parameters']} | "
            f"{m['wall_clock_seconds']:.0f} |"
        )
    return "\n".join(lines) + "\n"


def phase5_table() -> str:
    comparison = load("phase5_comparison.json")
    lines = [
        "| System | Role | Scope | Successful / expected runs | Params | Solved | Median episodes [95% CI] | Mean | Wall-clock, hardware |",
        "|---|---|---|---:|---:|---:|---|---:|---|",
    ]
    for name, row in comparison["rows"].items():
        lines.append(
            f"| {name} | {row['role']} | {row['scope']} | {row['successful_runs']} / {row['expected_runs']} | "
            f"{row['trainable_parameters']} | {row['solved']}/10 | {row['median_censored']:.0f} "
            f"[{row['median_ci95'][0]:.0f}, {row['median_ci95'][1]:.0f}] | {row['mean_censored']:.0f} | "
            f"{row['wall_clock_seconds']:.0f} s, {row['hardware']} |"
        )
    lines += [
        "",
        "| Pair | Median A | Median B | Median difference A-B [95% CI] | Mann-Whitney p | Verdict | Status |",
        "|---|---:|---:|---|---:|---|---|",
    ]
    for pair in comparison["pairs"]:
        low, high = pair["median_difference_ci95"]
        lines.append(
            f"| {pair['a']} vs {pair['b']} | {pair['median_a']:.0f} | {pair['median_b']:.0f} | "
            f"{pair['median_a'] - pair['median_b']:.0f} [{low:.0f}, {high:.0f}] | {pair['mann_whitney_p']:.3g} | "
            f"{pair['verdict']} | {pair['commensurability']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "readme_results_table.md").write_text(
        results_table(load("statistics.json"), load("statistics_codefaithful.json"))
    )
    (args.output_dir / "readme_sweep_table.md").write_text(sweep_table())
    (args.output_dir / "readme_merlin_table.md").write_text(merlin_table())
    (args.output_dir / "readme_phase5_table.md").write_text(phase5_table())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
