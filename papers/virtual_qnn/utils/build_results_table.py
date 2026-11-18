"""Build the README results table (paper vs reproduced) from generated comparison files.

Usage::

    python3 papers/virtual_qnn/utils/build_results_table.py --results papers/virtual_qnn/results

Reads ``results/comparison.json`` (written by ``compare_with_paper.py``) and
writes ``results/results_summary.md``: one row per reproduced paper value with
paper value, reproduced value (mean over repetitions), delta, metric/trend
agreement under the predeclared criteria (LOG.md, "Sweep Plans"), claim
support, seed count and comment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HEADER = (
    "| Paper item | Claim tested | Paper value | Reproduced value | Delta | Metric agreement | Trend agreement | "
    "Claim support | Seeds | Comment |\n|---|---|---:|---:|---:|---|---|---|---:|---|"
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args(argv)
    r = json.loads((args.results / "comparison.json").read_text())
    rows = []
    for x in r["E0"]["rows"]:
        target = x["paper_minimum_row"]
        swapped = (not x["matches_minimum_row"]) and x["matches_vqem_row"]
        rows.append(
            (
                "Table I, minimum overhead",
                "C1",
                f"{target}",
                f"{x['gamma_min_sdp']:.5f}",
                f"{x['gamma_min_sdp'] - target:+.4f}",
                "yes"
                if x["matches_minimum_row"]
                else "no (rows swapped)"
                if swapped
                else "no",
                "n/a",
                "SUPPORTED",
                "1 (deterministic SDP)",
                f"{x['channel']}; our optimum equals Table I's 'VQEM' row"
                if swapped
                else x["channel"],
            )
        )
    for x in r["E1"]["rows"]:
        rows.append(
            (
                "Table I, VQEM overhead γ",
                "C1",
                f"{x['paper_gamma_vqem_row']}",
                f"{x['gamma_mean']:.4f}",
                f"{x['gamma_mean'] - x['paper_gamma_vqem_row']:+.4f}",
                "yes"
                if x["gamma_agrees"]
                else f"no ({x['gap_to_sdp_percent']:+.2f}% vs optimum)",
                "n/a",
                "SUPPORTED",
                "3",
                f"{x['channel']}; vs SDP optimum {x['gamma_sdp']:.4f}"
                + (
                    "; AD: matches Table I 'Minimum' row"
                    if "amplitude" in x["channel"]
                    else ""
                ),
            )
        )
        rows.append(
            (
                "Table I, HS² error",
                "C1",
                f"{x['paper_hs']:.2e}",
                f"{x['hs_mean']:.2e}",
                f"{x['hs_mean'] - x['paper_hs']:+.1e}",
                "yes" if x["hs_agrees"] else "no",
                "n/a",
                "SUPPORTED",
                "3",
                x["channel"],
            )
        )
    for x in r["E2"]["rows"]:
        if x["alpha"] in (10, 100, 1000, 3000):
            rows.append(
                (
                    f"Fig. 4, α = {x['alpha']}",
                    "C2",
                    f"γ {x['paper_gamma']:.4f}; HS² {x['paper_hs']:.2e}",
                    f"γ {x['gamma_mean']:.4f}; HS² {x['hs_mean']:.2e}",
                    f"Δγ {x['gamma_diff']:+.4f}; HS² ×{x['hs_ratio']:.2f}",
                    "yes" if x["agrees"] else "no",
                    "yes" if r["E2"]["trend_agrees"] else "no",
                    "SUPPORTED",
                    "3",
                    "AD q = 0.3; all 8 α in sweep_tables.md",
                )
            )
    for label, key in (("noisy", "noisy"), ("ideal", "ideal")):
        for x in r["E3"][key]["rows"]:
            if x["budget"] in (1.0, 1.3, 1.5, 2.0):
                rows.append(
                    (
                        f"Fig. 5, {label}, γ = {x['budget']}",
                        "C3",
                        f"{x['paper_hs']:.4f}",
                        f"{x['hs_mean']:.4f}",
                        f"{x['diff']:+.4f}",
                        "yes" if x["agrees"] else "no",
                        "yes" if r["E3"]["trend_agrees"] else "no",
                        "PARTIALLY_SUPPORTED (equal-shot test unresolved)",
                        "3",
                        "authors' own runs differ by 0.013 at γ = 1.3"
                        if (label == "noisy" and x["budget"] == 1.3)
                        else "depolarizing p = 0.2",
                    )
                )
        e = r["E3"][key]
        rows.append(
            (
                f"Fig. 5, {label}, first γ with HS² < 1e-4",
                "C3",
                f"{e['paper_first_budget_hs_below_1e-4']}",
                f"{e['first_budget_hs_below_1e-4']}",
                f"{e['first_budget_hs_below_1e-4'] - e['paper_first_budget_hs_below_1e-4']:+.1f}",
                "yes" if e["threshold_agrees"] else "no",
                "yes" if r["E3"]["trend_agrees"] else "no",
                "PARTIALLY_SUPPORTED",
                "3",
                "SDP minimum 1.375",
            )
        )
    for x in r["E4"]["rows"]:
        rows.append(
            (
                f"Fig. 6d, γ = {x['budget']}",
                "C4",
                f"test {x['paper_test']:.3f} ± {x['paper_test_std']:.3f}",
                f"test {x['test_mean']:.3f} ± {x['test_std']:.3f}",
                f"{x['test_mean'] - x['paper_test']:+.3f}",
                "yes (within 1 paper std)" if x["agrees"] else "no",
                "yes" if r["E4"]["trend_agrees"] else "no",
                "PARTIALLY_SUPPORTED (needs ≳100 shots; not noise robustness)",
                "10 paired",
                f"train {x['train_mean']:.3f} vs paper {x['paper_train']:.3f}; data match in distribution only",
            )
        )
    for x in r["E5"]["rows"]:
        if x["budget"] in (1.0, 2.5) and x["half_distance"] in (0.45, 0.8, 1.7):
            rows.append(
                (
                    f"Fig. 7, d = {x['half_distance']}, γ = {x['budget']}",
                    "C5",
                    f"{x['paper_energy']:.4f}",
                    f"{x['energy_mean']:.4f} (best of 3: {x['energy_best_of_3']:.4f})",
                    f"{x['diff_mean']:+.4f}",
                    "yes"
                    if x["agrees_mean"]
                    else ("best-of-3 only" if x["agrees_best_of_3"] else "no"),
                    "yes" if r["E5"]["trend_agrees"] else "no",
                    "PARTIALLY_SUPPORTED",
                    "3",
                    f"|E − E_FCI| = {x['abs_error_fci_mean']:.3f} Ha; paper curve is HF",
                )
            )
    e5 = r["E5"]
    rows.append(
        (
            "Fig. 7, all 40 points",
            "C5",
            "—",
            f"{e5['n_agree_mean']}/40 within 0.02 Ha (mean), {e5['n_agree_best_of_3']}/40 (best of 3)",
            f"median |Δ| {e5['median_abs_diff_mean']:.3f} Ha",
            "no",
            f"yes ({e5['points_ordered']}/10 distances ordered)",
            "PARTIALLY_SUPPORTED",
            "3",
            "gap diagnosed: not iteration budget; best-of-N insufficient (LOG.md)",
        )
    )
    lines = [
        "# Results summary (generated by utils/build_results_table.py from results/comparison.json)",
        "",
        "Paper values: Table I and the authors' figure data. Reproduced values: mean over repetitions, paper-accurate settings, exact simulation. "
        "Agreement criteria were predeclared (LOG.md, Sweep Plans). Claim support reflects the Phase 5 fair comparison.",
        "",
        HEADER,
    ]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    (args.results / "results_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(args.results / "results_summary.md", len(rows), "rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
