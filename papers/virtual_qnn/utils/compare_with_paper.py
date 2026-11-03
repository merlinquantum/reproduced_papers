"""Compare sweep results with the paper using the predeclared criteria (LOG.md).

Usage (from the repository root)::

    python3 papers/virtual_qnn/utils/compare_with_paper.py \
        --sweeps-root papers/virtual_qnn/outdir/sweeps \
        --e0-run papers/virtual_qnn/outdir/e0_sdp/run_<timestamp> \
        --out papers/virtual_qnn/results

Reads only structured artifacts (``summary.json``, ``sweep_status.json``,
per-run ``metrics.json``) and writes ``comparison.json`` and
``comparison.md``. Every number is computed here at full precision; nothing is
transcribed from logs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR.parents[1]))
sys.path.insert(0, str(PROJECT_DIR.parent))

from virtual_qnn.lib import paper_values as paper  # noqa: E402


def load_summary(sweeps_root: Path, sweep_id: str) -> dict:
    """Load a sweep summary and refuse partial coverage."""
    summary = json.loads((sweeps_root / sweep_id / "summary.json").read_text())
    if summary["coverage"] != "COMPLETE SWEEP":
        raise ValueError(
            f"{sweep_id} is {summary['coverage']}; comparison requires full coverage"
        )
    return summary


def candidate_map(summary: dict, *axes: str) -> dict:
    """Index candidates by the tuple of their grid values for ``axes``."""
    return {tuple(c["grid_point"][a] for a in axes): c for c in summary["candidates"]}


def per_run(summary: dict, key: tuple, axes: tuple, metric: str) -> dict[int, float]:
    """Return ``{repetition: value}`` for one candidate."""
    candidate = candidate_map(summary, *axes)[key]
    return {run["repetition"]: run[metric] for run in candidate["runs"]}


def compare_e0(e0_run: Path) -> dict:
    metrics = json.loads((e0_run / "metrics.json").read_text())
    rows = []
    for channel in metrics["channels"]:
        key = (channel["noise_model"], channel["noise_level"])
        rows.append(
            {
                "channel": f"{key[0]} {key[1]}",
                "gamma_min_sdp": channel["gamma_min"],
                "gamma_min_analytic": channel.get("gamma_min_analytic"),
                "paper_minimum_row": paper.TABLE1_OVERHEAD_MINIMUM[key],
                "paper_vqem_row": paper.TABLE1_OVERHEAD_VQEM[key],
                "matches_minimum_row": abs(
                    channel["gamma_min"] - paper.TABLE1_OVERHEAD_MINIMUM[key]
                )
                < 5e-4,
                "matches_vqem_row": abs(
                    channel["gamma_min"] - paper.TABLE1_OVERHEAD_VQEM[key]
                )
                < 5e-4,
            }
        )
    return {"rows": rows}


def compare_e1(summary: dict, e0: dict) -> dict:
    sdp = {row["channel"]: row["gamma_min_sdp"] for row in e0["rows"]}
    rows = []
    for c in summary["candidates"]:
        key = (c["grid_point"]["--noise-model"], c["grid_point"]["--noise-level"])
        name = f"{key[0]} {key[1]}"
        hs, gamma = c["aggregates"]["hs_squared"], c["aggregates"]["overhead"]
        gap = (gamma["mean"] - sdp[name]) / sdp[name]
        rows.append(
            {
                "channel": name,
                "hs_mean": hs["mean"],
                "hs_std": hs["std"],
                "paper_hs": paper.TABLE1_HS[key],
                "gamma_mean": gamma["mean"],
                "gamma_std": gamma["std"],
                "gamma_min": gamma["min"],
                "gamma_sdp": sdp[name],
                "gap_to_sdp_percent": 100 * gap,
                "paper_gamma_vqem_row": paper.TABLE1_OVERHEAD_VQEM[key],
                "paper_gamma_minimum_row": paper.TABLE1_OVERHEAD_MINIMUM[key],
                "hs_agrees": hs["mean"] <= 1e-5,
                "gamma_agrees": abs(gap) <= 0.01,
                "n": c["successful_runs"],
            }
        )
    return {
        "rows": rows,
        "verdict": "quantitative"
        if all(r["hs_agrees"] and r["gamma_agrees"] for r in rows)
        else "quantitative except listed rows",
    }


def compare_e2(summary: dict) -> dict:
    by_alpha = candidate_map(summary, "--penalty")
    rows = []
    for alpha, ref_gamma, ref_hs in zip(
        paper.FIG4_PENALTY, paper.FIG4_OVERHEAD, paper.FIG4_HS
    ):
        c = by_alpha[(alpha,)]
        hs, gamma = c["aggregates"]["hs_squared"], c["aggregates"]["overhead"]
        rows.append(
            {
                "alpha": alpha,
                "hs_mean": hs["mean"],
                "hs_std": hs["std"],
                "paper_hs": ref_hs,
                "hs_ratio": hs["mean"] / ref_hs,
                "gamma_mean": gamma["mean"],
                "gamma_std": gamma["std"],
                "paper_gamma": ref_gamma,
                "gamma_diff": gamma["mean"] - ref_gamma,
                "agrees": (1 / 3 <= hs["mean"] / ref_hs <= 3)
                and abs(gamma["mean"] - ref_gamma) <= 0.02,
            }
        )
    hs_means = [r["hs_mean"] for r in rows]
    monotone = all(b <= a * (1 + 1e-2) for a, b in zip(hs_means, hs_means[1:]))
    final_gap = abs(rows[-1]["gamma_mean"] - 1.3 / 0.7) / (1.3 / 0.7)
    return {
        "rows": rows,
        "trend_agrees": monotone and final_gap <= 0.01,
        "quantitative": all(r["agrees"] for r in rows),
    }


def _first_budget_below(
    budgets: list[float], values: list[float], threshold: float
) -> float | None:
    return next((b for b, v in zip(budgets, values) if v < threshold), None)


def compare_e3(noisy: dict, ideal: dict) -> dict:
    out = {}
    for label, summary, reference in (
        ("noisy", noisy, paper.FIG5_HS_NOISY),
        ("ideal", ideal, paper.FIG5_HS_IDEAL),
    ):
        by_budget = candidate_map(summary, "--vqem-budget")
        rows = []
        for budget, ref in zip(paper.FIG5_BUDGET, reference):
            agg = by_budget[(budget,)]["aggregates"]["hs_squared"]
            rows.append(
                {
                    "budget": budget,
                    "hs_mean": agg["mean"],
                    "hs_std": agg["std"],
                    "paper_hs": ref,
                    "diff": agg["mean"] - ref,
                    "agrees": abs(agg["mean"] - ref) <= 0.01,
                }
            )
        ours = _first_budget_below(
            paper.FIG5_BUDGET, [r["hs_mean"] for r in rows], 1e-4
        )
        theirs = _first_budget_below(paper.FIG5_BUDGET, reference, 1e-4)
        out[label] = {
            "rows": rows,
            "first_budget_hs_below_1e-4": ours,
            "paper_first_budget_hs_below_1e-4": theirs,
            "threshold_agrees": ours is not None
            and theirs is not None
            and abs(ours - theirs) <= 0.1 + 1e-9,
        }
    noisy_above = all(
        n["hs_mean"] >= i["hs_mean"] - 1e-12
        for n, i in zip(out["noisy"]["rows"], out["ideal"]["rows"])
    )
    out["trend_agrees"] = (
        noisy_above
        and out["noisy"]["rows"][-1]["hs_mean"] < 1e-4
        and out["ideal"]["rows"][-1]["hs_mean"] < 1e-4
    )
    out["quantitative"] = (
        all(r["agrees"] for k in ("noisy", "ideal") for r in out[k]["rows"])
        and out["noisy"]["threshold_agrees"]
        and out["ideal"]["threshold_agrees"]
    )
    return out


def compare_e4(budget_summary: dict, baseline_summary: dict) -> dict:
    by_budget = candidate_map(budget_summary, "--vqc-budget")
    rows = []
    for i, budget in enumerate(paper.FIG6_BUDGET):
        agg = by_budget[(budget,)]["aggregates"]
        rows.append(
            {
                "budget": budget,
                "train_mean": agg["train_accuracy"]["mean"],
                "train_std": agg["train_accuracy"]["std"],
                "test_mean": agg["test_accuracy"]["mean"],
                "test_std": agg["test_accuracy"]["std"],
                "paper_train": paper.FIG6_TRAIN_MEAN[i],
                "paper_train_std": paper.FIG6_TRAIN_STD[i],
                "paper_test": paper.FIG6_TEST_MEAN[i],
                "paper_test_std": paper.FIG6_TEST_STD[i],
                "n_parameters": agg["n_parameters"]["mean"],
                "agrees": abs(agg["train_accuracy"]["mean"] - paper.FIG6_TRAIN_MEAN[i])
                <= paper.FIG6_TRAIN_STD[i]
                and abs(agg["test_accuracy"]["mean"] - paper.FIG6_TEST_MEAN[i])
                <= paper.FIG6_TEST_STD[i],
            }
        )
    test_means = [r["test_mean"] for r in rows]
    pooled = float(np.sqrt((rows[0]["test_std"] ** 2 + rows[-1]["test_std"] ** 2) / 2))
    axes = ("--vqc-budget",)

    def paired(a: dict[int, float], b: dict[int, float]) -> dict:
        keys = sorted(set(a) & set(b))
        diff = np.array([a[k] - b[k] for k in keys])
        test = stats.ttest_rel([a[k] for k in keys], [b[k] for k in keys])
        wins = int((diff > 0).sum())
        return {
            "n_pairs": len(keys),
            "mean_difference": float(diff.mean()),
            "std_difference": float(diff.std(ddof=1)),
            "t_statistic": float(test.statistic),
            "p_value_two_sided": float(test.pvalue),
            "pairs_improved": wins,
        }

    vqc3_test = per_run(budget_summary, (3.0,), axes, "test_accuracy")
    vqc1_test = per_run(budget_summary, (1.0,), axes, "test_accuracy")
    baselines = {}
    for c in baseline_summary["candidates"]:
        label = (
            c["grid_point"].get("--classical-kind")
            or Path(c["grid_point"]["--config"]).stem
        )
        agg = c["aggregates"]
        test_by_rep = {run["repetition"]: run["test_accuracy"] for run in c["runs"]}
        baselines[label] = {
            "train_mean": agg["train_accuracy"]["mean"],
            "train_std": agg["train_accuracy"]["std"],
            "test_mean": agg["test_accuracy"]["mean"],
            "test_std": agg["test_accuracy"]["std"],
            "n_parameters": agg["n_parameters"]["mean"],
            "vqc_budget3_minus_baseline_test": paired(vqc3_test, test_by_rep),
        }
    return {
        "rows": rows,
        "trend_agrees": all(b >= a - 1e-12 for a, b in zip(test_means, test_means[1:]))
        and test_means[-1] - test_means[0] > pooled,
        "trend_detail": {
            "test_gain_budget3_vs_1": test_means[-1] - test_means[0],
            "pooled_std": pooled,
            "monotone_nondecreasing": all(
                b >= a - 1e-12 for a, b in zip(test_means, test_means[1:])
            ),
        },
        "quantitative": all(r["agrees"] for r in rows),
        "paired_budget3_vs_budget1_test": paired(vqc3_test, vqc1_test),
        "baselines": baselines,
    }


def count_runs_below_fci(sweeps_root: Path, sweep_id: str) -> dict:
    """Count completed runs whose energy fell below the exact (FCI) energy."""
    sweep_dir = sweeps_root / sweep_id
    status = json.loads((sweep_dir / "sweep_status.json").read_text())
    below = []
    for run in status["runs"]:
        metrics = json.loads((sweep_dir / run["metrics_path"]).read_text())
        if metrics["below_fci"]:
            below.append(
                {
                    "run_id": run["run_id"],
                    "budget": metrics["budget"],
                    "half_distance": metrics["half_distance_bohr"],
                    "energy_minus_fci": metrics["energy"] - metrics["fci_energy"],
                }
            )
    return {"n_runs": len(status["runs"]), "n_below_fci": len(below), "runs": below}


def compare_e5(vvqe: dict, noiseless: dict) -> dict:
    by_point = candidate_map(vvqe, "--half-distance", "--vvqe-budget")
    noiseless_by_d = candidate_map(noiseless, "--half-distance")
    rows, ordered = [], []
    for i, d in enumerate(paper.FIG7_HALF_DISTANCE):
        energies = {}
        for budget in (1.0, 1.5, 2.0, 2.5):
            c = by_point[(d, budget)]
            agg = c["aggregates"]
            ref = paper.FIG7_ENERGY[budget][i]
            energies[budget] = agg["energy"]["mean"]
            rows.append(
                {
                    "half_distance": d,
                    "budget": budget,
                    "energy_mean": agg["energy"]["mean"],
                    "energy_std": agg["energy"]["std"],
                    "energy_best_of_3": agg["energy"]["min"],
                    "paper_energy": ref,
                    "diff_mean": agg["energy"]["mean"] - ref,
                    "diff_best_of_3": agg["energy"]["min"] - ref,
                    "abs_error_fci_mean": agg["abs_error_fci"]["mean"],
                    "abs_error_hf_mean": agg["abs_error_hf"]["mean"],
                    "exact_negativity_mean": agg["exact_negativity"]["mean"],
                    "min_eigenvalue_sigma_min": agg["min_eigenvalue_sigma"]["min"],
                    "agrees_mean": abs(agg["energy"]["mean"] - ref) <= 0.02,
                    "agrees_best_of_3": abs(agg["energy"]["min"] - ref) <= 0.02,
                }
            )
        ordered.append(energies[2.5] < energies[1.5] < energies[1.0])
    noiseless_rows = [
        {
            "half_distance": d,
            "energy_mean": noiseless_by_d[(d,)]["aggregates"]["energy"]["mean"],
            "energy_std": noiseless_by_d[(d,)]["aggregates"]["energy"]["std"],
            "abs_error_fci_mean": noiseless_by_d[(d,)]["aggregates"]["abs_error_fci"][
                "mean"
            ],
            "abs_error_hf_mean": noiseless_by_d[(d,)]["aggregates"]["abs_error_hf"][
                "mean"
            ],
        }
        for d in paper.FIG7_HALF_DISTANCE
    ]
    diffs_mean = np.array([r["diff_mean"] for r in rows])
    diffs_best = np.array([r["diff_best_of_3"] for r in rows])
    return {
        "rows": rows,
        "noiseless_vqe": noiseless_rows,
        "trend_agrees": all(ordered),
        "points_ordered": int(sum(ordered)),
        "quantitative_mean": all(r["agrees_mean"] for r in rows),
        "quantitative_best_of_3": all(r["agrees_best_of_3"] for r in rows),
        "n_agree_mean": int(sum(r["agrees_mean"] for r in rows)),
        "n_agree_best_of_3": int(sum(r["agrees_best_of_3"] for r in rows)),
        "median_abs_diff_mean": float(np.median(np.abs(diffs_mean))),
        "median_abs_diff_best_of_3": float(np.median(np.abs(diffs_best))),
    }


def _md(report: dict) -> str:
    lines = [
        "# Comparison with arXiv:2609.11327 (generated by utils/compare_with_paper.py)",
        "",
    ]
    lines += [
        "All values are exact-simulation results; mean ± population std over repetitions. Scope labels follow the scientific-validity policy.",
        "",
    ]
    lines += [
        "## E0 — SDP minimum overhead (Table I)",
        "",
        "| Channel | SDP (ours) | Analytic | Paper 'Minimum' row | Paper 'VQEM' row |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in report["E0"]["rows"]:
        analytic = (
            f"{r['gamma_min_analytic']:.6f}"
            if r["gamma_min_analytic"] is not None
            else "—"
        )
        lines.append(
            f"| {r['channel']} | {r['gamma_min_sdp']:.6f} | {analytic} | {r['paper_minimum_row']} | {r['paper_vqem_row']} |"
        )
    lines += [
        "",
        "## E1 — VQEM Alg. 1, α = 1000 (Table I; 3 seeds, paper-accurate)",
        "",
        "| Channel | HS² ours | HS² paper | γ ours | γ SDP | gap | γ paper (VQEM / Min rows) | agree |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for r in report["E1"]["rows"]:
        lines.append(
            f"| {r['channel']} | {r['hs_mean']:.2e} ± {r['hs_std']:.1e} | {r['paper_hs']:.2e} | {r['gamma_mean']:.4f} ± {r['gamma_std']:.4f} | {r['gamma_sdp']:.4f} | {r['gap_to_sdp_percent']:+.2f}% | {r['paper_gamma_vqem_row']} / {r['paper_gamma_minimum_row']} | {'yes' if r['hs_agrees'] and r['gamma_agrees'] else 'no'} |"
        )
    lines += [
        "",
        "## E2 — penalty sweep, AD q = 0.3 (Fig. 4; 3 seeds, paper-accurate)",
        "",
        "| α | HS² ours | HS² paper | ratio | γ ours | γ paper | agree |",
        "|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in report["E2"]["rows"]:
        lines.append(
            f"| {r['alpha']} | {r['hs_mean']:.3e} | {r['paper_hs']:.3e} | {r['hs_ratio']:.2f} | {r['gamma_mean']:.4f} ± {r['gamma_std']:.4f} | {r['paper_gamma']:.4f} | {'yes' if r['agrees'] else 'no'} |"
        )
    lines += [
        "",
        f"Trend agreement: {report['E2']['trend_agrees']}; quantitative: {report['E2']['quantitative']}.",
    ]
    lines += [
        "",
        "## E3 — budget sweep, depolarizing p = 0.2 (Fig. 5; 3 seeds, paper-accurate)",
        "",
        "| Budget | Noisy ours | Noisy paper | Ideal ours | Ideal paper |",
        "|---:|---:|---:|---:|---:|",
    ]
    for n, i in zip(report["E3"]["noisy"]["rows"], report["E3"]["ideal"]["rows"]):
        lines.append(
            f"| {n['budget']:.1f} | {n['hs_mean']:.4f} ± {n['hs_std']:.4f} | {n['paper_hs']:.4f} | {i['hs_mean']:.2e} | {i['paper_hs']:.2e} |"
        )
    for k in ("noisy", "ideal"):
        e = report["E3"][k]
        lines.append(
            f"\n{k}: first budget with HS² < 1e-4 — ours {e['first_budget_hs_below_1e-4']}, paper {e['paper_first_budget_hs_below_1e-4']}."
        )
    lines += [
        "",
        f"Trend agreement: {report['E3']['trend_agrees']}; quantitative: {report['E3']['quantitative']}.",
    ]
    lines += [
        "",
        "## E4 — virtual quantum classifier (Fig. 6d; 10 paired repetitions, paper-accurate, distribution-level data match)",
        "",
        "| Budget | Params | Train ours | Train paper | Test ours | Test paper | within 1 paper std |",
        "|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in report["E4"]["rows"]:
        lines.append(
            f"| {r['budget']} | {r['n_parameters']:.0f} | {r['train_mean']:.3f} ± {r['train_std']:.3f} | {r['paper_train']:.3f} ± {r['paper_train_std']:.3f} | {r['test_mean']:.3f} ± {r['test_std']:.3f} | {r['paper_test']:.3f} ± {r['paper_test_std']:.3f} | {'yes' if r['agrees'] else 'no'} |"
        )
    p = report["E4"]["paired_budget3_vs_budget1_test"]
    lines += [
        "",
        f"Paired test accuracy, budget 3 − budget 1: mean {p['mean_difference']:+.3f} (sd {p['std_difference']:.3f}), t = {p['t_statistic']:.2f}, p = {p['p_value_two_sided']:.2g}, improved in {p['pairs_improved']}/{p['n_pairs']} pairs.",
    ]
    lines += [
        "",
        "Baselines (same 10 data sets):",
        "",
        "| Baseline | Params | Train | Test | VQC(γ=3) − baseline, test (paired) |",
        "|---|---:|---:|---:|---|",
    ]
    for name, b in report["E4"]["baselines"].items():
        q = b["vqc_budget3_minus_baseline_test"]
        lines.append(
            f"| {name} | {b['n_parameters']:.0f} | {b['train_mean']:.3f} ± {b['train_std']:.3f} | {b['test_mean']:.3f} ± {b['test_std']:.3f} | {q['mean_difference']:+.3f}, p = {q['p_value_two_sided']:.2g} |"
        )
    lines += [
        "",
        f"Trend agreement: {report['E4']['trend_agrees']}; quantitative: {report['E4']['quantitative']}.",
    ]
    lines += [
        "",
        "## E5 — virtual VQE, HeH⁺ (Fig. 7; 3 seeds, paper-accurate)",
        "",
        "| d (bohr) | γ | E mean ± std | best of 3 | paper | Δ mean | Δ best | |E−E_FCI| mean | min eig σ |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in report["E5"]["rows"]:
        lines.append(
            f"| {r['half_distance']} | {r['budget']} | {r['energy_mean']:.4f} ± {r['energy_std']:.4f} | {r['energy_best_of_3']:.4f} | {r['paper_energy']:.4f} | {r['diff_mean']:+.4f} | {r['diff_best_of_3']:+.4f} | {r['abs_error_fci_mean']:.4f} | {r['min_eigenvalue_sigma_min']:.1e} |"
        )
    e5 = report["E5"]
    lines += [
        "",
        f"Ordering E(2.5) < E(1.5) < E(1) holds at {e5['points_ordered']}/10 distances. Within 0.02 Ha of paper: {e5['n_agree_mean']}/40 (mean), {e5['n_agree_best_of_3']}/40 (best of 3). Median |Δ|: {e5['median_abs_diff_mean']:.4f} Ha (mean), {e5['median_abs_diff_best_of_3']:.4f} Ha (best of 3). Runs with energy below FCI (non-variational): {e5['below_fci']['n_below_fci']}/{e5['below_fci']['n_runs']}.",
        "",
        "Noiseless VQE ceiling (same ansatz):",
        "",
        "| d | E mean | |E−E_FCI| | |E−E_HF| |",
        "|---:|---:|---:|---:|",
    ]
    for r in e5["noiseless_vqe"]:
        lines.append(
            f"| {r['half_distance']} | {r['energy_mean']:.4f} ± {r['energy_std']:.4f} | {r['abs_error_fci_mean']:.4f} | {r['abs_error_hf_mean']:.4f} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare reproduction sweeps with the paper"
    )
    parser.add_argument("--sweeps-root", type=Path, required=True)
    parser.add_argument("--e0-run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.sweeps_root
    e0 = compare_e0(args.e0_run)
    report = {
        "E0": e0,
        "E1": compare_e1(load_summary(root, "e1_vqem_penalty"), e0),
        "E2": compare_e2(load_summary(root, "e2_vqem_penalty_sweep")),
        "E3": compare_e3(
            load_summary(root, "e3_vqem_budget_noisy"),
            load_summary(root, "e3_vqem_budget_ideal"),
        ),
        "E4": compare_e4(
            load_summary(root, "e4_vqc_budget"), load_summary(root, "e4_vqc_baselines")
        ),
        "E5": compare_e5(
            load_summary(root, "e5_vvqe"), load_summary(root, "e5_vqe_noiseless")
        ),
    }
    report["E5"]["below_fci"] = count_runs_below_fci(root, "e5_vvqe")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "comparison.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    (args.out / "comparison.md").write_text(_md(report), encoding="utf-8")
    print(f"Wrote {args.out / 'comparison.json'} and {args.out / 'comparison.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
