"""Paired statistical analysis of the powered (n=100) CV runs.

Recreates, as committed and re-runnable code, the paired-fold analysis
behind the README/LOG statistical tables (previously computed in
throwaway container scratch): for each router threshold, the paired
AUCPR difference (MoE - XGBoost baseline) across CV folds with mean,
95% CI, paired t-test, Wilcoxon signed-rank, win-rate, and worst folds.

Usage (from the paper directory):

    # one run
    python utils/analyze_powered_runs.py outdir/moe_gate_powered/run_<STAMP>

    # A/B two runs of the same config (e.g. pre/post bug-fix)
    python utils/analyze_powered_runs.py RUN_A RUN_B --labels pre,post

    # curate: write the compact analysis JSON next to figures
    python utils/analyze_powered_runs.py RUN --json-out results/gate_powered_analysis.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats


def load_paired(run_dir: Path) -> dict:
    """Per-fold paired AUCPR arrays: baseline plus MoE at each threshold."""
    metrics = json.loads((run_dir / "metrics.json").read_text())
    splits = metrics["per_split"]
    thresholds = [str(t) for t in metrics["router_thresholds"]]
    baseline = np.array([s["xgboost_baseline"]["aucpr"] for s in splits])
    moe = {
        t: np.array([s["thresholds"][t]["aucpr"] for s in splits]) for t in thresholds
    }
    return {"baseline": baseline, "moe": moe, "n": len(splits)}


def threshold_stats(diff: np.ndarray) -> dict:
    """Mean/CI/tests/win-rate for one paired-difference vector."""
    n = diff.size
    mean = float(diff.mean())
    sem = float(diff.std(ddof=1) / np.sqrt(n))
    t_res = stats.ttest_rel(diff, np.zeros_like(diff))
    try:
        w_res = stats.wilcoxon(diff)
        wilcoxon_p = float(w_res.pvalue)
    except ValueError:  # all-zero differences
        wilcoxon_p = 1.0
    return {
        "n": n,
        "mean_diff": mean,
        "ci95": [mean - 1.96 * sem, mean + 1.96 * sem],
        "median_diff": float(np.median(diff)),
        "paired_t_p": float(t_res.pvalue),
        "wilcoxon_p": wilcoxon_p,
        "win_rate": float((diff > 0).mean()),
        "worst_fold_diff": float(diff.min()),
        "best_fold_diff": float(diff.max()),
        # Per-fold paired differences, kept so distribution figures can be
        # regenerated from the curated JSON without the raw run directory.
        "fold_diffs": [round(float(d), 6) for d in diff],
    }


def analyze(run_dir: Path) -> dict:
    data = load_paired(run_dir)
    return {
        "run": run_dir.name,
        "config_dir": run_dir.parent.name,
        "n_folds": data["n"],
        "baseline_aucpr_mean": float(data["baseline"].mean()),
        "per_threshold": {
            t: threshold_stats(m - data["baseline"]) for t, m in data["moe"].items()
        },
    }


def render(analysis: dict) -> str:
    lines = [
        f"== {analysis['config_dir']}/{analysis['run']} "
        f"(n={analysis['n_folds']}, baseline AUCPR "
        f"{analysis['baseline_aucpr_mean']:.4f}) ==",
        f"{'gamma':>6} {'mean diff':>10} {'95% CI':>20} {'t p':>8} "
        f"{'wilcox p':>9} {'win%':>6} {'worst':>8}",
    ]
    for t, s in analysis["per_threshold"].items():
        ci = f"[{s['ci95'][0]:+.4f},{s['ci95'][1]:+.4f}]"
        lines.append(
            f"{t:>6} {s['mean_diff']:>+10.4f} {ci:>20} {s['paired_t_p']:>8.3f} "
            f"{s['wilcoxon_p']:>9.3f} {100 * s['win_rate']:>5.0f} "
            f"{s['worst_fold_diff']:>+8.3f}"
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument(
        "--labels", default=None, help="comma-separated labels matching the runs"
    )
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    labels = args.labels.split(",") if args.labels else [r.name for r in args.runs]
    analyses = []
    for label, run in zip(labels, args.runs):
        analysis = analyze(run)
        analysis["label"] = label
        analyses.append(analysis)
        print(render(analysis))
        print()

    if len(analyses) == 2:
        a, b = analyses
        print(f"== paired delta per threshold: {b['label']} - {a['label']} ==")
        for t in a["per_threshold"]:
            da, db = a["per_threshold"][t], b["per_threshold"][t]
            print(
                f"  gamma={t}: mean diff {da['mean_diff']:+.4f} -> "
                f"{db['mean_diff']:+.4f}  (t p {da['paired_t_p']:.3f} -> "
                f"{db['paired_t_p']:.3f}, worst {da['worst_fold_diff']:+.3f} -> "
                f"{db['worst_fold_diff']:+.3f})"
            )

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        payload = analyses[0] if len(analyses) == 1 else {"runs": analyses}
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
