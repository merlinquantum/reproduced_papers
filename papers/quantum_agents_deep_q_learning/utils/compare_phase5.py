"""Phase 5: parameter-matched PQC vs NN comparison (frozen protocol in LOG.md).

Reads only structured artifacts (``sweep_status.json``, ``metrics.json``) of
the batches given with ``--batches`` and the selections written by
``utils/select_phase5.py``, and writes ``phase5_comparison.{json,md}`` and
``figures/fig_phase5_matched.png`` to ``--output-dir``.

Rows: the tuned matched pairs (PQC 5 layers vs NN (5,5); PQC 25 layers vs NN
(14,14)), the untuned matched NNs and the paper's own best arms for context,
and the MerLin Cart Pole model (partially commensurable: untuned, 82
parameters). Statistics: solved fraction, mean / median episodes to solve
(0-based, unsolved censored at 5000) with 95% percentile bootstrap intervals
(10000 resamples, seed 20261118), two-sided Mann-Whitney U, bootstrap 95%
interval of the median difference ("no clear separation" when it contains 0).

Example (from the paper folder)::

    python utils/compare_phase5.py --batches outdir/batches/e5_eval_nn \\
        outdir/batches/e5_eval_pqc outdir/batches/cf_e4_pqc_depth \\
        outdir/batches/cf_e4_nn_width outdir/batches/cf_cartpole_optimal \\
        outdir/batches/merlin_core --output-dir results
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

CAP = 5000
RESAMPLES = 10000
SEED = 20261118

# (row key, config stem, role, provenance, scope, matching basis, hardware)
ROWS = [
    (
        "PQC 5 layers, tuned",
        "e5_eval_pqc_l5",
        "paper method",
        "local",
        "paper-method-tuned",
        "62 params vs NN (5,5) 67; same 6-setting grid",
        "remote L4",
    ),
    (
        "NN (5,5), tuned",
        "e5_eval_mlp_5_5_classical",
        "added fair baseline",
        "local",
        "added-baseline-tuned",
        "67 params vs PQC 5 layers 62; same grid",
        "laptop CPU",
    ),
    (
        "PQC 25 layers, tuned (= paper setting)",
        "e4_cartpole_pqc_l25_codefaithful",
        "paper method",
        "local",
        "code-faithful",
        "302 params vs NN (14,14) 310; same grid",
        "remote L4",
    ),
    (
        "NN (14,14), tuned",
        "e5_eval_mlp_14_14_classical",
        "added fair baseline",
        "local",
        "added-baseline-tuned",
        "310 params vs PQC 25 layers 302; same grid",
        "laptop CPU",
    ),
    (
        "PQC 5 layers, paper setting",
        "e2e3_cartpole_optimal_full_codefaithful",
        "paper method (context)",
        "local",
        "code-faithful",
        "paper's Table 2/3 setting",
        "remote L4",
    ),
    (
        "NN (5,5), untuned",
        "e4_cartpole_mlp_5_5_codefaithful",
        "baseline before tuning (context)",
        "local",
        "added-baseline",
        "batch 16 / target 1, not tuned",
        "laptop CPU",
    ),
    (
        "NN (14,14), untuned",
        "e4_cartpole_mlp_14_14_codefaithful",
        "baseline before tuning (context)",
        "local",
        "added-baseline",
        "batch 16 / target 1, not tuned",
        "laptop CPU",
    ),
    (
        "NN (20,20), paper setting",
        "e4_cartpole_mlp_20_20_codefaithful",
        "paper baseline (context)",
        "local",
        "code-faithful",
        "562 params, not matched",
        "laptop CPU",
    ),
    (
        "NN (30,30), paper setting",
        "e4_cartpole_mlp_30_30_codefaithful",
        "paper baseline (context)",
        "local",
        "code-faithful",
        "1142 params, not matched",
        "laptop CPU",
    ),
    (
        "MerLin 5 layers (adaptation)",
        "e2e3_cartpole_optimal_full_merlin",
        "photonic adaptation",
        "local",
        "merlin-adaptation (simulated)",
        "82 params, untuned: PARTIALLY_COMMENSURABLE",
        "remote CPU",
    ),
]
PAIRS = [
    ("PQC 5 layers, tuned", "NN (5,5), tuned"),
    ("PQC 25 layers, tuned (= paper setting)", "NN (14,14), tuned"),
    ("MerLin 5 layers (adaptation)", "NN (5,5), tuned"),
    ("MerLin 5 layers (adaptation)", "PQC 5 layers, tuned"),
]
PAPER = {
    "PQC 5 layers, paper setting": "~320 (Fig. 6a)",
    "PQC 25 layers, tuned (= paper setting)": "~350 (Fig. 6a); 500 (text)",
    "NN (20,20), paper setting": "~430 (Fig. 6b); 250 (text)",
    "NN (30,30), paper setting": "~250 (Fig. 6b)",
}


def load(batches: list[Path]) -> dict:
    runs = {}
    for batch in batches:
        for entry in json.loads((batch / "sweep_status.json").read_text())["runs"]:
            stem = Path(entry["config_file"]).stem
            if entry["status"] != "COMPLETED":
                continue
            if stem in runs:
                raise ValueError(f"Duplicate run for {stem}")
            metrics = json.loads((batch / entry["metrics_path"]).read_text())
            runs[stem] = {"metrics": metrics, "run_dir": str(batch / entry["run_dir"])}
    return runs


def censored(metrics: dict) -> np.ndarray:
    values = [
        a["solved_at_episode_index"] if a["solved"] else CAP for a in metrics["agents"]
    ]
    if not all(math.isfinite(v) for v in values):
        raise ValueError("Non-finite metric")
    return np.asarray(values, dtype=float)


def interval(samples: np.ndarray) -> list[float]:
    return [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--batches", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rng = np.random.default_rng(SEED)
    runs = load(args.batches)
    table, values = {}, {}
    for key, stem, role, provenance, scope, basis, hardware in ROWS:
        if stem not in runs:
            raise KeyError(f"Missing expected run {stem}")
        metrics = runs[stem]["metrics"]
        episodes = censored(metrics)
        if sorted(a["seed"] for a in metrics["agents"]) != list(range(10)):
            raise ValueError(f"{stem} is not seeds 0-9")
        boot = rng.choice(episodes, size=(RESAMPLES, len(episodes)), replace=True)
        values[key] = episodes
        table[key] = {
            "config": stem,
            "run_dir": runs[stem]["run_dir"],
            "role": role,
            "provenance": provenance,
            "scope": scope,
            "matching_basis": basis,
            "hardware": hardware,
            "successful_runs": 1,
            "expected_runs": 1,
            "agents": len(episodes),
            "solved": int(sum(a["solved"] for a in metrics["agents"])),
            "mean_censored": float(episodes.mean()),
            "mean_ci95": interval(boot.mean(1)),
            "median_censored": float(np.median(episodes)),
            "median_ci95": interval(np.median(boot, 1)),
            "best_agent": min(
                [
                    a["solved_at_episode_index"]
                    for a in metrics["agents"]
                    if a["solved"]
                ],
                default=None,
            ),
            "trainable_parameters": metrics["trainable_parameters_per_agent"],
            "wall_clock_seconds": metrics["wall_clock_seconds"],
            "seconds_per_lockstep_iteration": metrics["seconds_per_lockstep_iteration"],
            "paper_value": PAPER.get(key, "--"),
        }
    pairs = []
    for first, second in PAIRS:
        a, b = values[first], values[second]
        da = rng.choice(a, size=(RESAMPLES, len(a)), replace=True)
        db = rng.choice(b, size=(RESAMPLES, len(b)), replace=True)
        difference = interval(np.median(da, 1) - np.median(db, 1))
        p_value = float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)
        pairs.append(
            {
                "a": first,
                "b": second,
                "median_a": float(np.median(a)),
                "median_b": float(np.median(b)),
                "median_difference_ci95": difference,
                "mann_whitney_p": p_value,
                "verdict": "no clear separation"
                if difference[0] <= 0 <= difference[1]
                else (
                    f"{first} faster"
                    if np.median(a) < np.median(b)
                    else f"{second} faster"
                ),
                "commensurability": "PARTIALLY_COMMENSURABLE"
                if "MerLin" in first
                else "COMMENSURABLE",
            }
        )
    result = {
        "rows": table,
        "pairs": pairs,
        "bootstrap": {"resamples": RESAMPLES, "seed": SEED},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "phase5_comparison.json").write_text(
        json.dumps(result, indent=2)
    )

    lines = [
        "# Phase 5: parameter-matched comparison (Cart Pole, code-faithful settings)",
        "",
        "Episodes to solve: 0-based, unsolved censored at 5000; 10 agents (seeds 0-9) per row; "
        "95% bootstrap intervals. All values locally executed; paper values read from figures "
        "or text where given.",
        "",
        "| System | Role | Provenance | Scope | Successful / expected runs | Solved | Mean [95% CI] | "
        "Median [95% CI] | Best agent | Matching basis | Params | Wall-clock (s), hardware | Paper |",
        "|---|---|---|---|---:|---:|---|---|---:|---|---:|---|---|",
    ]
    for key, row in table.items():
        lines.append(
            f"| {key} | {row['role']} | {row['provenance']} | {row['scope']} | "
            f"{row['successful_runs']} / {row['expected_runs']} | {row['solved']}/10 | "
            f"{row['mean_censored']:.0f} [{row['mean_ci95'][0]:.0f}, {row['mean_ci95'][1]:.0f}] | "
            f"{row['median_censored']:.0f} [{row['median_ci95'][0]:.0f}, {row['median_ci95'][1]:.0f}] | "
            f"{row['best_agent']} | {row['matching_basis']} | {row['trainable_parameters']} | "
            f"{row['wall_clock_seconds']:.0f}, {row['hardware']} | {row['paper_value']} |"
        )
    lines += [
        "",
        "| Pair | Median A | Median B | Median difference A-B [95% CI] | MWU p | Verdict | Status |",
        "|---|---:|---:|---|---:|---|---|",
    ]
    for pair in pairs:
        ci = pair["median_difference_ci95"]
        lines.append(
            f"| {pair['a']} vs {pair['b']} | {pair['median_a']:.0f} | {pair['median_b']:.0f} | "
            f"{pair['median_a'] - pair['median_b']:.0f} [{ci[0]:.0f}, {ci[1]:.0f}] | "
            f"{pair['mann_whitney_p']:.3g} | {pair['verdict']} | {pair['commensurability']} |"
        )
    text = "\n".join(lines) + "\n"
    (args.output_dir / "phase5_comparison.md").write_text(text)

    figure, axis = plt.subplots(figsize=(11, 4.5))
    keys = list(table)
    colours = [
        "tab:blue" if "PQC" in k else "tab:red" if "MerLin" in k else "tab:gray"
        for k in keys
    ]
    for index, (key, colour) in enumerate(zip(keys, colours)):
        row = table[key]
        axis.scatter(
            np.full(10, index) + rng.uniform(-0.15, 0.15, 10),
            values[key],
            s=12,
            color=colour,
            alpha=0.6,
        )
        axis.errorbar(
            index,
            row["median_censored"],
            yerr=[
                [row["median_censored"] - row["median_ci95"][0]],
                [row["median_ci95"][1] - row["median_censored"]],
            ],
            fmt="D",
            color="black",
            capsize=4,
        )
    axis.set_yscale("log")
    axis.set_xticks(range(len(keys)))
    axis.set_xticklabels([k.replace(", ", "\n") for k in keys], fontsize=7)
    axis.set_ylabel("episodes to solve (log; 5000 = not solved)")
    axis.axvline(3.5, color="lightgray")
    figure.suptitle(
        "Phase 5: parameter-matched PQC vs NN (first 4 columns, equal 6-setting tuning grid) and "
        "context rows\nagents (dots), median with 95% bootstrap CI (diamonds); locally executed, "
        "10 agents; MerLin = simulated adaptation",
        fontsize=9,
    )
    figure.tight_layout()
    (args.output_dir / "figures").mkdir(exist_ok=True)
    figure.savefig(args.output_dir / "figures" / "fig_phase5_matched.png", dpi=140)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
