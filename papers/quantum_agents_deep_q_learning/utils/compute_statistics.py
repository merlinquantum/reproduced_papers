"""Statistics for the Phase 3 batches: per-config summaries, claim tests, paper comparison.

Reads only structured artifacts (``sweep_status.json``, ``metrics.json``,
``scores.npz``) from the batch folders given on the command line and writes
``statistics.json`` and ``statistics.md`` to ``--output-dir``.

Statistics (Fair Baseline Plan, LOG.md):

* episodes-to-solve per agent: 0-based solve episode index (reference-code
  ``env_solved_at``), unsolved agents censored at the episode cap;
* per config: solved fraction with a Wilson 95% interval, mean and median of
  the censored values with percentile-bootstrap 95% intervals (10000
  resamples, fixed seed);
* pairwise: two-sided Mann-Whitney U on censored values and a bootstrap 95%
  interval of the difference in medians; "no clear separation" when the
  interval contains 0;
* Frozen Lake: per-agent MAE to Q* over the 44 non-terminal pairs at time step
  10000 (last available value if the agent stopped earlier).

Example (from the paper folder)::

    python utils/compute_statistics.py --batches outdir/batches/e1_frozenlake \\
        outdir/batches/cartpole_optimal --output-dir results
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_SEED = 20261113
MAE_STEP = 10000

# Paper values. "text" values are quoted from the paper; "figure" values were
# read by eye from the dashed lines of Fig. 6 (300-dpi render, about +-20
# episodes) and from Fig. 3b; they are approximate.
PAPER_REFERENCE = {
    "e2e3_cartpole_optimal_full_original": {
        "statement": "solves in <= 600 episodes on average (text); Fig. 6a dashed line ~320",
        "mean_episodes": 320,
        "source": "figure",
    },
    "e4_cartpole_pqc_l10_original": {"mean_episodes": 635, "source": "figure"},
    "e4_cartpole_pqc_l15_original": {"mean_episodes": 690, "source": "figure"},
    "e4_cartpole_pqc_l20_original": {"mean_episodes": 770, "source": "figure"},
    "e4_cartpole_pqc_l25_original": {
        "statement": "best PQC on average, 500 episodes (text); Fig. 6a dashed line ~350",
        "mean_episodes": 350,
        "source": "figure",
    },
    "e4_cartpole_pqc_l30_original": {"mean_episodes": 940, "source": "figure"},
    "e4_cartpole_mlp_10_10_classical": {"mean_episodes": 1110, "source": "figure"},
    "e4_cartpole_mlp_15_15_classical": {"mean_episodes": 540, "source": "figure"},
    "e4_cartpole_mlp_20_20_classical": {
        "statement": "best NN on average, 250 episodes (text); Fig. 6b dashed line ~430",
        "mean_episodes": 430,
        "source": "figure",
    },
    "e4_cartpole_mlp_24_24_classical": {"mean_episodes": 415, "source": "figure"},
    "e4_cartpole_mlp_30_30_classical": {
        "statement": "best individual NN 186 episodes (Fig. 7); Fig. 6b dashed line ~250",
        "mean_episodes": 250,
        "best_agent": 186,
        "source": "figure",
    },
    "e4_cartpole_mlp_64_64_classical": {"mean_episodes": 345, "source": "figure"},
}
PAPER_BEST_PQC_AGENT = 206  # Fig. 7, 5 layers
PAPER_FL_MAE_AT_10000 = {5: 0.19, 10: 0.17, 15: 0.155}  # Fig. 3b, read by eye

COMPARISONS = [
    # (claim, label, config A, config B)
    (
        "C3",
        "optimal: full vs w/o re-uploading",
        "e2e3_cartpole_optimal_full_original",
        "e2e3_cartpole_optimal_no_reuploading_original",
    ),
    (
        "C3",
        "optimal: full vs w/o input weights",
        "e2e3_cartpole_optimal_full_original",
        "e2e3_cartpole_optimal_no_input_weights_original",
    ),
    (
        "C3",
        "sub-optimal: full vs w/o re-uploading",
        "e2e3_cartpole_suboptimal_full_original",
        "e2e3_cartpole_suboptimal_no_reuploading_original",
    ),
    (
        "C3",
        "sub-optimal: full vs w/o input weights",
        "e2e3_cartpole_suboptimal_full_original",
        "e2e3_cartpole_suboptimal_no_input_weights_original",
    ),
    (
        "C2",
        "optimal: trainable w_o vs fixed [0,1]",
        "e2e3_cartpole_optimal_full_original",
        "e2e3_cartpole_optimal_fixed_range_1_original",
    ),
    (
        "C2",
        "optimal: trainable w_o vs fixed [0,90]",
        "e2e3_cartpole_optimal_full_original",
        "e2e3_cartpole_optimal_fixed_range_90_original",
    ),
    (
        "C2",
        "optimal: trainable w_o vs fixed [0,180]",
        "e2e3_cartpole_optimal_full_original",
        "e2e3_cartpole_optimal_fixed_range_180_original",
    ),
    (
        "C2",
        "sub-optimal: trainable w_o vs fixed [0,1]",
        "e2e3_cartpole_suboptimal_full_original",
        "e2e3_cartpole_suboptimal_fixed_range_1_original",
    ),
    (
        "C2",
        "sub-optimal: trainable w_o vs fixed [0,90]",
        "e2e3_cartpole_suboptimal_full_original",
        "e2e3_cartpole_suboptimal_fixed_range_90_original",
    ),
    (
        "C2",
        "sub-optimal: trainable w_o vs fixed [0,180]",
        "e2e3_cartpole_suboptimal_full_original",
        "e2e3_cartpole_suboptimal_fixed_range_180_original",
    ),
    (
        "C5",
        "best paper PQC (25 layers) vs best paper NN (20,20)",
        "e4_cartpole_pqc_l25_original",
        "e4_cartpole_mlp_20_20_classical",
    ),
    (
        "C5",
        "iso-parameter: PQC 5 layers (62) vs NN (5,5) (67)",
        "e2e3_cartpole_optimal_full_original",
        "e4_cartpole_mlp_5_5_classical",
    ),
    (
        "C5",
        "iso-parameter: PQC 25 layers (302) vs NN (14,14) (310)",
        "e4_cartpole_pqc_l25_original",
        "e4_cartpole_mlp_14_14_classical",
    ),
    (
        "C1",
        "Frozen Lake: 5 vs 15 layers (episodes)",
        "e1_frozenlake_l5_original",
        "e1_frozenlake_l15_original",
    ),
    (
        "variant",
        "Cart Pole: arctan (paper) vs tanh (code) encoding",
        "e2e3_cartpole_optimal_full_original",
        "e2_cartpole_optimal_full_tanh_variant",
    ),
    (
        "variant",
        "Cart Pole: replay 10000 vs unbounded",
        "e2e3_cartpole_optimal_full_original",
        "e2_cartpole_optimal_full_unbounded_replay_variant",
    ),
    (
        "variant",
        "Frozen Lake: code ansatz vs Fig. 1 ring",
        "e1_frozenlake_l5_original",
        "e1_frozenlake_l5_ringansatz_variant",
    ),
]
DEPTH_SCAN = [
    "e2e3_cartpole_optimal_full_original",
    "e4_cartpole_pqc_l10_original",
    "e4_cartpole_pqc_l15_original",
    "e4_cartpole_pqc_l20_original",
    "e4_cartpole_pqc_l25_original",
    "e4_cartpole_pqc_l30_original",
]
WIDTH_SCAN = [
    "e4_cartpole_mlp_10_10_classical",
    "e4_cartpole_mlp_15_15_classical",
    "e4_cartpole_mlp_20_20_classical",
    "e4_cartpole_mlp_24_24_classical",
    "e4_cartpole_mlp_30_30_classical",
    "e4_cartpole_mlp_64_64_classical",
]
FL_DEPTHS = {
    5: "e1_frozenlake_l5_original",
    10: "e1_frozenlake_l10_original",
    15: "e1_frozenlake_l15_original",
}


def set_stem(stem: str, run_set: str) -> str:
    """Config stem of ``stem`` (written for the paper-table set) in ``run_set``."""
    if run_set == "paper-table":
        return stem
    for suffix in ("_original", "_classical"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)] + "_codefaithful"
    raise ValueError(f"No code-faithful counterpart for {stem}")


def wilson_interval(successes: int, trials: int) -> list[float]:
    """Wilson score 95% interval for a binomial proportion."""
    z = 1.959963984540054
    proportion = successes / trials
    centre = (proportion + z * z / (2 * trials)) / (1 + z * z / trials)
    half = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / trials + z * z / (4 * trials * trials)
        )
        / (1 + z * z / trials)
    )
    return [centre - half, centre + half]


def bootstrap_interval(
    values: np.ndarray, statistic, rng: np.random.Generator
) -> list[float]:
    """Percentile-bootstrap 95% interval of ``statistic`` over agents."""
    resamples = rng.choice(
        values, size=(BOOTSTRAP_RESAMPLES, len(values)), replace=True
    )
    estimates = statistic(resamples, axis=1)
    return [float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5))]


def median_difference_interval(
    first: np.ndarray, second: np.ndarray, rng
) -> list[float]:
    """Bootstrap 95% interval of median(first) - median(second)."""
    a = rng.choice(first, size=(BOOTSTRAP_RESAMPLES, len(first)), replace=True)
    b = rng.choice(second, size=(BOOTSTRAP_RESAMPLES, len(second)), replace=True)
    differences = np.median(a, axis=1) - np.median(b, axis=1)
    return [
        float(np.percentile(differences, 2.5)),
        float(np.percentile(differences, 97.5)),
    ]


def mae_at_step(matrix: np.ndarray, step: int) -> np.ndarray:
    """Per-agent MAE at ``step`` (0-based), last finite value if the agent stopped earlier."""
    values = []
    for row in matrix:
        finite = np.flatnonzero(np.isfinite(row[: step + 1]))
        values.append(row[finite[-1]])
    return np.asarray(values)


def load_runs(batch_dirs: list[Path]) -> dict[str, dict]:
    """Collect every COMPLETED, non-superseded run of the given batches, keyed by config stem."""
    runs: dict[str, dict] = {}
    for batch_dir in batch_dirs:
        status = json.loads((batch_dir / "sweep_status.json").read_text())
        for entry in status["runs"]:
            stem = Path(entry["config_file"]).stem
            if entry["status"] != "COMPLETED" or stem.endswith("_reduced"):
                continue
            if stem in runs:
                raise ValueError(f"Config {stem} appears in two batches")
            run_dir = batch_dir / entry["run_dir"]
            metrics = json.loads((run_dir / "metrics.json").read_text())
            arrays = np.load(run_dir / "scores.npz")
            runs[stem] = {
                "batch": batch_dir.name,
                "run_dir": str(run_dir),
                "metrics": metrics,
                "arrays": {key: arrays[key] for key in arrays.files},
            }
    return runs


def censored_episodes(run: dict) -> np.ndarray:
    cap = run["metrics"]["max_episodes"]
    values = [
        agent["solved_at_episode_index"] if agent["solved"] else cap
        for agent in run["metrics"]["agents"]
    ]
    return np.asarray(values, dtype=float)


def summarise_config(stem: str, run: dict, rng) -> dict:
    episodes = censored_episodes(run)
    agents = run["metrics"]["agents"]
    solved = [agent["solved_at_episode_index"] for agent in agents if agent["solved"]]
    record = {
        "config": stem,
        "batch": run["batch"],
        "run_dir": run["run_dir"],
        "label": run["metrics"]["experiment"]["label"],
        "trainable_parameters": run["metrics"]["trainable_parameters_per_agent"],
        "episode_cap": run["metrics"]["max_episodes"],
        "agents": len(agents),
        "solved": len(solved),
        "solved_fraction_ci95": wilson_interval(len(solved), len(agents)),
        "solved_episode_indices": [
            agent["solved_at_episode_index"] for agent in agents
        ],
        "mean_solved_only": float(np.mean(solved)) if solved else None,
        "best_agent": int(min(solved)) if solved else None,
        "mean_censored": float(np.mean(episodes)),
        "mean_censored_ci95": bootstrap_interval(episodes, np.mean, rng),
        "median_censored": float(np.median(episodes)),
        "median_censored_ci95": bootstrap_interval(episodes, np.median, rng),
    }
    if "q_star_mae_nonterminal_44" in run["arrays"]:
        mae = mae_at_step(run["arrays"]["q_star_mae_nonterminal_44"], MAE_STEP - 1)
        record["mae44_at_step_10000"] = mae.tolist()
        record["mae44_at_step_10000_mean"] = float(mae.mean())
        record["mae44_at_step_10000_sem"] = float(mae.std(ddof=1) / math.sqrt(len(mae)))
        start = run["arrays"]["q_star_mae_nonterminal_44"][:, 0]
        record["mae44_at_step_0_mean"] = float(start.mean())
    reference = PAPER_REFERENCE.get(stem)
    if reference is not None:
        record["paper"] = reference
        low, high = record["mean_censored_ci95"]
        record["paper_mean_inside_our_ci95"] = bool(
            low <= reference["mean_episodes"] <= high
        )
    return record


def compare(label, claim, first_stem, second_stem, runs, rng) -> dict:
    first = censored_episodes(runs[first_stem])
    second = censored_episodes(runs[second_stem])
    test = stats.mannwhitneyu(first, second, alternative="two-sided")
    interval = median_difference_interval(first, second, rng)
    separated = not (interval[0] <= 0.0 <= interval[1])
    return {
        "claim": claim,
        "comparison": label,
        "a": first_stem,
        "b": second_stem,
        "median_a": float(np.median(first)),
        "median_b": float(np.median(second)),
        "median_difference_a_minus_b_ci95": interval,
        "mann_whitney_u": float(test.statistic),
        "p_value": float(test.pvalue),
        "verdict": "separated" if separated else "no clear separation",
    }


def frozen_lake_mae(runs, rng) -> dict:
    per_depth = {
        depth: mae_at_step(
            runs[stem]["arrays"]["q_star_mae_nonterminal_44"], MAE_STEP - 1
        )
        for depth, stem in FL_DEPTHS.items()
    }
    shallow, deep = per_depth[5], per_depth[15]
    a = rng.choice(
        shallow, size=(BOOTSTRAP_RESAMPLES, len(shallow)), replace=True
    ).mean(axis=1)
    b = rng.choice(deep, size=(BOOTSTRAP_RESAMPLES, len(deep)), replace=True).mean(
        axis=1
    )
    differences = a - b
    test = stats.mannwhitneyu(shallow, deep, alternative="two-sided")
    return {
        "metric": "MAE to Q* over 44 non-terminal pairs at time step 10000",
        "mean_by_depth": {
            str(depth): float(values.mean()) for depth, values in per_depth.items()
        },
        "paper_by_depth_read_from_fig3b": {
            str(k): v for k, v in PAPER_FL_MAE_AT_10000.items()
        },
        "mean_difference_5_minus_15": float(shallow.mean() - deep.mean()),
        "mean_difference_5_minus_15_ci95": [
            float(np.percentile(differences, 2.5)),
            float(np.percentile(differences, 97.5)),
        ],
        "paper_difference_5_minus_15": 0.05,
        "mann_whitney_p_value": float(test.pvalue),
    }


def scan_test(stems: list[str], runs) -> dict:
    groups = [censored_episodes(runs[stem]) for stem in stems]
    test = stats.kruskal(*groups)
    means = {stem: float(group.mean()) for stem, group in zip(stems, groups)}
    return {
        "configs": stems,
        "kruskal_h": float(test.statistic),
        "p_value": float(test.pvalue),
        "lowest_mean_censored": min(means, key=means.get),
        "mean_censored": means,
    }


def to_markdown(result: dict) -> str:
    lines = [
        f"# Phase 3 statistics - {result['run_set']} run set",
        "",
        f"Generated from: {', '.join(result['batches'])}",
        "",
        "Config names are the paper-table stems; for the code-faithful set each row is the matching `_codefaithful` config.",
        "",
    ]
    lines += [
        "## Per configuration (episodes-to-solve, 0-based index, unsolved censored at cap)",
        "",
        "| Config | Label | Params | Solved | Mean (solved only) | Mean censored [95% CI] | Median censored [95% CI] | Best agent | Paper mean (source) | Paper inside CI |",
        "|---|---|---:|---:|---:|---|---|---:|---|---|",
    ]
    for record in result["configs"].values():
        paper = record.get("paper")
        paper_text = f"{paper['mean_episodes']} ({paper['source']})" if paper else "--"
        inside = record.get("paper_mean_inside_our_ci95", "--")
        mean_solved = (
            "--"
            if record["mean_solved_only"] is None
            else f"{record['mean_solved_only']:.0f}"
        )
        lines.append(
            f"| `{record['config']}` | {record['label']} | {record['trainable_parameters']} | "
            f"{record['solved']}/{record['agents']} | {mean_solved} | "
            f"{record['mean_censored']:.0f} [{record['mean_censored_ci95'][0]:.0f}, {record['mean_censored_ci95'][1]:.0f}] | "
            f"{record['median_censored']:.0f} [{record['median_censored_ci95'][0]:.0f}, {record['median_censored_ci95'][1]:.0f}] | "
            f"{record['best_agent']} | {paper_text} | {inside} |"
        )
    lines += [
        "",
        "## Pairwise comparisons",
        "",
        "| Claim | Comparison | Median A | Median B | Median diff A-B [95% CI] | MWU p | Verdict |",
        "|---|---|---:|---:|---|---:|---|",
    ]
    for item in result["comparisons"]:
        ci = item["median_difference_a_minus_b_ci95"]
        lines.append(
            f"| {item['claim']} | {item['comparison']} | {item['median_a']:.0f} | {item['median_b']:.0f} | "
            f"{item['median_a'] - item['median_b']:.0f} [{ci[0]:.0f}, {ci[1]:.0f}] | {item['p_value']:.3g} | {item['verdict']} |"
        )
    fl = result["frozen_lake_mae"]
    lines += [
        "",
        "## Frozen Lake MAE (C1)",
        "",
        f"Mean MAE44 at step 10000 by depth: {fl['mean_by_depth']} (paper, read from Fig. 3b: {fl['paper_by_depth_read_from_fig3b']}).",
        f"Difference 5 - 15 layers: {fl['mean_difference_5_minus_15']:.4f} [95% CI {fl['mean_difference_5_minus_15_ci95'][0]:.4f}, {fl['mean_difference_5_minus_15_ci95'][1]:.4f}], MWU p = {fl['mann_whitney_p_value']:.3g}; paper: ~0.05.",
        "",
        "## Scans (C5)",
        "",
    ]
    for name in ("pqc_depth_scan", "nn_width_scan"):
        scan = result[name]
        lines.append(
            f"- {name}: Kruskal-Wallis H = {scan['kruskal_h']:.2f}, p = {scan['p_value']:.3g}; lowest mean (censored): `{scan['lowest_mean_censored']}`."
        )
    lines += [
        "",
        f"Best individual agents: PQC `{result['best_individual']['pqc']['config']}` at {result['best_individual']['pqc']['episodes']} (paper 206, 5 layers); "
        f"NN `{result['best_individual']['nn']['config']}` at {result['best_individual']['nn']['episodes']} (paper 186, (30,30)).",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--batches", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--set",
        choices=["paper-table", "code-faithful"],
        required=True,
        help="Which run set the batches hold (config suffixes _original/_classical or _codefaithful)",
    )
    args = parser.parse_args()
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    loaded = load_runs(args.batches)
    if args.set == "paper-table":
        runs = loaded
    else:
        # Re-key code-faithful runs by their paper-table stem so the claim
        # tables, scans and paper references stay identical between sets.
        reverse = {}
        for stem in (
            list(PAPER_REFERENCE)
            + [s for c in COMPARISONS for s in c[2:]]
            + DEPTH_SCAN
            + WIDTH_SCAN
            + list(FL_DEPTHS.values())
        ):
            if stem.endswith(("_original", "_classical")):
                reverse[set_stem(stem, args.set)] = stem
        reverse.update(
            {
                set_stem(
                    f"e4_cartpole_mlp_{w}_classical", args.set
                ): f"e4_cartpole_mlp_{w}_classical"
                for w in ("5_5", "14_14")
            }
        )
        reverse.update(
            {
                set_stem(
                    "e1_frozenlake_mlp_classical", args.set
                ): "e1_frozenlake_mlp_classical"
            }
        )
        runs = {reverse[stem]: run for stem, run in loaded.items() if stem in reverse}
    configs = {stem: summarise_config(stem, runs[stem], rng) for stem in sorted(runs)}
    pqc = list(DEPTH_SCAN)
    nn = WIDTH_SCAN
    best_pqc = min(pqc, key=lambda stem: configs[stem]["best_agent"] or math.inf)
    best_nn = min(nn, key=lambda stem: configs[stem]["best_agent"] or math.inf)
    result = {
        "batches": [str(path) for path in args.batches],
        "bootstrap": {"resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED},
        "configs": configs,
        "run_set": args.set,
        "comparisons": [
            compare(label, claim, a, b, runs, rng)
            for claim, label, a, b in COMPARISONS
            if a in runs and b in runs
        ],
        "frozen_lake_mae": frozen_lake_mae(runs, rng),
        "pqc_depth_scan": scan_test(DEPTH_SCAN, runs),
        "nn_width_scan": scan_test(WIDTH_SCAN, runs),
        "best_individual": {
            "pqc": {
                "config": best_pqc,
                "episodes": configs[best_pqc]["best_agent"],
                "paper": PAPER_BEST_PQC_AGENT,
            },
            "nn": {
                "config": best_nn,
                "episodes": configs[best_nn]["best_agent"],
                "paper": 186,
            },
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tag = "" if args.set == "paper-table" else "_codefaithful"
    (args.output_dir / f"statistics{tag}.json").write_text(json.dumps(result, indent=2))
    (args.output_dir / f"statistics{tag}.md").write_text(to_markdown(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
