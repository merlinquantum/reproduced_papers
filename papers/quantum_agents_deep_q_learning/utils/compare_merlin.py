"""Phase 4: compare every MerLin run with its code-faithful qubit counterpart.

Reads ``sweep_status.json`` / ``metrics.json`` / ``scores.npz`` of the MerLin
batches and the code-faithful qubit batches (passed on the command line), and
writes ``merlin_comparison.json``, ``merlin_comparison.md`` and
``fig_merlin_vs_qubit.png`` to ``--output-dir``.

Statistics follow the Phase 4 Adaptation Plan (LOG.md): per config solved
fraction, mean episodes to solve (0-based, unsolved censored at 5000) with a
95% percentile bootstrap interval (10000 resamples, fixed seed), two-sided
Mann-Whitney U on censored episodes ("no clear separation" when p > 0.05),
Frozen Lake MAE44 at time step 10000. The preservation criteria per claim
were declared in the plan before the runs and are evaluated here verbatim.

Example (from the paper folder)::

    python utils/compare_merlin.py --merlin-batches outdir/batches/merlin_core \\
        outdir/batches/merlin_fl_reupload --qubit-batches outdir/batches/cf_frozenlake \\
        outdir/batches/cf_cartpole_optimal outdir/batches/cf_cartpole_suboptimal \\
        --output-dir results
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from lib.metrics import padded_score_matrix  # noqa: E402

BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_SEED = 20261117
CAP = 5000
ARMS = (
    "full",
    "no_reuploading",
    "no_input_weights",
    "fixed_range_1",
    "fixed_range_90",
    "fixed_range_180",
)
SETTINGS = ("optimal", "suboptimal")


def load_runs(batch_dirs: list[Path]) -> dict[str, dict]:
    runs = {}
    for batch_dir in batch_dirs:
        for entry in json.loads((batch_dir / "sweep_status.json").read_text())["runs"]:
            if entry["status"] != "COMPLETED":
                continue
            stem = Path(entry["config_file"]).stem
            run_dir = batch_dir / entry["run_dir"]
            arrays = np.load(run_dir / "scores.npz")
            runs[stem] = {
                "run_dir": str(run_dir),
                "metrics": json.loads((run_dir / "metrics.json").read_text()),
                "arrays": {key: arrays[key] for key in arrays.files},
            }
    return runs


def censored(run: dict) -> np.ndarray:
    return np.asarray(
        [
            a["solved_at_episode_index"] if a["solved"] else CAP
            for a in run["metrics"]["agents"]
        ],
        dtype=float,
    )


def bootstrap_mean(values: np.ndarray, rng) -> list[float]:
    samples = rng.choice(
        values, size=(BOOTSTRAP_RESAMPLES, len(values)), replace=True
    ).mean(1)
    return [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))]


def mae_at_10000(run: dict) -> float | None:
    matrix = run["arrays"].get("q_star_mae_nonterminal_44")
    if matrix is None:
        return None
    return float(
        np.mean([row[np.flatnonzero(np.isfinite(row[:10000]))[-1]] for row in matrix])
    )


def summary(run: dict, rng) -> dict:
    values = censored(run)
    return {
        "run_dir": run["run_dir"],
        "solved": int(sum(a["solved"] for a in run["metrics"]["agents"])),
        "agents": len(values),
        "mean_censored": float(values.mean()),
        "mean_censored_ci95": bootstrap_mean(values, rng),
        "median_censored": float(np.median(values)),
        "trainable_parameters": run["metrics"]["trainable_parameters_per_agent"],
        "mae44_at_10000": mae_at_10000(run),
        "seconds_per_lockstep_iteration": run["metrics"][
            "seconds_per_lockstep_iteration"
        ],
        "wall_clock_seconds": run["metrics"]["wall_clock_seconds"],
        "seeds": [a["seed"] for a in run["metrics"]["agents"]],
    }


def pair(merlin: dict, qubit: dict, rng) -> dict:
    a, b = censored(merlin), censored(qubit)
    if np.all(a == a[0]) and np.all(b == a[0]):
        p_value = 1.0  # identical constant samples (e.g. 0/10 vs 0/10)
    else:
        p_value = float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)
    return {
        "merlin": summary(merlin, rng),
        "qubit": summary(qubit, rng),
        "mann_whitney_p": p_value,
        "agreement": "no clear separation"
        if p_value > 0.05
        else (
            "MerLin slower"
            if np.median(a) > np.median(b) or a.mean() > b.mean()
            else "MerLin faster"
        ),
    }


def criteria(results: dict) -> dict:
    """Preservation criteria declared in the Phase 4 Adaptation Plan (LOG.md)."""
    cp = results["pairs"]

    def solved(name: str) -> int:
        return cp[name]["merlin"]["solved"]

    def median(name: str) -> float:
        return cp[name]["merlin"]["median_censored"]

    fl = [solved(f"e1_frozenlake_l{depth}_merlin") for depth in (5, 10, 15)]
    fl_means = [
        cp[f"e1_frozenlake_l{depth}_merlin"]["merlin"]["mean_censored"]
        for depth in (5, 10, 15)
    ]
    out = {
        "C1 planned (basis phases once): all depths solve >= 8/10 and depth does not hurt": {
            "observed": f"solved {fl}, mean episodes {[round(m) for m in fl_means]}",
            "met": all(s >= 8 for s in fl) and fl_means[2] <= fl_means[0],
        },
        "C2: fixed [0,1] 0/10 (both sets); trainable w_o and fixed [0,90] >= 8/10 (best set)": {
            "observed": (
                f"[0,1] {solved('e2e3_cartpole_optimal_fixed_range_1_merlin')}/10 and "
                f"{solved('e2e3_cartpole_suboptimal_fixed_range_1_merlin')}/10; w_o "
                f"{solved('e2e3_cartpole_optimal_full_merlin')}/10; [0,90] "
                f"{solved('e2e3_cartpole_optimal_fixed_range_90_merlin')}/10"
            ),
            "met": solved("e2e3_cartpole_optimal_fixed_range_1_merlin") == 0
            and solved("e2e3_cartpole_suboptimal_fixed_range_1_merlin") == 0
            and solved("e2e3_cartpole_optimal_full_merlin") >= 8
            and solved("e2e3_cartpole_optimal_fixed_range_90_merlin") >= 8,
        },
        "C3: sub-optimal set, both ablations not faster than full (median)": {
            "observed": (
                f"median full {median('e2e3_cartpole_suboptimal_full_merlin'):.0f}, w/o re-uploading "
                f"{median('e2e3_cartpole_suboptimal_no_reuploading_merlin'):.0f}, w/o w_d "
                f"{median('e2e3_cartpole_suboptimal_no_input_weights_merlin'):.0f}"
            ),
            "met": median("e2e3_cartpole_suboptimal_no_reuploading_merlin")
            >= median("e2e3_cartpole_suboptimal_full_merlin")
            and median("e2e3_cartpole_suboptimal_no_input_weights_merlin")
            >= median("e2e3_cartpole_suboptimal_full_merlin"),
        },
        "C4: full arm, best set, >= 8/10": {
            "observed": f"{solved('e2e3_cartpole_optimal_full_merlin')}/10",
            "met": solved("e2e3_cartpole_optimal_full_merlin") >= 8,
        },
    }
    if "e1_frozenlake_l5_merlin_reupload" in cp:
        dl = [
            solved(f"e1_frozenlake_l{depth}_merlin_reupload") for depth in (5, 10, 15)
        ]
        dm = [
            cp[f"e1_frozenlake_l{depth}_merlin_reupload"]["merlin"]["mean_censored"]
            for depth in (5, 10, 15)
        ]
        out[
            "C1 diagnostic (basis phases re-uploaded), same criterion; not the planned model"
        ] = {
            "observed": f"solved {dl}, mean episodes {[round(m) for m in dm]}",
            "met": all(s >= 8 for s in dl) and dm[2] <= dm[0],
        }
    return out


def plot(results: dict, merlin_runs: dict, qubit_runs: dict, path: Path) -> None:
    names = [f"e1_frozenlake_l{depth}" for depth in (5, 10, 15)] + [
        f"e2e3_cartpole_{s}_{a}" for s in SETTINGS for a in ARMS
    ]
    figure, axes = plt.subplots(5, 3, figsize=(13, 15))
    for axis, base in zip(axes.flat, names):
        max_score = 1.0 if base.startswith("e1_") else 200.0
        curves = [
            (qubit_runs[base + "_codefaithful"], "qubit (code-faithful)", "tab:blue"),
            (merlin_runs[base + "_merlin"], "MerLin (planned)", "tab:red"),
        ]
        variant_colours = {
            "_merlin_reupload": ("MerLin re-uploaded (diagnostic)", "tab:green"),
            "_merlin_dualrail": ("MerLin dual-rail input (design study)", "tab:purple"),
            "_merlin_8modes": (
                "MerLin 8 modes / 4 photons (design study)",
                "tab:orange",
            ),
        }
        for suffix, (label, colour) in variant_colours.items():
            if base + suffix in merlin_runs:
                curves.append((merlin_runs[base + suffix], label, colour))
        for run, label, colour in curves:
            rows = [r[np.isfinite(r)].tolist() for r in run["arrays"]["scores"]]
            matrix = padded_score_matrix(rows, CAP)
            solved = sum(a["solved"] for a in run["metrics"]["agents"])
            axis.plot(
                matrix.mean(0),
                color=colour,
                linewidth=0.8,
                label=f"{label}, {solved}/10",
            )
        axis.set_title(
            base.replace("e2e3_cartpole_", "CP ").replace("e1_frozenlake_", "FL "),
            fontsize=9,
        )
        axis.set_ylim(0, max_score * 1.05)
        axis.legend(fontsize=6, loc="lower right")
    figure.suptitle(
        "Phase 4 MerLin adaptation vs qubit (code-faithful): mean score over 10 agents "
        "(padded with last score)\nlabels: merlin-adaptation (simulated, 6 modes / 3 photons, "
        "threshold detectors, no post-selection)",
        fontsize=10,
    )
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)


def to_markdown(results: dict) -> str:
    lines = [
        "# Phase 4: MerLin adaptation vs qubit model (code-faithful settings)",
        "",
        "Episodes to solve: 0-based index, unsolved agents censored at 5000; 10 agents (seeds 0-9);",
        "95% bootstrap intervals; Mann-Whitney U on censored episodes. Label: merlin-adaptation",
        "(simulated linear optics, not hardware).",
        "",
        "| Config | MerLin solved | MerLin mean [95% CI] | Qubit solved | Qubit mean [95% CI] | MWU p | Agreement | MerLin params | Qubit params | MAE44 @10k MerLin / qubit |",
        "|---|---:|---|---:|---|---:|---|---:|---:|---|",
    ]
    for name, item in results["pairs"].items():
        m, q = item["merlin"], item["qubit"]
        mae = (
            "--"
            if m["mae44_at_10000"] is None
            else f"{m['mae44_at_10000']:.3f} / {q['mae44_at_10000']:.3f}"
        )
        lines.append(
            f"| `{name}` | {m['solved']}/10 | {m['mean_censored']:.0f} [{m['mean_censored_ci95'][0]:.0f}, "
            f"{m['mean_censored_ci95'][1]:.0f}] | {q['solved']}/10 | {q['mean_censored']:.0f} "
            f"[{q['mean_censored_ci95'][0]:.0f}, {q['mean_censored_ci95'][1]:.0f}] | "
            f"{item['mann_whitney_p']:.3g} | {item['agreement']} | {m['trainable_parameters']} | "
            f"{q['trainable_parameters']} | {mae} |"
        )
    lines += [
        "",
        "## Preservation criteria (declared in the Phase 4 Adaptation Plan)",
        "",
        "| Criterion | Observed | Met |",
        "|---|---|---|",
    ]
    for criterion, value in results["criteria"].items():
        lines.append(
            f"| {criterion} | {value['observed']} | {'yes' if value['met'] else 'no'} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--merlin-batches", type=Path, nargs="+", required=True)
    parser.add_argument("--qubit-batches", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    merlin_runs = load_runs(args.merlin_batches)
    qubit_runs = load_runs(args.qubit_batches)
    pairs = {}
    for name in sorted(merlin_runs):
        base = name.split("_merlin")[0]
        qubit_name = base + "_codefaithful"
        if qubit_name not in qubit_runs:
            raise KeyError(f"No qubit counterpart {qubit_name} for {name}")
        pairs[name] = pair(merlin_runs[name], qubit_runs[qubit_name], rng)
    results = {
        "merlin_batches": [str(p) for p in args.merlin_batches],
        "qubit_batches": [str(p) for p in args.qubit_batches],
        "bootstrap": {"resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED},
        "pairs": pairs,
    }
    results["criteria"] = criteria(results)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "merlin_comparison.json").write_text(
        json.dumps(results, indent=2)
    )
    (args.output_dir / "merlin_comparison.md").write_text(to_markdown(results))
    figure_dir = args.output_dir / "figures"
    figure_dir.mkdir(exist_ok=True)
    plot(results, merlin_runs, qubit_runs, figure_dir / "fig_merlin_vs_qubit.png")
    for batch in args.merlin_batches:
        (batch / "figures").mkdir(exist_ok=True)
        plot(
            results,
            merlin_runs,
            qubit_runs,
            batch / "figures" / "fig_merlin_vs_qubit.png",
        )
    print(to_markdown(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
