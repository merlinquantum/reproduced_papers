"""Paper-style Figs. 3-7 from the Phase 3 batch folders.

Reads ``sweep_status.json``, ``scores.npz`` and ``metrics.json`` of the
batches passed with ``--batches`` (no run directory is hard-coded) and saves
every figure to ``--output-dir`` and, as a copy, into the batch folder that
holds the plotted runs (the first batch named for that figure).

Curves follow the paper / reference plotting code: per-episode score averaged
over agents, each agent padded with its last score after it stopped, band =
+- one standard deviation clipped to the score range (Figs. 3a-6). Fig. 3b
shows the MAE to Q* over the 44 non-terminal pairs (+- s.e.m.). Every figure
title carries its result-scope label.

Example (from the paper folder)::

    python utils/plot_paper_figures.py --set paper-table --batches outdir/batches/* --output-dir results/figures
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from lib.metrics import padded_score_matrix  # noqa: E402

SCOPES = {
    "paper-table": "paper-accurate (Appendix B table values), 10 agents (seeds 0-9), this reproduction",
    "code-faithful": "code-faithful (reference-code replay / update settings), 10 agents (seeds 0-9), this reproduction",
}
SCOPE = SCOPES["paper-table"]
NAME_TAG = ""


def paper_table_stem(stem: str) -> str:
    """Key a code-faithful run by its paper-table config stem."""
    if not stem.endswith("_codefaithful"):
        return stem
    base = stem[: -len("_codefaithful")]
    return base + ("_classical" if "_mlp" in base else "_original")


def load_runs(batch_dirs: list[Path]) -> dict[str, dict]:
    """Completed, non-superseded runs keyed by config stem."""
    runs = {}
    for batch_dir in batch_dirs:
        status_path = batch_dir / "sweep_status.json"
        if not status_path.exists():
            continue
        for entry in json.loads(status_path.read_text())["runs"]:
            stem = Path(entry["config_file"]).stem
            if entry["status"] != "COMPLETED" or stem.endswith("_reduced"):
                continue
            run_dir = batch_dir / entry["run_dir"]
            arrays = np.load(run_dir / "scores.npz")
            runs[stem] = {
                "batch_dir": batch_dir,
                "metrics": json.loads((run_dir / "metrics.json").read_text()),
                "arrays": {key: arrays[key] for key in arrays.files},
            }
    return runs


def score_matrix(run: dict, length: int) -> np.ndarray:
    scores = run["arrays"]["scores"]
    rows = [row[np.isfinite(row)].tolist() for row in scores]
    return padded_score_matrix(rows, length)


def mean_solve(run: dict) -> float:
    cap = run["metrics"]["max_episodes"]
    return float(
        np.mean(
            [
                a["solved_at_episode_index"] if a["solved"] else cap
                for a in run["metrics"]["agents"]
            ]
        )
    )


def draw_curve(axis, run, length, label, colour, max_score):
    matrix = score_matrix(run, length)
    mean, std = matrix.mean(axis=0), matrix.std(axis=0)
    episodes = np.arange(length)
    axis.plot(episodes, mean, color=colour, label=label, linewidth=1.0)
    axis.fill_between(
        episodes,
        np.clip(mean - std, 0, max_score),
        np.clip(mean + std, 0, max_score),
        color=colour,
        alpha=0.2,
    )


def save(figure, name: str, output_dir: Path, batch_dir: Path) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}{NAME_TAG}.png"
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    copy_dir = batch_dir / "figures"
    copy_dir.mkdir(exist_ok=True)
    shutil.copy2(path, copy_dir / path.name)
    return [str(path), str(copy_dir / path.name)]


def solved_text(run) -> str:
    agents = run["metrics"]["agents"]
    return f"{sum(a['solved'] for a in agents)}/{len(agents)} solved"


def figure_3(runs, output_dir):
    depths = {
        5: "e1_frozenlake_l5_original",
        10: "e1_frozenlake_l10_original",
        15: "e1_frozenlake_l15_original",
    }
    colours = {5: "tab:blue", 10: "tab:green", 15: "tab:orange"}
    figure, (left, right) = plt.subplots(1, 2, figsize=(11, 4))
    for depth, stem in depths.items():
        run = runs[stem]
        draw_curve(
            left, run, 5000, f"{depth} layers ({solved_text(run)})", colours[depth], 1.0
        )
        mae = run["arrays"]["q_star_mae_nonterminal_44"][:, :10000]
        # Agents that stopped keep their last MAE (they no longer change).
        filled = np.array(
            [np.where(np.isfinite(r), r, r[np.isfinite(r)][-1]) for r in mae]
        )
        mean = filled.mean(axis=0)
        sem = filled.std(axis=0, ddof=1) / np.sqrt(len(filled))
        steps = np.arange(filled.shape[1])
        right.plot(steps, mean, color=colours[depth], label=f"{depth} layers")
        right.fill_between(
            steps, mean - sem, mean + sem, color=colours[depth], alpha=0.25
        )
    left.set(xlabel="Episode", ylabel="Score", title="(a) average score")
    right.set(
        xlabel="Time step",
        ylabel="MAE(Q, Q*) over 44 non-terminal pairs",
        title="(b) MAE to optimal Q-values",
    )
    left.legend(fontsize=8)
    right.legend(fontsize=8)
    figure.suptitle(f"Fig. 3 Frozen Lake - {SCOPE}", fontsize=10)
    return save(figure, "fig3_frozenlake", output_dir, runs[depths[5]]["batch_dir"])


def ablation_figure(runs, output_dir, name, title, arms):
    figure, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    batch_dir = None
    for axis, setting in zip(axes, ("optimal", "suboptimal")):
        for suffix, label, colour in arms:
            stem = f"e2e3_cartpole_{setting}_{suffix}_original"
            run = runs[stem]
            batch_dir = batch_dir or run["batch_dir"]
            draw_curve(axis, run, 5000, f"{label} ({solved_text(run)})", colour, 200.0)
        axis.set(
            xlabel="Episode",
            title=f"({'a' if setting == 'optimal' else 'b'}) {'best' if setting == 'optimal' else 'sub-optimal'} hyperparameters",
        )
        axis.legend(fontsize=8, loc="lower right")
    axes[0].set_ylabel("Score")
    figure.suptitle(f"{title} - {SCOPE}", fontsize=10)
    return save(figure, name, output_dir, batch_dir)


def scan_figure(runs, output_dir, name, title, stems_labels):
    figure, axes = plt.subplots(3, 2, figsize=(9, 8), sharex=True, sharey=True)
    for axis, (stem, label) in zip(axes.flat, stems_labels):
        run = runs[stem]
        draw_curve(axis, run, 1000, f"{label}, {solved_text(run)}", "tab:blue", 200.0)
        mean = mean_solve(run)
        if mean <= 1000:
            axis.axvline(mean, color="grey", linestyle="--")
        axis.text(
            0.98,
            0.05,
            f"mean solve {mean:.0f}",
            transform=axis.transAxes,
            ha="right",
            fontsize=8,
        )
        axis.legend(fontsize=7, loc="upper left")
    for axis in axes[-1]:
        axis.set_xlabel("Episode")
    for axis in axes[:, 0]:
        axis.set_ylabel("Score")
    figure.suptitle(
        f"{title} - {SCOPE}\ndashed line: mean episodes to solve (unsolved agents counted at the 5000 cap)",
        fontsize=9,
    )
    return save(figure, name, output_dir, runs[stems_labels[0][0]]["batch_dir"])


def figure_7(runs, output_dir, pqc_stems, nn_stems):
    def best(stems):
        candidates = []
        for stem in stems:
            for row, agent in enumerate(runs[stem]["metrics"]["agents"]):
                if agent["solved"]:
                    candidates.append((agent["solved_at_episode_index"], stem, row))
        return min(candidates)

    figure, axis = plt.subplots(figsize=(6, 4))
    for (episode, stem, row), label, colour in (
        (best(pqc_stems), "best PQC", "tab:green"),
        (best(nn_stems), "best NN", "tab:grey"),
    ):
        scores = runs[stem]["arrays"]["scores"][row]
        scores = scores[np.isfinite(scores)]
        params = runs[stem]["metrics"]["trainable_parameters_per_agent"]
        axis.plot(
            scores,
            color=colour,
            label=f"{label}: {stem} ({params} params), solved at {episode}",
        )
    axis.set(xlabel="Episode", ylabel="Score")
    axis.legend(fontsize=7)
    figure.suptitle(
        f"Fig. 7 best individual agents - {SCOPE}\npaper: PQC 5 layers 206, NN (30,30) 186",
        fontsize=9,
    )
    return save(figure, "fig7_best_agents", output_dir, runs[pqc_stems[1]]["batch_dir"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--batches", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--set", choices=list(SCOPES), required=True)
    args = parser.parse_args()
    global SCOPE, NAME_TAG
    SCOPE = SCOPES[args.set]
    NAME_TAG = "" if args.set == "paper-table" else "_codefaithful"
    runs = {
        paper_table_stem(stem): run for stem, run in load_runs(args.batches).items()
    }
    written = []
    written += figure_3(runs, args.output_dir)
    written += ablation_figure(
        runs,
        args.output_dir,
        "fig4_encoding_ablation",
        "Fig. 4 Cart Pole data encoding",
        [
            ("no_reuploading", "w/o data re-uploading", "tab:blue"),
            ("no_input_weights", "w/o trainable input scaling", "tab:red"),
            ("full", "re-uploading + trainable scaling", "tab:green"),
        ],
    )
    written += ablation_figure(
        runs,
        args.output_dir,
        "fig5_readout_ablation",
        "Fig. 5 Cart Pole output range",
        [
            ("full", "trainable output weights", "tab:green"),
            ("fixed_range_1", "fixed range [0, 1]", "purple"),
            ("fixed_range_90", "fixed range [0, 90]", "gold"),
            ("fixed_range_180", "fixed range [0, 180]", "magenta"),
        ],
    )
    pqc = [
        ("e2e3_cartpole_optimal_full_original", "5 (62)"),
        ("e4_cartpole_pqc_l10_original", "10 (122)"),
        ("e4_cartpole_pqc_l15_original", "15 (182)"),
        ("e4_cartpole_pqc_l20_original", "20 (242)"),
        ("e4_cartpole_pqc_l25_original", "25 (302)"),
        ("e4_cartpole_pqc_l30_original", "30 (362)"),
    ]
    nn = [
        ("e4_cartpole_mlp_10_10_classical", "(10, 10), 182"),
        ("e4_cartpole_mlp_15_15_classical", "(15, 15), 347"),
        ("e4_cartpole_mlp_20_20_classical", "(20, 20), 562"),
        ("e4_cartpole_mlp_24_24_classical", "(24, 24), 770"),
        ("e4_cartpole_mlp_30_30_classical", "(30, 30), 1142"),
        ("e4_cartpole_mlp_64_64_classical", "(64, 64), 4610"),
    ]
    written += scan_figure(
        runs, args.output_dir, "fig6a_pqc_depth", "Fig. 6a PQCs: #layers (#params)", pqc
    )
    written += scan_figure(
        runs, args.output_dir, "fig6b_nn_width", "Fig. 6b NNs: (units), #params", nn
    )
    written += figure_7(runs, args.output_dir, [s for s, _ in pqc], [s for s, _ in nn])
    print("\n".join(written))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
