"""Plot E1 accuracy versus client count and per-round convergence.

``python utils/plot_results.py <sweeps_dir> <results_dir>`` reads the structured
``metrics.json`` files of the finished sweeps (no run directory is hard-coded) and saves
two figures into ``results_dir`` and into the primary sweep directory
(``e1_vqtn_hybrid``). Every figure carries its result-scope label.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_with_paper import PAPER_ACCURACY, load_runs  # noqa: E402

CLIENTS = (1, 2, 4, 8, 16)
SETTING_NAMES = ("centralized", "clients2", "clients4", "clients8", "clients16")
PAPER_KEYS = ("N/A", "2", "4", "8", "16")
SCOPE_LABEL = "5 seeds, full data; inferred circuit, hybrid read-out (deviation); not paper-accurate"


def _stats(runs: list[dict]) -> tuple[float, float]:
    values = [100 * run["history"][-1]["test_accuracy"] for run in runs]
    return statistics.fmean(values), statistics.stdev(values)


def plot_accuracy_vs_clients(sweeps: Path, outputs: list[Path]) -> None:
    hybrid = load_runs(sweeps / "e1_vqtn_hybrid")
    classical = load_runs(sweeps / "control_classical_e1")
    faithful = load_runs(sweeps / "sensitivity_faithful_0v9")
    figure, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)
    positions = range(len(CLIENTS))
    for axis, task in zip(axes, PAPER_ACCURACY):
        for label, runs, colour, marker in (
            ("hybrid VQTN (ours)", hybrid, "tab:blue", "o"),
            ("classical MLP (ours)", classical, "tab:green", "s"),
        ):
            stats = [_stats(runs[f"task{task}_{name}"]) for name in SETTING_NAMES]
            axis.errorbar(
                positions,
                [m for m, _ in stats],
                [s for _, s in stats],
                label=label,
                color=colour,
                marker=marker,
                capsize=3,
            )
        if task == "0v9":
            stats = [_stats(faithful[f"task{task}_{name}"]) for name in SETTING_NAMES]
            axis.errorbar(
                positions,
                [m for m, _ in stats],
                [s for _, s in stats],
                label="faithful VQTN (ours)",
                color="tab:red",
                marker="^",
                capsize=3,
            )
        paper = PAPER_ACCURACY[task]
        axis.errorbar(
            positions,
            [paper[key][0] for key in PAPER_KEYS],
            [paper[key][1] for key in PAPER_KEYS],
            label="paper (Table II)",
            color="black",
            linestyle="--",
            marker="x",
            capsize=3,
        )
        axis.set_xticks(list(positions), ["non-fed", "2", "4", "8", "16"])
        axis.set_xlabel("clients")
        axis.set_title(f"MNIST {task[0]} vs {task[2]}")
        axis.grid(alpha=0.3)
    axes[0].set_ylabel("test accuracy (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=4, fontsize=8)
    figure.suptitle(f"E1: accuracy vs number of clients ({SCOPE_LABEL})", fontsize=9)
    figure.tight_layout(rect=(0, 0.07, 1, 1))
    for path in outputs:
        path.mkdir(parents=True, exist_ok=True)
        figure.savefig(path / "e1_accuracy_vs_clients.png", dpi=150)
    plt.close(figure)


def plot_convergence(sweeps: Path, outputs: list[Path]) -> None:
    hybrid = load_runs(sweeps / "e1_vqtn_hybrid")
    classical = load_runs(sweeps / "control_classical_e1")
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for axis, (title, runs) in zip(
        axes, (("hybrid VQTN (ours)", hybrid), ("classical MLP (ours)", classical))
    ):
        for name, label in zip(
            SETTING_NAMES, ("non-federated", "2 clients", "4", "8", "16")
        ):
            histories = [
                [entry["test_accuracy"] * 100 for entry in run["history"]]
                for run in runs[f"task0v9_{name}"]
            ]
            mean = [statistics.fmean(column) for column in zip(*histories)]
            axis.plot(range(1, len(mean) + 1), mean, label=label)
        axis.set_title(title)
        axis.set_xlabel("round (epoch for non-federated)")
        axis.grid(alpha=0.3)
    axes[0].set_ylabel("mean test accuracy over 5 seeds (%)")
    axes[0].legend(fontsize=8, loc="lower right")
    figure.suptitle(
        f"E1 task 0 vs 9: convergence per round ({SCOPE_LABEL})", fontsize=9
    )
    figure.tight_layout()
    for path in outputs:
        path.mkdir(parents=True, exist_ok=True)
        figure.savefig(path / "e1_convergence_0v9.png", dpi=150)
    plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sweeps_dir", type=Path)
    parser.add_argument("results_dir", type=Path)
    args = parser.parse_args(argv)
    outputs = [args.results_dir, args.sweeps_dir / "e1_vqtn_hybrid"]
    plot_accuracy_vs_clients(args.sweeps_dir, outputs)
    plot_convergence(args.sweeps_dir, outputs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
