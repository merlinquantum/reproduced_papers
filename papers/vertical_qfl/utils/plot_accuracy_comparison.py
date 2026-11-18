"""Plot test accuracy of the four VQFL experiments (analogue of the paper's Figs. 2-3).

Reads ``metrics.json`` of the runs listed in each validated sweep's
``sweep_status.json``; no run directory is hard-coded. Saves one PNG to
``results/`` and a copy next to each sweep.

Example
-------
    python utils/plot_accuracy_comparison.py --sweep E1=outdir/sweep_e1_... --sweep E2=...
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parents[1]
LABELS = {
    "E1": "E1: 8 clients, all pixels",
    "E2": "E2: 1 client",
    "E3": "E3: 1 slice x 8 clients",
    "E4": "E4: 4 slices x 2 (half the pixels)",
}
LABELS.update(
    {
        "E1p": "E1p: photonic, 8 clients",
        "E2p": "E2p: photonic, 1 client",
        "E3p": "E3p: photonic, 1 slice x 8",
        "E4p": "E4p: photonic, half the pixels",
    }
)
COLORS = {"E1": "#1b6ca8", "E2": "#b5651d", "E3": "#6b8e23", "E4": "#8e44ad"}
for _k in ("1", "2", "3", "4"):
    LABELS[f"E{_k}h"] = LABELS[f"E{_k}p"].replace("photonic", "photonic v2 head")
    LABELS[f"E{_k}f"] = LABELS[f"E{_k}p"].replace("photonic", "photonic v2 no head")
COLORS.update(
    {"E1p": COLORS["E1"], "E2p": COLORS["E2"], "E3p": COLORS["E3"], "E4p": COLORS["E4"]}
)
for _k in ("1", "2", "3", "4"):
    COLORS[f"E{_k}h"] = COLORS[f"E{_k}"]
    COLORS[f"E{_k}f"] = COLORS[f"E{_k}"]
# Line style by model family: gate model solid, first photonic design dash-dot,
# photonic v2 with head dashed, photonic v2 without head dotted.
LINESTYLE = {"": "-", "p": "-.", "h": "--", "f": ":"}
SCOPE = "paper-accurate hyperparameters, 5 feature distributions, 1 training seed each"


def load_curves(sweep_dir: Path) -> np.ndarray:
    status = json.loads((sweep_dir / "sweep_status.json").read_text())
    if status["status"] != "COMPLETED":
        raise ValueError(f"{sweep_dir} is {status['status']}")
    curves = []
    for run in status["runs"]:
        metrics = json.loads((sweep_dir / run["metrics_path"]).read_text())
        curves.append(metrics["history"]["test_accuracy"])
    return np.array(curves)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--sweep", action="append", required=True, metavar="LABEL=DIR")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_DIR / "results")
    parser.add_argument(
        "--name", default="accuracy_comparison", help="Output file stem"
    )
    parser.add_argument(
        "--title", default=None, help="Figure title (default: gate-model scope label)"
    )
    args = parser.parse_args()

    sweep_dirs = {}
    curves = {}
    for item in args.sweep:
        label, directory = item.split("=", 1)
        sweep_dirs[label] = Path(directory)
        curves[label] = load_curves(sweep_dirs[label])

    fig, (left, right) = plt.subplots(
        1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.5, 1]}
    )
    for label, data in curves.items():
        epochs = np.arange(1, data.shape[1] + 1)
        mean, std = data.mean(axis=0), data.std(axis=0, ddof=1)
        left.plot(
            epochs,
            100 * mean,
            color=COLORS[label],
            ls=LINESTYLE[label[2:]],
            label=LABELS[label],
        )
        left.fill_between(
            epochs,
            100 * (mean - std),
            100 * (mean + std),
            color=COLORS[label],
            alpha=0.18,
        )
    if "E1" in curves:
        left.axhline(94, color="grey", ls="--", lw=1)
        left.text(1, 94.4, "paper text: E1 over 94%", color="grey", fontsize=8)
    left.set(
        xlabel="epoch",
        ylabel="test accuracy (%)",
        title="Test accuracy per epoch (mean +- std over 5 distributions)",
    )
    left.legend(loc="lower right", fontsize=8)
    left.grid(alpha=0.25)

    positions = np.arange(len(curves))
    for position, (label, data) in zip(positions, curves.items()):
        finals = 100 * data[:, -1]
        right.scatter(
            np.full(len(finals), position),
            finals,
            color=COLORS[label],
            alpha=0.7,
            zorder=3,
        )
        right.errorbar(
            position,
            finals.mean(),
            yerr=finals.std(ddof=1),
            color="black",
            capsize=5,
            zorder=4,
        )
    if "E1" in curves:
        right.axhline(94, color="grey", ls="--", lw=1)
        right.axhline(90, color="grey", ls=":", lw=1)
        right.text(
            len(curves) - 0.55, 94.3, "paper E1", color="grey", fontsize=8, ha="right"
        )
        right.text(
            len(curves) - 0.55, 90.3, "paper E4", color="grey", fontsize=8, ha="right"
        )
    right.set_xticks(
        positions,
        list(curves),
    )
    right.set(
        ylabel="final-epoch test accuracy (%)",
        title="Final epoch, each point is one distribution",
    )
    right.grid(alpha=0.25, axis="y")
    fig.suptitle(args.title or f"Vertical QFL on Semeion, {SCOPE}", fontsize=10)
    fig.tight_layout()

    args.out_dir.mkdir(exist_ok=True)
    target = args.out_dir / f"{args.name}.png"
    fig.savefig(target, dpi=150)
    for directory in sweep_dirs.values():
        shutil.copy(target, directory / f"{args.name}.png")
    print(f"Saved {target}")


if __name__ == "__main__":
    main()
