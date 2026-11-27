"""Phase 04 design-study figure: normalized effective dimension at n=1e5 for every photonic
design (hard and easy encodings), against the Phase 03 qubit and classical references.

Usage (from papers/power_of_QNN):
    python utils/plot_design_study.py --screening outdir/run_A --tier2 outdir/run_B \
        --confirm outdir/run_C outdir/run_D --phase3 outdir/run_E --scope paper-accurate

Exploratory screening bars use draws 0-2 and are hatched; confirmation runs (fresh draws
5-9) are drawn solid. Writes results/merlin_design_study.png.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from figure_io import RESULTS_DIR, load_metrics  # noqa: E402
from power_of_QNN.lib.plotting import _stamp  # noqa: E402

N_INDEX = 6  # n = 1e5 in the paper grid
# Design codes from LOG.md Amendments 2-3 (searches 1 and 2), spelled out for the figure.
LABELS = {
    "P0": "6-mode, encode then mesh (planned)",
    "P1": "6-mode, encode then mesh, dead phases removed",
    "P2": "10-mode, encode then mesh",
    "P3": "6-mode, interleaved uploads",
    "P4": "8-mode, interleaved uploads",
    "P5": "6-mode, interleaved, cluster input",
    "P6": "6-mode, interleaved, GHZ input",
    "P7": "6-mode, interleaved, photon-number readout",
    "T2a_P4cluster": "8-mode, interleaved, cluster input",
    "T2b_qloq4": "qubit QNN as 1 photon in 16 modes (QLOQ)",
    "T2c_feedforward": "feed-forward",
}


def _value(entry: dict) -> tuple[float, float]:
    return entry["normalized_effective_dimension_mean"][N_INDEX], entry[
        "normalized_effective_dimension_std"
    ][N_INDEX]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--screening", type=Path, required=True)
    parser.add_argument("--tier2", type=Path, required=True)
    parser.add_argument("--confirm", type=Path, nargs="+", required=True)
    parser.add_argument("--phase3", type=Path, required=True)
    parser.add_argument("--scope", required=True)
    args = parser.parse_args()
    exploratory = {
        **load_metrics(args.screening)["models"],
        **load_metrics(args.tier2)["models"],
    }
    confirmed = {}
    for run in args.confirm:
        confirmed.update(load_metrics(run)["models"])
    phase3 = load_metrics(args.phase3)["models"]

    figure, axis = plt.subplots(figsize=(10, 4.5))
    positions = np.arange(len(LABELS))
    for offset, encoding, color in (
        (-0.2, "hard", "tab:blue"),
        (0.2, "easy", "tab:orange"),
    ):
        for position, key in zip(positions, LABELS):
            name = f"{key}_{encoding}"
            if name in confirmed:
                mean, std = _value(confirmed[name])
                axis.bar(
                    position + offset,
                    mean,
                    0.38,
                    yerr=std,
                    color=color,
                    capsize=2,
                    label=f"{encoding} encoding (confirmed, draws 5-9)"
                    if key == "P4"
                    else None,
                )
            elif name in exploratory:
                mean, std = _value(exploratory[name])
                axis.bar(
                    position + offset,
                    mean,
                    0.38,
                    yerr=std,
                    color=color,
                    alpha=0.45,
                    hatch="//",
                    capsize=2,
                    label=f"{encoding} encoding (screening, draws 0-2)"
                    if key == "P0"
                    else None,
                )
    for name, style, label in (
        ("qnn", "-", "qubit QNN (paper)"),
        ("easy_qnn", "--", "easy QNN (paper)"),
        ("classical_4442", ":", "classical [4,4,4,2] (paper)"),
    ):
        axis.axhline(
            _value(phase3[name])[0],
            color="black",
            linestyle=style,
            linewidth=1,
            label=label,
        )
    axis.set_xticks(
        positions, list(LABELS.values()), rotation=35, ha="right", fontsize=8
    )
    axis.set_ylabel("normalised effective dimension at n = 1e5 (d = 40)")
    axis.set_ylim(0.3, 1.0)
    axis.legend(fontsize=7, ncol=2, loc="upper left")
    figure.tight_layout()
    _stamp(figure, args.scope)
    RESULTS_DIR.mkdir(exist_ok=True)
    figure.savefig(RESULTS_DIR / "merlin_design_study.png", dpi=150)
    plt.close(figure)


if __name__ == "__main__":
    main()
