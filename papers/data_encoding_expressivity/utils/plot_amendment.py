"""Plot the Phase 4 amendment: improved photonic models, the qubit fairness control, and the universality test.

Usage, from the paper directory::

    python utils/plot_amendment.py \
        --photonic P0=outdir/<run> --photonic P1=outdir/<run> ... \
        --qubit P0=outdir/<run> ... \
        --universality P0=outdir/<run> --universality P3=outdir/<run>

``merlin_amendment_protocols.png``: final full-grid MSE per seed for every
case under every protocol, with the target's B1 floor (about 0).
``merlin_universality.png``: final MSE per seed against the model degree K for
the single-photon theorem setting on the square wave, with the degree-K B1
floor. Both are saved to the first photonic run directory and ``results/``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from _figures import load_completed_metrics, save_to_run_and_results, stamp_scope

ROUND_OFF_FLOOR = 1e-16
PROTOCOL_COLORS = {
    "P0": "tab:gray",
    "P1": "tab:blue",
    "P2": "tab:orange",
    "P3": "tab:green",
}


def _labelled(items: list[str]) -> list[tuple[str, Path]]:
    pairs = []
    for item in items:
        label, separator, path = item.partition("=")
        if not separator:
            raise ValueError(f"expected '<protocol>=<run directory>', got {item!r}")
        pairs.append((label, Path(path)))
    return pairs


def plot_protocols(
    photonic: list[tuple[str, Path]], qubit: list[tuple[str, Path]]
) -> None:
    case_names: list[str] = []
    values: dict[tuple[str, str], list[float]] = {}
    for protocol, run_dir in [*photonic, *qubit]:
        metrics = load_completed_metrics(run_dir, "fit")
        for case in metrics["cases"]:
            if case["name"] not in case_names:
                case_names.append(case["name"])
            values[(case["name"], protocol)] = [
                max(run["final_mse"], ROUND_OFF_FLOOR)
                for run in metrics["runs"]
                if run["case"] == case["name"]
            ]
    protocols = list(dict.fromkeys(protocol for protocol, _ in photonic))
    figure, axis = plt.subplots(figsize=(11, 5.5))
    for row, name in enumerate(case_names):
        for column, protocol in enumerate(protocols):
            seeds = values.get((name, protocol))
            if seeds is None:
                continue
            y = np.full(len(seeds), row + (column - 1.5) * 0.18)
            axis.scatter(
                seeds,
                y,
                s=12,
                color=PROTOCOL_COLORS[protocol],
                label=protocol if row == 0 else None,
            )
    axis.axvline(1e-3, color="0.5", ls=":", lw=0.8)
    axis.axvspan(ROUND_OFF_FLOOR / 10, ROUND_OFF_FLOOR * 3, color="0.92", zorder=0)
    axis.set_xscale("log")
    axis.set_xlim(ROUND_OFF_FLOOR / 10, 1)
    axis.set_yticks(range(len(case_names)))
    axis.set_yticklabels(case_names, fontsize=7)
    axis.invert_yaxis()
    axis.set_xlabel(
        "final full-grid MSE per seed (B1 floor ≈ 0; grey band = round-off; dotted = 1e-3)"
    )
    axis.set_title(
        "Fig. 4 target: photonic (MerLin) and qubit models under protocols P0 (paper) to P3",
        fontsize=9,
    )
    axis.legend(fontsize=7, title="protocol", title_fontsize=7, loc="lower left")
    stamp_scope(
        figure,
        "MerLin adaptation + qubit control; P0 = paper protocol, P1-P3 = non-paper protocols; 6 seeds; simulation",
        ", ".join(str(path) for _, path in [*photonic, *qubit]),
    )
    figure.tight_layout(rect=(0, 0.03, 1, 1))
    save_to_run_and_results(figure, photonic[0][1], "merlin_amendment_protocols.png")


def plot_universality(universality: list[tuple[str, Path]], save_dir: Path) -> None:
    figure, axis = plt.subplots(figsize=(6, 4))
    floors: dict[int, float] = {}
    for index, (protocol, run_dir) in enumerate(universality):
        metrics = load_completed_metrics(run_dir, "fit")
        for case in metrics["cases"]:
            degree = max(case["b1_frequencies"])
            floors[degree] = case["b1_mse"]
            finals = [
                run["final_mse"]
                for run in metrics["runs"]
                if run["case"] == case["name"]
            ]
            axis.scatter(
                np.full(len(finals), degree + (index - 0.5) * 0.15),
                finals,
                s=14,
                color=PROTOCOL_COLORS[protocol],
                label=protocol if degree == 1 else None,
            )
    degrees = sorted(floors)
    axis.plot(
        degrees,
        [floors[d] for d in degrees],
        "k--",
        marker="_",
        label="B1 floor (best degree-K fit)",
    )
    axis.set_yscale("log")
    axis.set_xticks(degrees)
    axis.set_xlabel("degree K = m - 1 (one photon in m modes)")
    axis.set_ylabel("final full-grid MSE to the square wave")
    axis.set_title(
        "Photonic universality test: trained model vs degree-K floor", fontsize=9
    )
    axis.legend(fontsize=7)
    stamp_scope(
        figure,
        "theorem test, single photon (classically simulable); 6 seeds; simulation",
        ", ".join(str(path) for _, path in universality),
    )
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    save_to_run_and_results(figure, save_dir, "merlin_universality.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--photonic", action="append", required=True)
    parser.add_argument("--qubit", action="append", required=True)
    parser.add_argument("--universality", action="append", required=True)
    args = parser.parse_args()
    photonic = _labelled(args.photonic)
    plot_protocols(photonic, _labelled(args.qubit))
    universality = _labelled(args.universality)
    plot_universality(universality, universality[0][1])


if __name__ == "__main__":
    main()
