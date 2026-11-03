"""Unnormalized Fisher trace per parameter vs input size (Sec. 3.4 barren-plateau diagnostic).

The paper's Figs. 2 and 9 show F_hat, normalized so that mean tr F_hat = d, which removes
the overall scale. Sec. 3.4 ties barren plateaus to tr E[F] <= d (omega_S + omega_S^2), so
this figure plots mean tr F / d (before normalization) per draw against s_in = S.

Usage: python utils/plot_fisher_trace_scaling.py --run-dirs outdir/run_A outdir/run_B --scope paper-accurate --name <stem>
Model families are matched by name prefix (``qnn``, ``easy_qnn``, ``classical``).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from figure_io import load_metrics, outputs  # noqa: E402
from power_of_QNN.lib.plotting import _stamp  # noqa: E402

FAMILIES = {
    "qnn": "qubit QNN (paper)",
    "easy_qnn": "easy QNN (paper)",
    "classical": "classical (paper's networks)",
}


def family(name: str) -> str:
    for prefix in ("easy_qnn", "qnn", "classical"):
        if name.startswith(prefix):
            return prefix
    raise ValueError(f"Cannot assign model {name!r} to a family")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--scope", required=True)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    points: dict[str, list[tuple[int, np.ndarray]]] = {key: [] for key in FAMILIES}
    for run_dir in args.run_dirs:
        for name, entry in load_metrics(run_dir)["models"].items():
            traces = (
                np.asarray([draw["mean_trace"] for draw in entry["draws"]]) / entry["d"]
            )
            points[family(name)].append((entry["input_size"], traces))
    figure, axis = plt.subplots(figsize=(6, 4))
    for key, label in FAMILIES.items():
        series = sorted(points[key], key=lambda item: item[0])
        sizes = np.asarray([size for size, _ in series])
        means = np.asarray([traces.mean() for _, traces in series])
        stds = np.asarray([traces.std() for _, traces in series])
        slope = np.polyfit(sizes, np.log(means), 1)[0]
        axis.errorbar(
            sizes,
            means,
            yerr=stds,
            marker="o",
            capsize=3,
            label=f"{label}: x{np.exp(slope):.2f} per unit s_in",
        )
    axis.set_yscale("log")
    axis.set_xlabel("input size s_in (= qubits S for the quantum models)")
    axis.set_ylabel("mean tr F / d (unnormalized)")
    axis.legend(fontsize=8)
    figure.tight_layout()
    _stamp(figure, args.scope)
    for path in outputs(args.run_dirs[-1], args.name):
        figure.savefig(path, dpi=150)
    plt.close(figure)


if __name__ == "__main__":
    main()
