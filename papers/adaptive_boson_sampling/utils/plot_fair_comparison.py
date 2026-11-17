"""Plot the Phase 5 fair comparison (paper arm) from ``fair_comparison.json``.

Usage::

    python utils/plot_fair_comparison.py results/fair_comparison.json \
        --out results/fair_comparison.png

One panel per comparison condition. Bars: ABS kernels (Phase 3 reproduction,
blue), MerLin adaptation (purple), information-matched classical kernel
(green), dimension-matched random rank-d kernels (orange) and unmatched raw-feature
references (grey, hatched). Error bars are the variability recorded in the JSON
(std over dataset variants, or SE over partitions for 1D). Stars are
paper-reported accuracies, drawn on the qubit and qutrit bars of digitised
experimental kernels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

COLORS = {
    "ABS reproduction": "#2b6cb0",
    "ABS, MerLin adaptation": "#805ad5",
    "classical, information-matched": "#2f855a",
    "dimension-matched": "#dd6b20",
    "unmatched": "#a0aec0",
}
# Shown per panel when present; the full set is in the JSON and Markdown tables.
SHOWN = [
    "B2_qubit_experimental_fig4a", "B3_qutrit_experimental_fig4b",
    "B2_qubit_fitted_u0", "B3_qutrit_fitted_u0",
    "B2_qubit_fitted_u0_merlin_noisy", "B3_qutrit_fitted_u0_merlin_noisy",
    "gaussian_outcome_index_kernel", "random_rank2_kernel", "random_rank3_kernel",
    "rbf_svm_x", "rbf_svm_raw2d",
]  # fmt: skip
PAPER_KEYS = {"B2_qubit_experimental_fig4a": "qubit_B2",
              "B3_qutrit_experimental_fig4b": "qutrit_B3"}  # fmt: skip


def _color(role: str) -> str:
    for key, color in COLORS.items():
        if role.startswith(key) or key in role:
            return color
    raise ValueError(f"no colour for role {role!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("comparison", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.comparison.read_text())
    conditions = report["conditions"]

    figure, axes = plt.subplots(len(conditions), 1, figsize=(10, 2.9 * len(conditions)))
    for axis, condition in zip(np.atleast_1d(axes), conditions):
        models = condition["arms"]["paper"]["models"]
        labels = [label for label in SHOWN if label in models]
        positions = np.arange(len(labels))
        for position, label in zip(positions, labels):
            model = models[label]
            axis.bar(position, model["mean_test_accuracy"], yerr=model["variability"],
                     color=_color(model["role"]), capsize=3,
                     hatch="//" if "unmatched" in model["role"] else None)  # fmt: skip
            if label in PAPER_KEYS:
                axis.plot(position, condition["paper"][PAPER_KEYS[label]], "k*", ms=11)
        axis.set_xticks(positions, labels, rotation=25, ha="right", fontsize=7)
        axis.set_ylim(0, 1.05)
        axis.axhline(0.5, color="k", lw=0.6, ls=":")
        axis.set_ylabel("test accuracy")
        n_variants = models[labels[0]]["n_variants"]
        axis.set_title(f"{condition['name']} (paper arm, C = 50; {n_variants} dataset "
                       "variant(s); substitute-data, simulation)", fontsize=9)  # fmt: skip
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in COLORS.values()]
    handles.append(plt.Line2D([], [], marker="*", color="k", ls="", ms=10))
    figure.legend(handles, [*COLORS, "paper-reported"], loc="upper center", ncol=6,
                  fontsize=8)  # fmt: skip
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(args.out, dpi=140)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
