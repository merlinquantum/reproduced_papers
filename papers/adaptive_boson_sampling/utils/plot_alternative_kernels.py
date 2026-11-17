"""Histogram 1D accuracies over alternative outcome assignments (Fig. 5b-c).

Usage::

    python utils/plot_alternative_kernels.py outdir/run_A outdir/run_B ...

Each instance of a model with a random assignment contributes its mean test
accuracy over the partitions (paper arm only, as in the paper). The paper reports
accuracies spread over about 0.73-1.0 for both encodings; that band is shaded.
Saved to every run directory and ``results/<--name>.png`` (default
``alternative_kernels_1d``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
PAPER_RANGE = (0.73, 1.0)


def instance_accuracies(run_dir: Path) -> dict[str, list[float]]:
    """Return per-model lists of instance mean test accuracies (paper arm)."""
    metrics = json.loads((run_dir / "metrics.json").read_text())
    return {
        label: [
            instance["mean_test_accuracy"]
            for instance in result["arms"]["paper"]["instances"]
        ]
        for label, result in metrics["results"].items()
        if "paper" in result["arms"]
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--name", default="alternative_kernels_1d")
    args = parser.parse_args()

    accuracies: dict[str, list[float]] = {}
    for run_dir in args.run_dirs:
        accuracies.update(instance_accuracies(run_dir))
    accuracies = {k: v for k, v in accuracies.items() if len(v) > 1}

    count = len(accuracies)
    columns = min(count, 3)
    rows = int(np.ceil(count / columns))
    figure, axes = plt.subplots(
        rows, columns, figsize=(4 * columns, 3 * rows), squeeze=False
    )
    bins = np.linspace(0.3, 1.0, 15)
    for axis, (label, values) in zip(axes.flat, accuracies.items()):
        axis.axvspan(*PAPER_RANGE, color="gold", alpha=0.25, label="paper range")
        axis.hist(values, bins=bins, color="tab:blue", edgecolor="white")
        axis.axvline(np.mean(values), color="k", ls="--", lw=1, label="mean")
        axis.set_title(f"{label} (n={len(values)})", fontsize=8)
        axis.set_xlabel("mean test accuracy")
        axis.set_ylabel("kernels")
    for axis in list(axes.flat)[count:]:
        axis.set_visible(False)
    axes.flat[0].legend(fontsize=7)
    figure.suptitle("1D task, alternative assignments - substitute-data", fontsize=9)
    figure.tight_layout()

    RESULTS_DIR.mkdir(exist_ok=True)
    name = f"{args.name}.png"
    for target in [run / name for run in args.run_dirs] + [RESULTS_DIR / name]:
        figure.savefig(target, dpi=120, bbox_inches="tight")
    plt.close(figure)
    print(RESULTS_DIR / name)


if __name__ == "__main__":
    main()
