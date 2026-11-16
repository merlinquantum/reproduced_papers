"""Protocol check (App. A): per-shot Aer counts against the swap-test law.

For each random state pair, the fraction of ancilla zeros measured on Aer
(1,024 shots) is plotted against the probability ``1/2 + 1/2 |<a|b>|^2`` that
the binomial shortcut uses. The band is +-2 binomial standard deviations.

Usage: ``.venv/bin/python papers/distributed_secure_kernels/utils/plot_protocol_check.py [<run dir> ...]``
(default: the latest completed run of each protocol config).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from distributed_secure_kernels.utils.plot_style import (  # noqa: E402
    INK,
    MUTED,
    OURS,
    REFERENCE,
    save,
    style_axes,
)
from distributed_secure_kernels.utils.pooled_protocol_check import (  # noqa: E402
    PROJECT_DIR,
    latest_runs,
)

SHOTS = 1024


def main() -> None:
    run_dirs = [Path(arg) for arg in sys.argv[1:]] or latest_runs(
        PROJECT_DIR / "outdir"
    )
    ideal, distributed, centralised = [], [], []
    for run_dir in run_dirs:
        noiseless = json.loads((run_dir / "metrics.json").read_text())["noiseless"]
        ideal += noiseless["p0_ideal"]
        distributed += noiseless["distributed_zero_fraction"]
        centralised += noiseless["centralised_zero_fraction"]

    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    style_axes(ax)
    grid = np.linspace(0.5, 1.0, 200)
    band = 2 * np.sqrt(grid * (1 - grid) / SHOTS)
    ax.fill_between(grid, grid - band, grid + band, color=REFERENCE, alpha=0.15, lw=0)
    ax.plot(grid, grid, color=REFERENCE, lw=1.5, label="binomial law (±2 std)")
    ax.scatter(
        ideal,
        distributed,
        s=40,
        color=OURS,
        edgecolor="#fcfcfb",
        linewidth=1,
        zorder=3,
        label="distributed circuit",
    )
    ax.scatter(
        ideal,
        centralised,
        s=40,
        marker="s",
        facecolor="none",
        edgecolor=INK,
        linewidth=1.2,
        zorder=3,
        label="centralised circuit",
    )
    ax.set_xlim(0.48, 0.92)
    ax.set_ylim(0.44, 0.94)
    ax.set_xlabel("Predicted probability of reading 0:  ½ + ½ |⟨a|b⟩|²", color=INK)
    ax.set_ylabel("Fraction of 0s on Aer (1,024 shots)", color=INK)
    ax.set_title(
        "Per-shot Aer runs of the authors' circuits follow the swap-test law",
        color=INK,
        fontsize=11,
    )
    ax.legend(frameon=False, labelcolor=INK, loc="upper left")
    ax.text(
        0.99,
        0.01,
        f"{len(ideal)} random state pairs (2 and 4 qubits per party)",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color=MUTED,
    )
    for path in save(fig, "protocol_check", run_dirs[-1]):
        print(path)


if __name__ == "__main__":
    main()
