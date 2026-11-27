"""Plot Phase 5 extensions X1 and X2 from the generated extension JSON.

Usage, from the paper directory::

    python utils/plot_extensions.py --extensions results/phase5_extensions.json

Left panel (X1): final MSE per seed for the degree >= 5 systems under L-BFGS,
against the B1 floor. Right panel (X2): parameters needed per degree on the
square wave, for every system that reached the B1 floor in at least half its
seeds, labelled with photons and modes. Saves ``phase5_extensions.png`` next
to the input JSON.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROUND_OFF_FLOOR = 1e-16
COLORS = {"classical": "tab:green", "qubit": "tab:blue", "photonic": "tab:purple"}


def _short(row: dict) -> str:
    if row["modality"] == "classical":
        return f"B2 classical (D = {row['degree']})"
    return f"{row['case']} [{row['readout']}]"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--extensions", required=True, type=Path)
    args = parser.parse_args()
    rows = json.loads(args.extensions.read_text(encoding="utf-8"))["rows"]
    figure, (left, right) = plt.subplots(
        1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.3, 1]}
    )

    x1 = [r for r in rows if r["group"] == "X1" and r["degree"] >= 5]
    x1.sort(key=lambda r: (r["modality"], r["case"]))
    for index, row in enumerate(x1):
        run_metrics = json.loads(
            (Path(row["run"]) / "metrics.json").read_text(encoding="utf-8")
        )
        finals = [
            max(r["final_mse"], ROUND_OFF_FLOOR)
            for r in run_metrics["runs"]
            if r["case"] == row["case"]
        ]
        left.scatter(
            finals,
            np.full(len(finals), index),
            s=12,
            color=COLORS[row["modality"]],
            alpha=0.7,
        )
    left.axvline(1e-4, color="0.5", ls=":", lw=0.8)
    left.set_xscale("log")
    left.set_xlim(ROUND_OFF_FLOOR / 10, 1)
    left.set_yticks(range(len(x1)))
    left.set_yticklabels([_short(r) for r in x1], fontsize=7)
    left.set_xlabel("final MSE per seed (B1 ≈ 0; dotted = success threshold)")
    left.set_title("X1: fixed-readout ceiling under L-BFGS, 20 seeds", fontsize=9)

    x2 = [r for r in rows if r["group"] == "X2" and r["protocol"] == "P5"]
    for row in x2:
        if row["success_count"] * 2 < row["successful_runs"]:
            continue
        marker = (
            "s"
            if row["modality"] == "classical"
            else ("o" if "single_photon" in row["case"] else "^")
        )
        right.scatter(
            row["degree"],
            row["n_parameters"],
            color=COLORS[row["modality"]],
            marker=marker,
            s=40,
        )
        if row["modality"] == "photonic":
            run_metrics = json.loads(
                (Path(row["run"]) / "metrics.json").read_text(encoding="utf-8")
            )
            model = next(
                c["model"] for c in run_metrics["cases"] if c["name"] == row["case"]
            )
            right.annotate(
                f"n={model['n_photons']}, m={model['n_modes']}",
                (row["degree"], row["n_parameters"]),
                fontsize=6,
                xytext=(4, 2),
                textcoords="offset points",
            )
    right.set_yscale("log")
    right.set_xlabel("degree D reached (B1 floor reached in ≥ half the seeds)")
    right.set_ylabel("trainable parameters")
    right.set_title(
        "X2: square wave, L-BFGS. ○ single photon, △ multi-photon, □ B2", fontsize=9
    )
    figure.text(
        0.01,
        0.005,
        "[Phase 5 extensions, advantage-evidence label (B2 under identical conditions); non-paper protocols; "
        "photonic = MerLin adaptation, noiseless simulation; source: results/phase5_extensions.json]",
        fontsize=6,
        color="0.35",
    )
    figure.tight_layout(rect=(0, 0.03, 1, 1))
    output = args.extensions.with_name("phase5_extensions.png")
    figure.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"saved {output}")


if __name__ == "__main__":
    main()
