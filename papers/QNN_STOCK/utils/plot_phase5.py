"""Plot the primary Phase 5 comparisons from ``pairs.csv`` and ``claims.csv``.

One panel per claim: paired-by-seed difference (candidate minus reference)
with its 95 % t interval, one row per comparison and asset. Filled markers
are commensurable pairs, hollow markers partially commensurable ones.

Usage::

    python papers/QNN_STOCK/utils/plot_phase5.py --results papers/QNN_STOCK/results/phase5
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

METRIC_LABELS = {
    "inference/test/log_loss": "test log-loss difference",
    "final_fidelity": "final fidelity difference",
    "output/train/kl_joint": "train joint KL difference",
}


def _read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args(argv)

    pairs = [p for p in _read(args.results / "pairs.csv") if p["role"] == "primary"]
    outcomes = {c["claim"]: c["outcome"] for c in _read(args.results / "claims.csv")}
    claims = list(dict.fromkeys(p["claim"] for p in pairs))
    heights = [max(2, sum(p["claim"] == c for p in pairs)) for c in claims]

    fig, axes = plt.subplots(
        len(claims),
        1,
        figsize=(9, 0.32 * sum(heights) + 1.2 * len(claims)),
        gridspec_kw={"height_ratios": heights},
    )
    for ax, claim in zip(axes, claims):
        rows = [p for p in pairs if p["claim"] == claim]
        for y, row in enumerate(reversed(rows)):
            mean, low, high = (
                float(row[k]) for k in ("mean_difference", "ci95_low", "ci95_high")
            )
            filled = row["status"] == "COMMENSURABLE"
            ax.plot([low, high], [y, y], color="0.35", lw=1.4)
            ax.plot(mean, y, "o", ms=5, mfc="0.15" if filled else "white", mec="0.15")
        ax.axvline(0, color="0.6", lw=0.8, ls="--")
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(
            [f"{r['label']} [{r['asset']}]" for r in reversed(rows)], fontsize=7
        )
        metric = rows[0]["metric"]
        ax.set_xlabel(
            f"{METRIC_LABELS.get(metric, metric)} (candidate minus reference)",
            fontsize=8,
        )
        ax.set_title(f"{claim}: {outcomes.get(claim, '')}", fontsize=9, loc="left")
        ax.tick_params(axis="x", labelsize=7)
    fig.suptitle(
        "Phase 5 primary comparisons, substitute-data, simulated, paired 95 % t interval over seeds\n"
        "filled = commensurable, hollow = partially commensurable",
        fontsize=9,
    )
    # Keep 0.6 inch at the top for the two-line suptitle.
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.6 / fig.get_figheight()))
    out = args.results / "fig_phase5_primary.png"
    fig.savefig(out, dpi=150)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
