"""Originality vs broken-rate Pareto-front plot.

Reservoirs differ in the logit scale they produce, which means the same
softmax temperature lands them at different points on the
originality/broken-rate curve. Plotting against ``T`` therefore conflates
"the reservoir" with "the temperature calibration". The reservoir-invariant
view is to draw each model as a parametric curve in
``(originality at L=2, broken_rate of rule "2")`` space, with ``T`` as the
parameter.

Inputs are the ``aggregated.json`` files produced by
``utils.sweep.main`` for any number of sweeps. Markov and uncorrelated
baselines are also drawn (as single points - they have no temperature
parameter). Optionally a per-temperature ``reference`` entry can be added
to overlay the Moth-published Aer sequences.

Usage::

    python utils/plot_pareto.py \
        --aggregated sweeps/modes/aggregated.json sweeps/photons/aggregated.json \
        --reference results/reference_eval_metrics.json \
        --out results/pareto.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PAPER_DIR = Path(__file__).resolve().parents[1]
if str(_PAPER_DIR) not in sys.path:
    sys.path.insert(0, str(_PAPER_DIR))

from utils import aggregate as agg  # noqa: E402


def _temp_sort_key(temp: str) -> float:
    try:
        return float(temp)
    except ValueError:
        return float("inf")


def _curve_for_point(name: str, label: str, per_seed: list[dict]):
    aggregated = agg.aggregate_point(per_seed)
    temps = sorted(aggregated.keys(), key=_temp_sort_key)
    xs, ys, x_err, y_err = [], [], [], []
    for temp in temps:
        orig = aggregated[temp]["originality_L2"]
        broken = aggregated[temp]["broken_rate_2"]
        if orig["n"] == 0 or broken["n"] == 0:
            continue
        xs.append(orig["mean"])
        ys.append(broken["mean"])
        x_err.append(orig["std"])
        y_err.append(broken["std"])
    return label, xs, ys, x_err, y_err, temps[: len(xs)]


def _reference_curve(reference_metrics: dict, max_T: float = 10.0):
    refs = reference_metrics.get("reference", {})
    pairs = []
    for label, stats in refs.items():
        # Labels are "Aer_T=<temp>".
        try:
            temp = float(label.split("=", 1)[1])
        except ValueError:
            continue
        if temp > max_T:
            continue
        row = agg.extract_row(stats)
        pairs.append((temp, row["originality_L2"], row["broken_rate_2"]))
    pairs.sort()
    xs = [p[1] for p in pairs]
    ys = [p[2] for p in pairs]
    return xs, ys, [str(p[0]) for p in pairs]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregated", nargs="+", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--title",
        type=str,
        default="QRC Pareto front: originality vs broken-rate",
    )
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 5.4))

    # Color cycle - reuse default mpl colors.
    cmap = plt.get_cmap("tab10")
    color_idx = 0

    # Collect baseline values from the *first* aggregated file (they should be
    # broadly the same across files because seeds drive baseline RNG).
    first_per_seed: list[dict] = []
    for source in args.aggregated:
        payload = agg.load_aggregated(source)
        for name, info in payload["points"].items():
            label, xs, ys, x_err, y_err, temps = _curve_for_point(
                name,
                info["label"],
                info["per_seed_metrics"],
            )
            if not xs:
                continue
            color = cmap(color_idx % 10)
            ax.errorbar(
                xs,
                ys,
                xerr=x_err,
                yerr=y_err,
                marker="o",
                linestyle="-",
                color=color,
                label=label,
                capsize=3,
                alpha=0.85,
            )
            # Annotate first and last temperature for context.
            if len(temps) >= 1:
                ax.annotate(
                    f"T={temps[0]}",
                    (xs[0], ys[0]),
                    textcoords="offset points",
                    xytext=(4, 4),
                    fontsize=7,
                    color=color,
                )
            if len(temps) >= 2:
                ax.annotate(
                    f"T={temps[-1]}",
                    (xs[-1], ys[-1]),
                    textcoords="offset points",
                    xytext=(4, -10),
                    fontsize=7,
                    color=color,
                )
            color_idx += 1
            if not first_per_seed:
                first_per_seed = info["per_seed_metrics"]

    if first_per_seed:
        baselines = agg.baseline_summary(first_per_seed)
        if "markov" in baselines:
            m = baselines["markov"]
            ax.errorbar(
                [m["originality_L2"]["mean"]],
                [m["broken_rate_2"]["mean"]],
                xerr=[m["originality_L2"]["std"]],
                yerr=[m["broken_rate_2"]["std"]],
                marker="s",
                color="black",
                linestyle="none",
                capsize=3,
                label="Markov",
            )
        if "uncorrelated" in baselines:
            u = baselines["uncorrelated"]
            ax.errorbar(
                [u["originality_L2"]["mean"]],
                [u["broken_rate_2"]["mean"]],
                xerr=[u["originality_L2"]["std"]],
                yerr=[u["broken_rate_2"]["std"]],
                marker="D",
                color="gray",
                linestyle="none",
                capsize=3,
                label="Uncorrelated",
            )

    if args.reference is not None:
        with args.reference.open("r", encoding="utf-8") as handle:
            ref = json.load(handle)
        xs, ys, temps = _reference_curve(ref)
        if xs:
            ax.plot(
                xs,
                ys,
                marker="x",
                linestyle="--",
                color="green",
                label="Moth Aer (published)",
            )
            for x, y, t in zip(xs, ys, temps):
                if t in ("0.1", "1.0", "5.0"):
                    ax.annotate(
                        f"T={t}",
                        (x, y),
                        textcoords="offset points",
                        xytext=(3, 3),
                        fontsize=7,
                        color="green",
                    )

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.0)
    ax.set_xlabel("Originality at L=2 (higher = more novel)")
    ax.set_ylabel("Broken-rate, rule 2 (lower = more playable)")
    ax.set_title(args.title)
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(args.out, dpi=180)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
