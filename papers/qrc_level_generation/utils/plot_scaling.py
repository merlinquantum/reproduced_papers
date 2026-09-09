"""Scaling sweep plots: metric vs reservoir output dimension.

Takes one or more ``aggregated.json`` files and plots originality(L=2) and
broken-rate (rule "2") at a chosen temperature, vs the reservoir's output
dimension. Each aggregated file gets its own series so qubit and photonic
points can be overlaid on the same axes.

The output dimension is inferred from the first per-seed metrics block's
``training_history`` shape? - No, more robustly from
``baselines.markov.originality`` length context. The cleanest source is the
``config_excerpt``: gate-based uses ``2**n_qubits``; photonic uses
``C(n_modes, n_photons)``. We compute it here from the config snapshot.
"""

from __future__ import annotations

import argparse
import sys
from math import comb
from pathlib import Path

_PAPER_DIR = Path(__file__).resolve().parents[1]
if str(_PAPER_DIR) not in sys.path:
    sys.path.insert(0, str(_PAPER_DIR))

from utils import aggregate as agg  # noqa: E402


def _output_dim(metrics: dict) -> int | None:
    cfg = metrics.get("config_excerpt", {}).get("qrc", {})
    backend = cfg.get("backend")
    if backend == "qubit":
        n = int(cfg.get("n_qubits", 0))
        return 2**n if n else None
    if backend == "photonic":
        n_modes = int(cfg.get("n_modes", 0))
        n_photons = int(cfg.get("n_photons", 0))
        if n_modes <= 0 or n_photons <= 0:
            return None
        return comb(n_modes, n_photons)
    return None


def _final_loss(metrics: dict) -> float | None:
    history = metrics.get("training_history")
    if not history:
        return None
    return float(history[-1])


def _series_for_aggregated(path: Path, temperature: str) -> list[dict]:
    import statistics

    payload = agg.load_aggregated(path)
    series: list[dict] = []
    for name, info in payload["points"].items():
        per_seed = info["per_seed_metrics"]
        if not per_seed:
            continue
        aggregated = agg.aggregate_point(per_seed)
        if temperature not in aggregated:
            continue
        dim = _output_dim(per_seed[0])
        if dim is None:
            continue
        losses = [_final_loss(m) for m in per_seed]
        losses = [v for v in losses if v is not None]
        loss_mean = statistics.fmean(losses) if losses else float("nan")
        loss_std = statistics.pstdev(losses) if len(losses) > 1 else 0.0
        series.append(
            {
                "label": info["label"],
                "name": name,
                "output_dim": dim,
                "originality_L2_mean": aggregated[temperature]["originality_L2"][
                    "mean"
                ],
                "originality_L2_std": aggregated[temperature]["originality_L2"]["std"],
                "broken_rate_2_mean": aggregated[temperature]["broken_rate_2"]["mean"],
                "broken_rate_2_std": aggregated[temperature]["broken_rate_2"]["std"],
                "final_loss_mean": loss_mean,
                "final_loss_std": loss_std,
                "n_seeds": aggregated[temperature]["originality_L2"]["n"],
            }
        )
    series.sort(key=lambda d: d["output_dim"])
    return series


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--aggregated",
        nargs="+",
        required=True,
        help="One or more aggregated.json files; each is plotted as a series.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        required=True,
        help="One label per --aggregated entry.",
    )
    parser.add_argument(
        "--temperature",
        type=str,
        default="1.0",
        help="Which temperature value to display.",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--title",
        type=str,
        default=None,
    )
    args = parser.parse_args()

    if len(args.aggregated) != len(args.labels):
        raise SystemExit("--aggregated and --labels must have the same length")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharex=False)
    ax_o, ax_b, ax_l = axes

    markers = ["o", "s", "D", "^", "v"]
    cmap = plt.get_cmap("tab10")
    for idx, (source, label) in enumerate(zip(args.aggregated, args.labels)):
        series = _series_for_aggregated(Path(source), str(args.temperature))
        if not series:
            print(
                f"[plot_scaling] skipping {source}: no series at T={args.temperature}"
            )
            continue
        xs = [s["output_dim"] for s in series]
        o_mean = [s["originality_L2_mean"] for s in series]
        o_std = [s["originality_L2_std"] for s in series]
        b_mean = [s["broken_rate_2_mean"] for s in series]
        b_std = [s["broken_rate_2_std"] for s in series]
        l_mean = [s["final_loss_mean"] for s in series]
        l_std = [s["final_loss_std"] for s in series]
        marker = markers[idx % len(markers)]
        color = cmap(idx)
        ax_o.errorbar(
            xs, o_mean, yerr=o_std, marker=marker, color=color, capsize=3, label=label
        )
        ax_b.errorbar(
            xs, b_mean, yerr=b_std, marker=marker, color=color, capsize=3, label=label
        )
        ax_l.errorbar(
            xs, l_mean, yerr=l_std, marker=marker, color=color, capsize=3, label=label
        )
        # Annotate the n_modes/p notation next to each point.
        for s in series:
            ax_o.annotate(
                s["name"],
                (s["output_dim"], s["originality_L2_mean"]),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=6,
                color=color,
                alpha=0.7,
            )

    title = args.title or f"Scaling sweep at T = {args.temperature}"
    ax_o.set_xscale("log")
    ax_b.set_xscale("log")
    ax_l.set_xscale("log")
    ax_o.set_xlabel("Reservoir output dimension (log)")
    ax_b.set_xlabel("Reservoir output dimension (log)")
    ax_l.set_xlabel("Reservoir output dimension (log)")
    ax_o.set_ylabel("Originality at L=2")
    ax_b.set_ylabel("Broken-rate, rule 2")
    ax_l.set_ylabel("Final FNN cross-entropy")
    ax_o.set_title("Originality vs reservoir size")
    ax_b.set_title("Broken-rate vs reservoir size")
    ax_l.set_title("Training-loss floor vs reservoir size")
    # log(32) = 3.47 reference line (uniform prediction floor)
    ax_l.axhline(
        y=3.4657, color="gray", linestyle=":", alpha=0.7, label="log(32) uniform"
    )
    ax_o.grid(True, alpha=0.2)
    ax_b.grid(True, alpha=0.2)
    ax_l.grid(True, alpha=0.2)
    ax_o.legend(fontsize=8)
    ax_b.legend(fontsize=8)
    ax_l.legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(args.out, dpi=180)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
