"""Like plot_scaling.py but splits an aggregated.json by backend into two series.

The default scaling plot collapses every point into one series. For the
``isodim`` sweep we want to overlay qubit and photonic series so the
backend-axis difference is visible.
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


def _backend_and_dim(metrics: dict) -> tuple[str, int]:
    cfg = metrics.get("config_excerpt", {}).get("qrc", {})
    backend = cfg.get("backend")
    if backend == "qubit":
        return "qubit", 2 ** int(cfg.get("n_qubits", 0))
    if backend == "photonic":
        return "photonic", comb(
            int(cfg.get("n_modes", 0)), int(cfg.get("n_photons", 0))
        )
    return "unknown", 0


def _series_split(path: Path, temperature: str) -> dict[str, list[dict]]:
    import statistics

    payload = agg.load_aggregated(path)
    by_backend: dict[str, list[dict]] = {"qubit": [], "photonic": []}
    for name, info in payload["points"].items():
        per_seed = info["per_seed_metrics"]
        if not per_seed:
            continue
        backend, dim = _backend_and_dim(per_seed[0])
        if backend not in by_backend:
            continue
        aggregated = agg.aggregate_point(per_seed)
        if temperature not in aggregated:
            continue
        losses = [
            m.get("training_history", [None])[-1]
            for m in per_seed
            if m.get("training_history")
        ]
        loss_mean = statistics.fmean(losses) if losses else float("nan")
        loss_std = statistics.pstdev(losses) if len(losses) > 1 else 0.0
        by_backend[backend].append(
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
            }
        )
    for backend in by_backend:
        by_backend[backend].sort(key=lambda d: d["output_dim"])
    return by_backend


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregated", type=Path, required=True)
    parser.add_argument("--temperature", type=str, default="1.0")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_backend = _series_split(args.aggregated, str(args.temperature))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    ax_o, ax_b, ax_l = axes

    style = {
        "qubit": {"marker": "s", "color": "C0", "linestyle": "-", "label": "qubit"},
        "photonic": {
            "marker": "o",
            "color": "C3",
            "linestyle": "-",
            "label": "photonic (UNBUNCHED)",
        },
    }

    for backend in ("qubit", "photonic"):
        series = by_backend.get(backend, [])
        if not series:
            continue
        xs = [s["output_dim"] for s in series]
        s_ = style[backend]
        ax_o.errorbar(
            xs,
            [s["originality_L2_mean"] for s in series],
            yerr=[s["originality_L2_std"] for s in series],
            capsize=3,
            **s_,
        )
        ax_b.errorbar(
            xs,
            [s["broken_rate_2_mean"] for s in series],
            yerr=[s["broken_rate_2_std"] for s in series],
            capsize=3,
            **s_,
        )
        ax_l.errorbar(
            xs,
            [s["final_loss_mean"] for s in series],
            yerr=[s["final_loss_std"] for s in series],
            capsize=3,
            **s_,
        )
        for s in series:
            ax_o.annotate(
                s["name"],
                (s["output_dim"], s["originality_L2_mean"]),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=6,
                color=s_["color"],
                alpha=0.7,
            )

    # Baselines as horizontal references on the originality and broken-rate axes.
    payload = agg.load_aggregated(args.aggregated)
    first_per_seed = next(
        (
            info["per_seed_metrics"]
            for info in payload["points"].values()
            if info["per_seed_metrics"]
        ),
        [],
    )
    if first_per_seed:
        b = agg.baseline_summary(first_per_seed)
        if "markov" in b:
            ax_o.axhline(
                b["markov"]["originality_L2"]["mean"],
                color="black",
                linestyle="--",
                alpha=0.5,
                label="Markov",
            )
            ax_b.axhline(
                b["markov"]["broken_rate_2"]["mean"],
                color="black",
                linestyle="--",
                alpha=0.5,
                label="Markov",
            )
        if "uncorrelated" in b:
            ax_o.axhline(
                b["uncorrelated"]["originality_L2"]["mean"],
                color="gray",
                linestyle="-.",
                alpha=0.5,
                label="Uncorrelated",
            )
            ax_b.axhline(
                b["uncorrelated"]["broken_rate_2"]["mean"],
                color="gray",
                linestyle="-.",
                alpha=0.5,
                label="Uncorrelated",
            )

    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("Reservoir output dimension (log)")
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=8, loc="best")
    ax_o.set_ylabel("Originality at L=2")
    ax_b.set_ylabel("Broken-rate, rule 2")
    ax_l.set_ylabel("Final FNN cross-entropy")
    ax_o.set_title("Originality vs reservoir size")
    ax_b.set_title("Broken-rate vs reservoir size")
    ax_l.set_title("Training-loss floor")
    ax_l.axhline(
        y=3.4657, color="gray", linestyle=":", alpha=0.7, label="log(32) uniform"
    )
    ax_l.legend(fontsize=8)
    fig.suptitle(
        f"Iso-output-dim qubit vs photonic at T={args.temperature} (n_seeds=3)"
    )
    fig.tight_layout()
    fig.savefig(args.out, dpi=180)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
