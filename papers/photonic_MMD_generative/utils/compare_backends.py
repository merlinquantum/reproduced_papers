"""Phase 04: compare a MerLin-backend sweep with its Phase 03 JAX counterpart (pre-registered rule).

Usage::

    python papers/photonic_MMD_generative/utils/compare_backends.py <merlin_sweep_dir> <jax_sweep_dir> \
        --pair <merlin_label>=<jax_label> [--pair ...] [--out results/<name>.json]

For each pair of candidates and each metric present in both, reports both
means, stds, n, Δ = mean_merlin − mean_jax, SE = sqrt(s_m²/n_m + s_j²/n_j) and
whether |Δ| ≤ 2·SE ("preserved"), as fixed in the Phase 4 Adaptation Plan.
Reads summary.json files only.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

METRICS = (
    "test_mmd_estimator",
    "model.walsh_mmd",
    "model.gaussian_mmd_raw",
    "model.gaussian_mmd_binarized",
    "final_train_mmd_estimate",
    "steps_run",
    "seconds_per_step",
)


def by_label(summary: dict) -> dict:
    return {
        json.loads(c)["label"]: e["metrics"] for c, e in summary["candidates"].items()
    }


def main() -> None:
    """Print and save the comparison table."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("merlin_sweep", type=Path)
    parser.add_argument("jax_sweep", type=Path)
    parser.add_argument("--pair", action="append", required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    merlin = by_label(json.loads((args.merlin_sweep / "summary.json").read_text()))
    jax = by_label(json.loads((args.jax_sweep / "summary.json").read_text()))
    rows = []
    for pair in args.pair:
        m_label, j_label = pair.split("=")
        for metric in METRICS:
            a, b = merlin[m_label].get(metric), jax[j_label].get(metric)
            if not a or not b:
                continue
            se = math.sqrt(
                (a["std"] or 0) ** 2 / a["n"] + (b["std"] or 0) ** 2 / b["n"]
            )
            delta = a["mean"] - b["mean"]
            row = {
                "merlin": m_label,
                "jax": j_label,
                "metric": metric,
                "merlin_mean": a["mean"],
                "merlin_std": a["std"],
                "jax_mean": b["mean"],
                "jax_std": b["std"],
                "n": [a["n"], b["n"]],
                "delta": delta,
                "se": se,
                "preserved": None
                if metric in ("seconds_per_step", "steps_run")
                else abs(delta) <= 2 * se,
            }
            rows.append(row)
            print(
                f"{m_label:24s} vs {j_label:16s} {metric:30s} merlin {a['mean']:.4g}±{(a['std'] or 0):.2g} "
                f"jax {b['mean']:.4g}±{(b['std'] or 0):.2g}  Δ={delta:+.3g}  2SE={2 * se:.3g}  "
                f"{'' if row['preserved'] is None else ('PRESERVED' if row['preserved'] else 'DIFFERENT')}"
            )
    if args.out:
        args.out.write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
