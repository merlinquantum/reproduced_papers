"""Pretty-print sweep results as a Markdown-ready table.

Usage::

    python utils/print_sweep_table.py --aggregated sweeps/modes/aggregated.json
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


def _fmt(value_dict) -> str:
    n = value_dict.get("n", 0)
    if n == 0:
        return "n/a"
    return f"{value_dict['mean']:.3f} ± {value_dict['std']:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregated", type=Path, required=True)
    parser.add_argument(
        "--temperatures",
        nargs="+",
        default=["0.5", "1.0", "2.0", "5.0"],
    )
    args = parser.parse_args()

    payload = agg.load_aggregated(args.aggregated)
    seeds = payload.get("seeds", [])

    print(f"\n# Sweep summary — {args.aggregated} (n_seeds={len(seeds)})\n")
    header = (
        ["point", "label", "dim"]
        + [f"orig_L2 @T={t}" for t in args.temperatures]
        + [f"broken_2 @T={t}" for t in args.temperatures]
    )
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join("---" for _ in header) + "|")

    rows = []
    for name, info in payload["points"].items():
        per_seed = info["per_seed_metrics"]
        if not per_seed:
            continue
        aggregated = agg.aggregate_point(per_seed)
        dim = _output_dim(per_seed[0])
        cells = [name, info["label"], str(dim) if dim is not None else "?"]
        for t in args.temperatures:
            row = aggregated.get(t)
            cells.append(_fmt(row["originality_L2"]) if row else "n/a")
        for t in args.temperatures:
            row = aggregated.get(t)
            cells.append(_fmt(row["broken_rate_2"]) if row else "n/a")
        rows.append((dim if dim is not None else 0, cells))

    rows.sort(key=lambda x: x[0])
    for _, cells in rows:
        print("| " + " | ".join(cells) + " |")

    # Baselines (one row, taken from the first point's per-seed metrics)
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
        print()
        print("## Baselines (n_seeds = same as sweep)")
        print("| baseline | orig_L2 | orig_L10 | broken_2 | broken_3 |")
        print("|---|---|---|---|---|")
        for name in ("markov", "uncorrelated"):
            if name in b:
                row = b[name]
                print(
                    f"| {name} | {_fmt(row['originality_L2'])} | "
                    f"{_fmt(row['originality_L10'])} | {_fmt(row['broken_rate_2'])} | "
                    f"{_fmt(row['broken_rate_3'])} |"
                )


if __name__ == "__main__":
    main()
