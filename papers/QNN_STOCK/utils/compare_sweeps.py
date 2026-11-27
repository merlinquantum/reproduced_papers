"""Paired-by-seed comparison of the same candidates between two sweeps.

Used to compare a code change on identical configs, seeds and data, e.g. the
QMTL asset block of S4 (Fig. 7) against S1. Reads only each sweep's
``sweep_status.json`` and ``metrics.json`` files and writes ``--out``.

Usage::

    python papers/QNN_STOCK/utils/compare_sweeps.py <new_sweep> <old_sweep> \
        --candidates e4_qmtl_original e4_qmtl_ma5 --out results/phase3_s4/qmtl_fig7_vs_s1.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from pathlib import Path

PAPER_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PAPER_DIR / "utils"))

from summarize_sweep import T_975, load_rows  # noqa: E402

METRICS = (
    "final_fidelity",
    "inference/test/log_loss",
    "output/train/kl_joint",
    "inference/test/accuracy",
)


def compare(new_dir: Path, old_dir: Path, candidates: list[str]) -> list[dict]:
    """Pair candidates by seed; ``NEW=OLD`` pairs differently named candidates.

    A plain name is looked up under the same name in both sweeps. Rows are
    reported under the new sweep's candidate name.
    """

    pairs = dict(
        item.split("=", 1) if "=" in item else (item, item) for item in candidates
    )
    old_to_new = {old: new for new, old in pairs.items()}

    def index(rows, names):
        return {
            (names[r["candidate"]], r["model"], r["asset"], r["metric"], r["seed"]): r[
                "value"
            ]
            for r in rows
            if r["candidate"] in names and r["metric"] in METRICS
        }

    new = index(load_rows(new_dir), {name: name for name in pairs})
    old = index(load_rows(old_dir), old_to_new)
    groups = sorted({key[:4] for key in new})
    out = []
    for candidate, model, asset, metric in groups:
        seeds = sorted(
            s
            for (c, m, a, me, s) in new
            if (c, m, a, me) == (candidate, model, asset, metric)
            and (candidate, model, asset, metric, s) in old
        )
        if len(seeds) < 2:
            raise ValueError(
                f"fewer than two paired seeds for {candidate} {asset} {metric}"
            )
        new_values = [new[(candidate, model, asset, metric, s)] for s in seeds]
        old_values = [old[(candidate, model, asset, metric, s)] for s in seeds]
        diffs = [n - o for n, o in zip(new_values, old_values)]
        mean, std = statistics.fmean(diffs), statistics.stdev(diffs)
        half = T_975[len(diffs) - 1] * std / math.sqrt(len(diffs))
        out.append(
            {
                "candidate": candidate,
                "model": model,
                "asset": asset,
                "metric": metric,
                "n_pairs": len(diffs),
                "new_mean": statistics.fmean(new_values),
                "old_mean": statistics.fmean(old_values),
                "mean_diff": mean,
                "ci95_low": mean - half,
                "ci95_high": mean + half,
                "verdict": "not distinguishable"
                if mean - half <= 0 <= mean + half
                else ("new lower" if mean + half < 0 else "new higher"),
            }
        )
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("new_sweep", type=Path)
    parser.add_argument("old_sweep", type=Path)
    parser.add_argument(
        "--candidates",
        nargs="+",
        required=True,
        help="Candidate names, or NEW=OLD to pair differently named candidates",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    rows = compare(args.new_sweep, args.old_sweep, args.candidates)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} rows written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
