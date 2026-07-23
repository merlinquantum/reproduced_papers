"""Aggregate metrics across the main run and the two rigor ablations.

Reads the canonical run (full hybrid loss) plus the const-r and λ_tail=0
ablation runs and prints a single comparison table. Used by the README
and Confluence summaries after the ablations finish.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_PATHS = {
    "main (λ_tail=2, full VQC)": "outdir/run_20260527-123627",
    "ablation: const r=0.5": "outdir/ablation_const",
    "ablation: λ_tail = 0": "outdir/ablation_no_tail",
}


def _row(name: str, summary: dict, variant: str) -> str:
    s = summary.get(variant)
    if s is None:
        return f"  {variant:14s}    n/a"
    kl = s["tail_kl"]
    rec = s["rare_recall"]
    return (
        f"  {variant:14s}    tail_kl = {kl['mean']:.3f} ± {kl['std']:.3f}    "
        f"recall = {rec['mean']:.3f} ± {rec['std']:.3f}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--main",
        default=DEFAULT_PATHS["main (λ_tail=2, full VQC)"],
        help="Main run dir",
    )
    parser.add_argument(
        "--const",
        default=DEFAULT_PATHS["ablation: const r=0.5"],
        help="Const-r run dir",
    )
    parser.add_argument(
        "--no-tail",
        default=DEFAULT_PATHS["ablation: λ_tail = 0"],
        help="λ_tail=0 run dir",
    )
    parser.add_argument(
        "--root",
        default="/reproduced_papers/papers/QEGM_rare_events",
        help="Project root for resolving relative paths.",
    )
    parser.add_argument(
        "--json", action="store_true", help="Emit a single JSON payload."
    )
    args = parser.parse_args(argv)

    root = Path(args.root)
    blocks = {
        "main (λ_tail=2, full VQC)": Path(args.main),
        "ablation: const r=0.5": Path(args.const),
        "ablation: λ_tail = 0": Path(args.no_tail),
    }

    payload = {}
    for label, p in blocks.items():
        full = (root / p) if not p.is_absolute() else p
        metrics_path = full / "metrics.json"
        if not metrics_path.exists():
            print(f"# missing: {metrics_path}")
            continue
        data = json.loads(metrics_path.read_text())
        payload[label] = data["summary"]

    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    variants_order = ["vae", "qegm", "qegm_merlin", "qegm_const"]
    for label, summary in payload.items():
        print(f"== {label} ==")
        for variant in variants_order:
            if variant in summary:
                print(_row(label, summary, variant))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
