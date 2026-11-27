"""Exploratory diagnostic: per-n input scale that best matches the paper's d_QK(E1).

E1's fidelity kernel ``prod_j cos^2(s (x_ij - x_kj))`` has no free parameter
other than the input scale ``s``, so the scale at which its effective dimension
equals the paper's (journal Fig. 3a) indicates the paper's effective input
scale at each n. This is a descriptive, unranked analysis: a free scale fitted
per n matches the target by construction, so the result only shows *how* the
effective scale would have to vary with n. It does not validate a convention.

Usage (from the paper folder)::

    python utils/diagnose_input_scale.py --seeds 0,1 --output results/diagnostic_input_scale

Writes ``<output>.json`` and ``<output>.md``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parents[1]
sys.path[:0] = [str(REPO_ROOT), str(PROJECT_DIR)]

from lib import embeddings, geometry, paper_values, study  # noqa: E402
from lib.data import build_pca_features, subsample_indices  # noqa: E402

SIZES = (3, 4, 5, 6, 8, 10, 12, 14, 16, 20, 24, 28)
SCALES = tuple(round(float(s), 3) for s in np.arange(0.15, 0.6001, 0.025))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", default="0,1")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--data-root", default=str(REPO_ROOT / "data"))
    args = parser.parse_args(argv)
    seeds = [int(s) for s in args.seeds.split(",")]
    reference = paper_values.load_paper_values(args.data_root)
    pca = build_pca_features(args.data_root, max(SIZES))
    subsamples = [
        subsample_indices(
            pca.labels, 800, study.stream_rng(seed, study.STREAM_SUBSAMPLE_ALL)
        )
        for seed in seeds
    ]
    rows = []
    for n in SIZES:
        target = reference[("dimension_geometry", "d_quantum_kernel", "E1", n, None)]
        dims = {
            scale: float(
                np.mean(
                    [
                        geometry.effective_dimension(
                            embeddings.e1_quantum_kernel(scale * pca.features(n, idx))
                        )
                        for idx in subsamples
                    ]
                )
            )
            for scale in SCALES
        }
        best = min(SCALES, key=lambda s: abs(np.log(dims[s] / target)))
        rows.append(
            {
                "n": n,
                "paper_d": target,
                "best_scale": best,
                "d_at_best": dims[best],
                "d_at_0.3": dims[0.3],
                "d_by_scale": dims,
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(
        json.dumps({"seeds": seeds, "scales": SCALES, "rows": rows}, indent=2) + "\n"
    )
    lines = [
        "# Exploratory: per-n input scale matching the paper's d_QK(E1)",
        "",
        f"Seeds {seeds}; scale grid 0.15-0.60 (step 0.025); E1 analytic kernel. Unranked, descriptive.",
        "",
        "| n | paper d | best scale | ours at best | ours at 0.3 |",
        "|---|---|---|---|---|",
    ]
    lines += [
        f"| {r['n']} | {r['paper_d']:.2f} | {r['best_scale']:.3f} | {r['d_at_best']:.2f} | {r['d_at_0.3']:.2f} |"
        for r in rows
    ]
    args.output.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
