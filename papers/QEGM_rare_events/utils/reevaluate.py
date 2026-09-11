"""Re-score saved samples of a finished run at one or more tail thresholds.

The headline reproduction used ``tail_threshold = 1.5`` (matching the
paper's footnote describing the GMM tails). That threshold saturates
``rare_event_recall`` at 1.0 across all variants. This utility re-evaluates
the same saved sample arrays at stricter thresholds (e.g. 2.5, 3.0) so
recall becomes discriminative again, and also uses the paper's
rarity-score-style definition ``s(x) = −log p(x)`` to define the tail
when ``--rarity-score`` is set.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

_PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

from lib.metrics import (  # noqa: E402
    coverage_calibration,
    rare_event_recall,
    summarize,
    tail_kl_divergence,
)

_SAMPLE_PATTERN = re.compile(r"samples_(?P<variant>.+)_seed(?P<seed>-?\d+)\.npy$")


def _collect(run_dir: Path):
    grouped: dict[str, list[tuple[int, Path]]] = {}
    for path in sorted(run_dir.glob("samples_*_seed*.npy")):
        m = _SAMPLE_PATTERN.search(path.name)
        if not m:
            continue
        variant = m.group("variant")
        seed = int(m.group("seed"))
        grouped.setdefault(variant, []).append((seed, path))
    return grouped


def _gmm_log_pdf(x: np.ndarray, means, stds, weights) -> np.ndarray:
    means = np.asarray(means).reshape(1, -1)
    stds = np.asarray(stds).reshape(1, -1)
    weights = np.asarray(weights).reshape(1, -1)
    weights = weights / weights.sum()
    x = x.reshape(-1, 1)
    log_comp = -0.5 * ((x - means) / stds) ** 2 - np.log(stds * np.sqrt(2 * np.pi))
    return np.log((weights * np.exp(log_comp)).sum(axis=1) + 1e-30)


def reevaluate(
    run_dir: Path, tail_thresholds: list[float], use_rarity_score: bool
) -> dict:
    real = np.load(run_dir / "real_samples_test.npy").flatten()
    metrics_payload = json.loads((run_dir / "metrics.json").read_text())
    cfg_ds = metrics_payload.get("config_dataset", {})

    if use_rarity_score:
        means = cfg_ds.get("means", [-3.0, 0.0, 3.0])
        stds = cfg_ds.get("stds", [1.0, 0.7071, 1.2247])
        weights = cfg_ds.get("weights", [0.15, 0.7, 0.15])
        # Negative log-pdf, then quantile to a rarity threshold τ.
        nll_real = -_gmm_log_pdf(real, means, stds, weights)

    grouped = _collect(run_dir)
    out: dict = {"thresholds": tail_thresholds, "rarity_score": bool(use_rarity_score)}

    for variant, items in grouped.items():
        per_seed_per_threshold: dict[str, list[dict]] = {}
        for thr in tail_thresholds:
            per_seed: list[dict] = []
            if use_rarity_score:
                # Define the rarity threshold as the (1 - quantile) of NLL on real samples.
                # thr = 0.05 means top-5% rarest by NLL.
                if not 0.0 < thr < 1.0:
                    raise ValueError(
                        f"With --rarity-score, thresholds are tail-mass fractions "
                        f"in (0, 1); got {thr}. Pass e.g. --thresholds 0.05,0.1,0.2"
                    )
                quantile = 1.0 - thr
                tau = float(np.quantile(nll_real, quantile))
                real_tail_mask = nll_real >= tau
            for seed, path in items:
                generated = np.load(path).flatten()
                if use_rarity_score:
                    nll_gen = -_gmm_log_pdf(generated, means, stds, weights)
                    real_t = real[real_tail_mask]
                    gen_t = generated[nll_gen >= tau]
                    metrics = {
                        "tail_kl": tail_kl_divergence(
                            real_t,
                            gen_t,
                            tail_threshold=0.0,
                            n_bins=20,
                            eps=1e-4,
                        ),
                        "rare_recall": rare_event_recall(
                            real_t,
                            gen_t,
                            tail_threshold=0.0,
                        ),
                        "coverage": coverage_calibration(real, generated),
                    }
                else:
                    metrics = {
                        "tail_kl": tail_kl_divergence(
                            real,
                            generated,
                            tail_threshold=thr,
                            n_bins=20,
                            eps=1e-4,
                        ),
                        "rare_recall": rare_event_recall(
                            real,
                            generated,
                            tail_threshold=thr,
                        ),
                        "coverage": coverage_calibration(real, generated),
                    }
                per_seed.append({"seed": seed, **metrics})
            per_seed_per_threshold[str(thr)] = per_seed
        out[variant] = {
            "per_seed": per_seed_per_threshold,
            "summary": {
                str(thr): summarize(
                    [
                        {k: v for k, v in s.items() if k != "seed"}
                        for s in per_seed_per_threshold[str(thr)]
                    ]
                )
                for thr in tail_thresholds
            },
        }
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="Path to a finished run directory")
    parser.add_argument(
        "--thresholds",
        default="1.5,2.5,3.0",
        help="Comma-separated absolute thresholds (or quantile-tail fractions if --rarity-score).",
    )
    parser.add_argument(
        "--rarity-score",
        action="store_true",
        help="Use the paper's rarity-score definition s(x) = -log p(x); --thresholds become tail-mass fractions.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output JSON path (defaults to run_dir/metrics_reeval.json).",
    )
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    thresholds = [float(v) for v in args.thresholds.split(",")]
    payload = reevaluate(run_dir, thresholds, args.rarity_score)

    out_path = (
        Path(args.out)
        if args.out
        else run_dir
        / ("metrics_reeval_rarity.json" if args.rarity_score else "metrics_reeval.json")
    )
    out_path.write_text(json.dumps(payload, indent=2))
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
