"""Significance of QNN-vs-ANN test-AUC differences for Task 1 repetition sweeps.

For each sweep directory passed (or for all of them pooled with
``--combine NAME``), this reads every completed seed's
``test_predictions.csv`` and reports, per comparison:

- the per-seed AUC difference with a paired moving-block bootstrap 95% CI and
  two-sided p-value (resampling test days jointly for both models; blocks of
  ``--block`` days keep short-range dependence);
- a seed-level one-sample t-test of the per-seed differences (df = seeds - 1);
- the same bootstrap on the seed-averaged probabilities (ensemble of seeds).

Outputs ``results/<sweep_name>_statistics.csv`` plus a copy in the sweep
directory. Usage (repository root)::

    .venv/bin/python papers/QML_financial_benchmark/utils/stats_directional.py \\
        papers/QML_financial_benchmark/outdir/sweeps/e3_aapl_<stamp>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT_DIR.parents[1]), str(PROJECT_DIR)]

from lib.metrics import safe_auc  # noqa: E402

COMPARISONS = [
    ("leak_free:qnn", "leak_free:ann"),
    ("leak_free:qnn", "leak_free:ann_parameter_matched"),
    ("leak_free:qnn", "logistic_regression"),
    ("leak_free:ann", "logistic_regression"),
    ("paper:qnn", "paper:ann"),
    ("paper:qnn", "paper:ann_parameter_matched"),
]


def block_bootstrap_indices(
    n: int, block: int, generator: np.random.Generator
) -> np.ndarray:
    """Moving-block bootstrap resample of ``n`` positions."""
    starts = generator.integers(0, n - block + 1, size=int(np.ceil(n / block)))
    return (starts[:, None] + np.arange(block)[None, :]).reshape(-1)[:n]


def paired_auc_difference(
    labels, scores_a, scores_b, block: int, resamples: int, seed: int
) -> dict:
    """AUC(a) - AUC(b) on shared days with a paired block-bootstrap CI and p-value."""
    generator = np.random.default_rng(seed)
    observed = safe_auc(labels, scores_a) - safe_auc(labels, scores_b)
    differences = []
    for _ in range(resamples):
        rows = block_bootstrap_indices(len(labels), block, generator)
        if len(np.unique(labels[rows])) < 2:
            continue
        differences.append(
            safe_auc(labels[rows], scores_a[rows])
            - safe_auc(labels[rows], scores_b[rows])
        )
    differences = np.asarray(differences)
    # Two-sided p-value from the bootstrap distribution centred on zero.
    p_value = float(np.mean(np.abs(differences - differences.mean()) >= abs(observed)))
    return {
        "difference": float(observed),
        "ci_low": float(np.quantile(differences, 0.025)),
        "ci_high": float(np.quantile(differences, 0.975)),
        "bootstrap_p": p_value,
        "resamples_used": int(len(differences)),
    }


def variant_of(run: dict) -> str:
    """Variant label of a run: the first path component for multi-variant sweeps, else ``main``."""
    parts = Path(run["run_dir"]).parts
    return parts[0] if len(parts) == 3 else "main"


def analyse_sweeps(sweep_dirs: list[Path], block: int, resamples: int) -> pd.DataFrame:
    """All comparisons per variant, pooling completed runs of the given sweeps.

    Raises
    ------
    ValueError
        If no run completed, or two pooled runs share a variant and seed.
    """
    runs_by_variant: dict[str, dict[int, pd.DataFrame]] = {}
    for sweep_dir in sweep_dirs:
        status = json.loads((sweep_dir / "sweep_status.json").read_text())
        for run in status["runs"]:
            if run["status"] != "COMPLETED":
                continue
            seeds = runs_by_variant.setdefault(variant_of(run), {})
            if run["seed"] in seeds:
                raise ValueError(
                    f"Duplicate seed {run['seed']} for variant {variant_of(run)}"
                )
            seeds[run["seed"]] = pd.read_csv(
                sweep_dir / run["run_dir"] / "test_predictions.csv"
            )
    if not runs_by_variant:
        raise ValueError(f"No completed runs in {sweep_dirs}")
    sweep_names = "+".join(sweep_dir.name for sweep_dir in sweep_dirs)
    return pd.concat(
        [
            analyse_variant(predictions, block, resamples).assign(
                sweep=sweep_names, variant=variant
            )
            for variant, predictions in sorted(runs_by_variant.items())
        ],
        ignore_index=True,
    )


def analyse_variant(
    predictions: dict[int, pd.DataFrame], block: int, resamples: int
) -> pd.DataFrame:
    """Per-seed, seed-level and seed-ensemble comparisons for one variant's seeds."""
    rows = []
    for model_a, model_b in COMPARISONS:
        per_seed = []
        for seed, frame in sorted(predictions.items()):
            labels = frame["label"].to_numpy()
            result = paired_auc_difference(
                labels,
                frame[model_a].to_numpy(),
                frame[model_b].to_numpy(),
                block,
                resamples,
                seed,
            )
            per_seed.append(result["difference"])
            rows.append(
                {
                    "comparison": f"{model_a} - {model_b}",
                    "scope": f"seed {seed}",
                    **result,
                }
            )
        t_statistic, t_p = stats.ttest_1samp(per_seed, 0.0)
        rows.append(
            {
                "comparison": f"{model_a} - {model_b}",
                "scope": f"seed-level t-test (n={len(per_seed)})",
                "difference": float(np.mean(per_seed)),
                "ci_low": float(
                    np.mean(per_seed)
                    - stats.t.ppf(0.975, len(per_seed) - 1)
                    * np.std(per_seed, ddof=1)
                    / np.sqrt(len(per_seed))
                ),
                "ci_high": float(
                    np.mean(per_seed)
                    + stats.t.ppf(0.975, len(per_seed) - 1)
                    * np.std(per_seed, ddof=1)
                    / np.sqrt(len(per_seed))
                ),
                "t_statistic": float(t_statistic),
                "t_p": float(t_p),
            }
        )
        frames = list(predictions.values())
        labels = frames[0]["label"].to_numpy()
        if any(
            not np.array_equal(frame["label"].to_numpy(), labels) for frame in frames
        ):
            raise ValueError("Seeds disagree on test labels; runs are not comparable")
        ensemble_a = np.mean([frame[model_a].to_numpy() for frame in frames], axis=0)
        ensemble_b = np.mean([frame[model_b].to_numpy() for frame in frames], axis=0)
        rows.append(
            {
                "comparison": f"{model_a} - {model_b}",
                "scope": "seed-ensemble",
                **paired_auc_difference(
                    labels, ensemble_a, ensemble_b, block, resamples, 12345
                ),
            }
        )
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("sweep_dirs", nargs="+", type=Path)
    parser.add_argument(
        "--block",
        type=int,
        default=5,
        help="Bootstrap block length in trading days. Default 5.",
    )
    parser.add_argument(
        "--resamples", type=int, default=2000, help="Bootstrap resamples. Default 2000."
    )
    parser.add_argument(
        "--combine",
        help="Pool all given sweeps (e.g. seed 0 and seeds 1-2 of the same variants) under this result name",
    )
    args = parser.parse_args(argv)
    results_dir = PROJECT_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    groups = (
        [(args.combine, [path.resolve() for path in args.sweep_dirs])]
        if args.combine
        else [
            (path.resolve().name.rsplit("_", 1)[0], [path.resolve()])
            for path in args.sweep_dirs
        ]
    )
    for name, sweep_dirs in groups:
        table = analyse_sweeps(sweep_dirs, args.block, args.resamples)
        table.to_csv(results_dir / f"{name}_statistics.csv", index=False)
        for sweep_dir in sweep_dirs:
            table.to_csv(
                sweep_dir / f"{name}_statistics.csv"
                if args.combine
                else sweep_dir / "statistics.csv",
                index=False,
            )
        with pd.option_context("display.width", 220, "display.max_columns", 20):
            print(table.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
