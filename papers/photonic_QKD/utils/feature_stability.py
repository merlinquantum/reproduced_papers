"""Epoch-to-epoch stability of the conditioning vector z, per S4 arm.

For every completed feature-conditioned run, reads ``feature_trace.npz``
(raw and EMA-used z per epoch) and reports the mean relative change of the
z actually used for training, ``||z_t - z_{t-1}|| / ||z_{t-1}||``, next to the
run's final test accuracy. This tests whether the S4 ranking follows feature
stability rather than the feature's origin (photonic or classical).

Usage (from the repository root)::

    python papers/photonic_QKD/utils/feature_stability.py <S4 dir> [<completion dir> ...]

Writes ``results/s4_feature_stability.json`` and ``.md``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

PAPER_DIR = Path(__file__).resolve().parents[1]


def relative_steps(trace: np.ndarray) -> np.ndarray:
    """``||z_t - z_{t-1}|| / ||z_{t-1}||`` for consecutive epochs."""
    differences = np.linalg.norm(np.diff(trace, axis=0), axis=1)
    norms = np.linalg.norm(trace[:-1], axis=1)
    return differences / np.maximum(norms, 1e-12)


def collect(sweep_dirs: list[Path]) -> dict[str, list[dict[str, float]]]:
    arms: dict[str, list[dict[str, float]]] = {}
    for sweep_dir in sweep_dirs:
        status = json.loads((sweep_dir / "sweep_status.json").read_text())
        for run in status["runs"]:
            if run["status"] != "COMPLETED":
                continue
            run_dir = sweep_dir / run["run_dir"]
            trace_path = run_dir / "feature_trace.npz"
            if not trace_path.exists():
                continue
            metrics = json.loads((run_dir / "metrics.json").read_text())
            trace = np.load(trace_path)
            used = relative_steps(trace["used"])
            raw = relative_steps(trace["raw"])
            arms.setdefault(run["candidate"], []).append(
                {
                    "seed": run["seed"],
                    "used_step_mean": float(used.mean()),
                    "used_step_last10": float(used[-10:].mean()),
                    "raw_step_mean": float(raw.mean()),
                    "test_accuracy": float(
                        metrics["final_evaluation"]["test_accuracy"]
                    ),
                }
            )
    return arms


def _ranks(values: list[float]) -> np.ndarray:
    """Ranks with ties given their average rank."""
    array = np.asarray(values, dtype=float)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array))
    ranks[order] = np.arange(len(array), dtype=float)
    for value in np.unique(array):
        tied = array == value
        ranks[tied] = ranks[tied].mean()
    return ranks


def spearman(x: list[float], y: list[float]) -> float:
    """Spearman rank correlation (Pearson on tie-averaged ranks)."""
    if len(x) < 3:
        return float("nan")
    return float(np.corrcoef(_ranks(x), _ranks(y))[0, 1])


def _mean_std(values: list[float]) -> tuple[float, float]:
    mean = sum(values) / len(values)
    std = (
        math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))
        if len(values) > 1
        else 0.0
    )
    return mean, std


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("sweep_dirs", type=Path, nargs="+")
    args = parser.parse_args()
    arms = collect([path.resolve() for path in args.sweep_dirs])
    rows = []
    for arm, runs in sorted(arms.items()):
        used_mean, used_std = _mean_std([run["used_step_mean"] for run in runs])
        raw_mean, raw_std = _mean_std([run["raw_step_mean"] for run in runs])
        test_mean, test_std = _mean_std([run["test_accuracy"] for run in runs])
        rows.append(
            {
                "arm": arm,
                "n": len(runs),
                "used_step_mean": [used_mean, used_std],
                "raw_step_mean": [raw_mean, raw_std],
                "test_accuracy": [test_mean, test_std],
                "runs": runs,
            }
        )
    rows.sort(key=lambda row: row["used_step_mean"][0])
    # B0 (z = 0) is a broken network at chance accuracy, so it is excluded
    # from the correlation; it stays in the table.
    pairs = [
        (run["used_step_mean"], run["test_accuracy"])
        for arm, runs in arms.items()
        if arm != "B0_zero"
        for run in runs
    ]
    # Rank correlation: fixed-z arms have a step of exactly 0, which a log
    # scale would need an arbitrary floor for.
    correlation = spearman([p[0] for p in pairs], [p[1] for p in pairs])
    report = {
        "metric": "mean over epochs of ||z_t - z_(t-1)|| / ||z_(t-1)|| for the z used in training (EMA applied when enabled)",
        "spearman_step_vs_test_accuracy": correlation,
        "n_runs": len(pairs),
        "rows": rows,
    }
    lines = [
        "# S4 conditioning-vector stability vs accuracy",
        "",
        f"Metric: {report['metric']}",
        "",
        "| arm | n | used-z relative step | raw-z relative step | test acc % |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['arm']} | {row['n']} | {row['used_step_mean'][0]:.4f} ± {row['used_step_mean'][1]:.4f} | "
            f"{row['raw_step_mean'][0]:.4f} ± {row['raw_step_mean'][1]:.4f} | "
            f"{row['test_accuracy'][0]:.2f} ± {row['test_accuracy'][1]:.2f} |"
        )
    lines += [
        "",
        f"Spearman rank correlation (used-z step vs test accuracy, per run, B0 excluded, n={len(pairs)}): {correlation:.3f}",
        "",
    ]
    text = "\n".join(lines)
    results_dir = PAPER_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    (results_dir / "s4_feature_stability.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    (results_dir / "s4_feature_stability.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
