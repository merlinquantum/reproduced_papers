"""Export per-run training-curve facts from sweep folders into ``results/``.

Raw run histories live in the git-ignored ``outdir/``. The notebook and the
README cite two facts derived from them:
- whether a model is still under-trained at the end of the budget (train
  metric at the selected epoch, best validation by an earlier epoch);
- how fast a model learns (first epoch the validation metric crosses a
  threshold).

This script writes those facts, per run, to a small JSON file so the
evidence is committed.

Usage (from ``papers/QPE_GNN``)::

    python utils/export_history_summary.py \
        --sweep-dir outdir/sweep_zinc_reduced_seeds01 \
        --sweep-dir outdir/sweep_zinc_reduced_seeds23 \
        --metric mae --direction minimize --checkpoint-epoch 150 \
        --output results/zinc_reduced/training_history_summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def summarise_run(
    run_dir: Path,
    metric: str,
    direction: str,
    checkpoint_epoch: int | None,
    threshold: float | None,
) -> dict:
    """Extract curve facts from one completed run's ``metrics.json``.

    Parameters
    ----------
    run_dir : pathlib.Path
        Run folder holding ``metrics.json``.
    metric : str
        History key suffix, e.g. ``"mae"`` or ``"acc"``.
    direction : str
        ``"minimize"`` or ``"maximize"``.
    checkpoint_epoch : int | None
        Report the best validation value reached by this epoch. Default value
        is None (not reported).
    threshold : float | None
        Report the first epoch whose validation value crosses it. Default
        value is None (not reported).

    Returns
    -------
    dict
        Per-run facts with 1-based epochs.
    """
    metrics = _read_json(run_dir / "metrics.json")
    history = metrics["history"]
    validation = history[f"val_{metric}"]
    train = history[f"train_{metric}"]
    best_epoch = int(metrics["best_epoch"])
    better = min if direction == "minimize" else max
    record = {
        "run_id": metrics["run_id"],
        "seed": metrics["seed"],
        "epochs_completed": metrics["epochs_completed"],
        "best_epoch": best_epoch,
        f"train_{metric}_at_best_epoch": train[best_epoch - 1],
        f"val_{metric}_at_best_epoch": validation[best_epoch - 1],
        f"test_{metric}": metrics["test_metrics"][metric],
        f"best_val_{metric}": better(validation),
        f"train_{metric}_epoch_1": train[0],
        f"train_{metric}_last_10_mean": sum(train[-10:]) / len(train[-10:]),
    }
    if checkpoint_epoch is not None:
        record[f"best_val_{metric}_by_epoch_{checkpoint_epoch}"] = better(
            validation[:checkpoint_epoch]
        )
    if threshold is not None:
        crossed = [
            epoch
            for epoch, value in enumerate(validation, start=1)
            if (value <= threshold if direction == "minimize" else value >= threshold)
        ]
        record[f"first_epoch_val_{metric}_crosses_{threshold}"] = (
            crossed[0] if crossed else None
        )
    return record


def export(
    sweep_dirs: list[Path],
    metric: str,
    direction: str,
    checkpoint_epoch: int | None,
    threshold: float | None,
) -> dict:
    """Summarise every completed run of the given sweep folders."""
    runs = []
    for sweep_dir in sweep_dirs:
        status = _read_json(sweep_dir / "sweep_status.json")
        for entry in status["runs"]:
            if entry["status"] != "COMPLETED":
                raise ValueError(
                    f"{entry['run_id']} in {sweep_dir} is {entry['status']}"
                )
            record = summarise_run(
                sweep_dir / entry["run_dir"],
                metric,
                direction,
                checkpoint_epoch,
                threshold,
            )
            record["candidate"] = entry["candidate"]
            record["source"] = f"{sweep_dir.name}/{entry['run_dir']}"
            runs.append(record)
    runs.sort(key=lambda r: (r["candidate"], r["seed"]))
    return {
        "metric": metric,
        "direction": direction,
        "checkpoint_epoch": checkpoint_epoch,
        "threshold": threshold,
        "sweep_dirs": [d.name for d in sweep_dirs],
        "runs": runs,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sweep-dir", type=Path, action="append", required=True)
    parser.add_argument("--metric", required=True)
    parser.add_argument("--direction", choices=("minimize", "maximize"), required=True)
    parser.add_argument("--checkpoint-epoch", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else PROJECT_DIR / path

    summary = export(
        [resolve(d) for d in args.sweep_dir],
        args.metric,
        args.direction,
        args.checkpoint_epoch,
        args.threshold,
    )
    output = resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"{output}: {len(summary['runs'])} runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
