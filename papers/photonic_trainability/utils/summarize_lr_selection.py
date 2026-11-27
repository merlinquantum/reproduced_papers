"""Rebuild the embedded learning-rate selections from raw run artifacts.

Usage (from the repository root)::

    python3 papers/photonic_trainability/utils/summarize_lr_selection.py \\
        <SWEEP_ID>=<RUN_DIR> [<SWEEP_ID>=<RUN_DIR> ...] [--out <CSV>]

For each training run it reads ``learning_rate_selection.csv``, the run's
``config_snapshot.json`` (planned candidates) and ``metrics.json`` (rate the
run used), then:

* compares the planned and observed candidate sets (exact equality required
  for ``COMPLETE SWEEP``);
* rejects non-finite or duplicate candidates;
* re-applies the predeclared rule (lowest finite final exact loss on the
  selection seed, ties within 1e-12 to the smaller rate) at full precision;
* checks that the re-derived rate equals the rate the run used.

It prints one table row per candidate and writes them to ``--out``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

TIE_TOLERANCE = 1e-12


def summarize(
    sweep_id: str, run_dir: Path
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Return the candidate rows and the verdict for one embedded selection."""
    config = json.loads((run_dir / "config_snapshot.json").read_text(encoding="utf-8"))
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    status = json.loads((run_dir / "run_status.json").read_text(encoding="utf-8"))
    with (run_dir / "learning_rate_selection.csv").open(encoding="utf-8") as handle:
        observed = list(csv.DictReader(handle))
    planned = [float(value) for value in config["learning_rate_grid"]]
    rates = [float(row["learning_rate"]) for row in observed]
    losses = [float(row["final_exact_loss"]) for row in observed]
    finite = [(rate, loss) for rate, loss in zip(rates, losses) if math.isfinite(loss)]
    best = min(loss for _, loss in finite)
    selected = min(rate for rate, loss in finite if loss - best <= TIE_TOLERANCE)
    complete = (
        sorted(rates) == sorted(planned)
        and len(set(rates)) == len(rates)
        and len(finite) == len(rates)
        and status["status"] == "COMPLETED"
    )
    rows = [
        {
            "sweep_id": sweep_id,
            "candidate_learning_rate": rate,
            "selection_seed": int(config["selection_seed"]),
            "repetitions": 1,
            "final_exact_loss": loss,
            "status": "COMPLETED" if math.isfinite(loss) else "INVALID",
            "selected": rate == selected,
            "run_dir": str(run_dir),
        }
        for rate, loss in zip(rates, losses)
    ]
    verdict = {
        "sweep_id": sweep_id,
        "planned_candidates": planned,
        "observed_candidates": rates,
        "classification": "COMPLETE SWEEP" if complete else "PARTIAL SWEEP",
        "selected_learning_rate": selected,
        "run_used_learning_rate": float(metrics["learning_rate"]),
        "selection_matches_run": selected == float(metrics["learning_rate"]),
        "selected_at_grid_edge": selected in (min(planned), max(planned)),
        "code": status["code"],
    }
    return rows, verdict


def main(argv: list[str] | None = None) -> int:
    """Summarize every given selection; non-zero exit if any check fails."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("selections", nargs="+", help="SWEEP_ID=RUN_DIR pairs")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    all_rows, failures = [], 0
    for item in args.selections:
        sweep_id, run_dir = item.split("=", 1)
        rows, verdict = summarize(sweep_id, Path(run_dir))
        all_rows.extend(rows)
        for row in rows:
            print(
                f"{sweep_id}\tlr={row['candidate_learning_rate']:g}\t"
                f"final_exact_loss={row['final_exact_loss']!r}\t{row['status']}\t"
                f"selected={row['selected']}"
            )
        print(json.dumps(verdict))
        if (
            verdict["classification"] != "COMPLETE SWEEP"
            or not verdict["selection_matches_run"]
        ):
            failures += 1
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
            writer.writeheader()
            writer.writerows(all_rows)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
