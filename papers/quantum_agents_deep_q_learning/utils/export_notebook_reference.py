"""Export the compact per-agent records the notebook needs from ``outdir/``.

``outdir/`` is disposable runtime output; ``results/`` is the durable place.
This script reads the structured artifacts that ``results/phase5_comparison.json``,
``results/merlin_comparison.json`` and the batch folders point to and writes
``results/notebook_reference.json`` with

* ``solved_at``: per configuration, the 0-based solve episode of each agent
  (``null`` if unsolved) for the Phase 5 rows, every MerLin run, its qubit
  counterparts and the tanh diagnostic batch;
* ``cartpole_full_scores_first300``: the first 300 episode scores of the 10
  agents of the code-faithful 5-layer Cart Pole run (seeds 0-9);
* ``frozenlake_l15_mae44_mean``: the mean error to Q* (44 non-terminal pairs)
  at selected time steps of the code-faithful 15-layer Frozen Lake run;
* provenance (source run directories).

Example (from the paper folder)::

    python utils/export_notebook_reference.py --output results/notebook_reference.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
CHECKPOINT_STEPS = [0, 250, 500, 1000, 2000, 3000]


def agent_records(run_dir: Path) -> dict:
    metrics = json.loads((run_dir / "metrics.json").read_text())
    agents = sorted(metrics["agents"], key=lambda agent: agent["seed"])
    return {
        "seeds": [agent["seed"] for agent in agents],
        "solved_at": [
            agent["solved_at_episode_index"] if agent["solved"] else None
            for agent in agents
        ],
        "wall_clock_seconds": metrics["wall_clock_seconds"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    results = PROJECT_DIR / "results"
    phase5 = json.loads((results / "phase5_comparison.json").read_text())["rows"]
    merlin = json.loads((results / "merlin_comparison.json").read_text())["pairs"]
    records: dict[str, dict] = {}
    for row in phase5.values():
        records[row["config"]] = {
            "run_dir": row["run_dir"],
            **agent_records(PROJECT_DIR / row["run_dir"]),
        }
    for name, pair in merlin.items():
        records[name] = {
            "run_dir": pair["merlin"]["run_dir"],
            **agent_records(PROJECT_DIR / pair["merlin"]["run_dir"]),
        }
        qubit_name = name.split("_merlin")[0] + "_codefaithful"
        records.setdefault(
            qubit_name,
            {
                "run_dir": pair["qubit"]["run_dir"],
                **agent_records(PROJECT_DIR / pair["qubit"]["run_dir"]),
            },
        )
    diag_batch = PROJECT_DIR / "outdir" / "batches" / "diag_tanh"
    for entry in json.loads((diag_batch / "sweep_status.json").read_text())["runs"]:
        stem = Path(entry["config_file"]).stem
        records[stem] = {
            "run_dir": str(Path("outdir/batches/diag_tanh") / entry["run_dir"]),
            **agent_records(diag_batch / entry["run_dir"]),
        }

    cartpole_dir = (
        PROJECT_DIR / merlin["e2e3_cartpole_optimal_full_merlin"]["qubit"]["run_dir"]
    )
    scores = np.load(cartpole_dir / "scores.npz")["scores"]
    flat_dir = (
        PROJECT_DIR
        / json.loads((results / "statistics_codefaithful.json").read_text())["configs"][
            "e1_frozenlake_l15_original"
        ]["run_dir"]
    )
    mae = np.load(flat_dir / "scores.npz")["q_star_mae_nonterminal_44"]
    mae_mean = []
    for step in CHECKPOINT_STEPS:
        values = [row[np.flatnonzero(np.isfinite(row[: step + 1]))[-1]] for row in mae]
        mae_mean.append(float(np.mean(values)))

    output = {
        "solved_at": records,
        "cartpole_full_scores_first300": {
            "source": str(cartpole_dir.relative_to(PROJECT_DIR)),
            "seeds": list(range(10)),
            "scores": scores[:, :300].tolist(),
        },
        "frozenlake_l15_mae44_mean": {
            "source": str(flat_dir.relative_to(PROJECT_DIR)),
            "steps": CHECKPOINT_STEPS,
            "mean": mae_mean,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output))
    print(f"{len(records)} configurations exported to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
