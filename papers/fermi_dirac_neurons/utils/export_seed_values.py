"""Export per-seed final metrics of the main sweeps to ``results/per_seed_values.csv``.

Usage (from the repository root)::

    python3 papers/fermi_dirac_neurons/utils/export_seed_values.py \
        papers/fermi_dirac_neurons/outdir/sweeps

Raw run directories (``outdir/``) are not committed; the notebook reads this
compact, durable table instead. One row per (sweep, config, model, seed,
metric): final training loss logged before the last update, test (held-out
"validation" split) MSE or accuracy, the model's learning rate, and whether it
diverged. Only COMPLETED runs of the listed sweeps are exported, and every
planned run must be COMPLETED.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
SWEEPS = (
    "paper_seeds_v1",
    "e8_classical_data_v1",
    "p4_learning_v1",
    "p5_final_v1",
    "p5_photonic_final_v1",
)


def main() -> None:
    """Write the per-seed table."""
    sweeps_dir = Path(sys.argv[1])
    rows = []
    for sweep in SWEEPS:
        status = json.loads((sweeps_dir / sweep / "sweep_status.json").read_text())
        incomplete = [
            run["run_id"] for run in status["runs"] if run["status"] != "COMPLETED"
        ]
        if incomplete:
            raise ValueError(f"{sweep}: runs not COMPLETED: {incomplete[:5]}")
        for run in status["runs"]:
            metrics = json.loads((sweeps_dir / sweep / run["metrics_path"]).read_text())
            for model, record in metrics["models"].items():
                values = {"train_loss": record["last_logged_train_loss"]}
                for name, value in record["final"]["validation"].items():
                    values[f"test_{name}"] = value
                for metric, value in values.items():
                    rows.append(
                        {
                            "sweep": sweep,
                            "config": run["config"],
                            "model": model,
                            "seed": run["seed"],
                            "metric": metric,
                            "value": repr(float(value)),
                            "learning_rate": record.get("learning_rate", ""),
                            "diverged": record.get("diverged_at_epoch") is not None,
                            "n_parameters": record["n_parameters"],
                        }
                    )
    output = PROJECT_DIR / "results" / "per_seed_values.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(output, len(rows))


if __name__ == "__main__":
    main()
