"""Compare original-protocol runs with the paper's values and the released CSVs.

Usage (from the repository root)::

    python3 papers/fermi_dirac_neurons/utils/compare_reference.py \
        papers/fermi_dirac_neurons/outdir/originals_seed0 [--out results/originals_seed0_comparison]

``RUNS_ROOT`` holds one directory per configuration, ``<config>/run_*/``, each
with ``metrics.json``. Writes
``<out>.json`` and ``<out>.csv`` with one row per (configuration, model):

* Fig. 7: our loss logged at the last epoch vs the released CSV's last row
  (both are the loss before the final update).
* Fig. 8: our logistic loss at the last epoch vs the CSV, our validation
  accuracy under both prediction rules vs Table II and vs the CSV's last
  logged accuracy (D4, D7).
* Fig. 9: our final training loss vs the pixel reading of the figure
  (``lib.paper_values``), and the log10 ratio.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR.parent))
sys.path.insert(0, str(PROJECT_DIR.parents[1]))

from fermi_dirac_neurons.lib.data import load_reference_curve  # noqa: E402
from fermi_dirac_neurons.lib.paper_values import (  # noqa: E402
    FIG7_REFERENCE_CSV,
    FIG8_REFERENCE_CSV,
    FIG9_FINAL_TRAIN_LOSS_READING,
    TABLE_II_VALIDATION_ACCURACY_PCT,
)


def default_data_root() -> Path:
    """Resolve the shared data root exactly as the runtime does for this paper."""
    from runtime_lib.data_paths import resolve_data_root

    defaults = json.loads((PROJECT_DIR / "configs" / "defaults.json").read_text())
    return resolve_data_root(defaults["data_root"], PROJECT_DIR)


CONFIG_PATTERN = re.compile(
    r"(fig[789])_([a-z]+)_(\d)q_(original|papertext|classical|sens_eta03|sens_zeta1)"
)


def _single_run_dir(config_dir: Path) -> Path:
    run_dirs = sorted(config_dir.glob("run_*"))
    if len(run_dirs) != 1:
        raise ValueError(
            f"Expected exactly one run in {config_dir}, found {len(run_dirs)}"
        )
    return run_dirs[0]


def _last_finite(values) -> float:
    finite = [float(value) for value in values if math.isfinite(value)]
    return finite[-1]


def compare_run(config_name: str, run_dir: Path, data_root: Path) -> list[dict]:
    """Return comparison rows for one run directory."""
    match = CONFIG_PATTERN.fullmatch(config_name)
    if match is None:
        raise ValueError(f"Unrecognised configuration name {config_name!r}")
    figure, family, n_text, variant = match.groups()
    n_qubits = int(n_text)
    config = {"data_root": str(data_root)}
    metrics = json.loads((run_dir / "metrics.json").read_text())
    rows = []
    for model, record in sorted(metrics["models"].items()):
        is_quantum = model.startswith("quantum")
        row = {
            "config": config_name,
            "figure": figure,
            "variant": variant,
            "n_qubits": n_qubits,
            "seed": metrics["seed"],
            "model": model,
            "n_parameters": record["n_parameters"],
            "train_loss_last_epoch": record["last_logged_train_loss"],
            "run_dir": str(run_dir),
        }
        if figure == "fig7":
            reference = load_reference_curve(
                config, FIG7_REFERENCE_CSV[(family, n_qubits)]
            )
            paper = reference["Quantum_Loss" if is_quantum else "Classical_Loss"][-1]
            row.update(
                paper_train_loss=float(paper), paper_source="released CSV, last epoch"
            )
        elif figure == "fig8":
            reference = load_reference_curve(config, FIG8_REFERENCE_CSV[n_qubits])
            prefix = "Quantum" if is_quantum else "Classical"
            validation = record["final"]["validation"]
            row.update(
                paper_train_loss=float(reference[f"{prefix}_Loss"][-1]),
                paper_source="released CSV, last epoch; Table II",
                validation_accuracy_linear_pct=100 * validation["accuracy_linear"],
                validation_accuracy_activation_pct=100
                * validation["accuracy_activation"],
                csv_last_logged_accuracy_pct=_last_finite(
                    reference[f"{prefix}_Accuracy_Pct"]
                ),
                table_ii_accuracy_pct=TABLE_II_VALIDATION_ACCURACY_PCT.get(
                    model, {}
                ).get(n_qubits),
            )
        else:
            reading = FIG9_FINAL_TRAIN_LOSS_READING[(family, n_qubits)].get(model)
            row.update(paper_train_loss=reading, paper_source="Fig. 9 pixel reading")
        ours, paper = row["train_loss_last_epoch"], row.get("paper_train_loss")
        row["log10_ratio_to_paper"] = (
            math.log10(ours / paper) if paper and ours > 0 else None
        )
        rows.append(row)
    return rows


def main() -> None:
    """Compare every configuration directory under the runs root."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs_root", type=Path)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Shared data root (default: as resolved by the runtime)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output path stem (.json and .csv are added)",
    )
    args = parser.parse_args()
    rows = []
    for config_dir in sorted(
        path for path in args.runs_root.iterdir() if path.is_dir()
    ):
        rows += compare_run(
            config_dir.name,
            _single_run_dir(config_dir),
            args.data_root or default_data_root(),
        )
    stem = args.out or args.runs_root / "comparison"
    stem.parent.mkdir(parents=True, exist_ok=True)
    stem.with_suffix(".json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8"
    )
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with stem.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(stem.with_suffix(".json"))


if __name__ == "__main__":
    main()
