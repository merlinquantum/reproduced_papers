"""E8 analysis (claim C8): does the quantum advantage need quantum training data?

Usage (from the repository root)::

    python3 papers/fermi_dirac_neurons/utils/analyze_e8.py \
        papers/fermi_dirac_neurons/outdir/sweeps/e8_classical_data_v1

For every E8 config (task x training data) the quantum model is compared with
the commuting model on the declared metrics, paired by seed, with the same
two-sided Wilcoxon rule and tolerances as ``utils/sweep_statistics.py``.
Writes ``results/e8_classical_data_summary.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR.parent))
sys.path.insert(0, str(PROJECT_DIR.parents[1]))

from fermi_dirac_neurons.utils.sweep_statistics import (  # noqa: E402
    load_values,
    paired_comparison,
)

MODEL_PAIRS = {
    "fig7": ("quantum_tfim", "classical_im", ["train_loss", "validation_mse"]),
    "fig8": (
        "quantum_heisenberg",
        "classical_fcim",
        ["validation_accuracy_linear", "validation_accuracy_activation", "train_loss"],
    ),
}


def main() -> None:
    """Write the paired comparisons for every E8 config."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sweep_dir", type=Path)
    args = parser.parse_args()
    values, seeds = load_values(args.sweep_dir)
    configs = sorted({config for config, _, _ in values})
    rows = []
    for config in configs:
        figure = config.split("_")[1]
        quantum, commuting, metrics = MODEL_PAIRS[figure]
        for metric in metrics:
            rows.append(
                {
                    "config": config,
                    **paired_comparison(
                        values, seeds, (config, quantum), (config, commuting), metric
                    ),
                }
            )
    output = PROJECT_DIR / "results" / "e8_classical_data_summary.json"
    output.write_text(
        json.dumps(
            {"sweep_dir": str(args.sweep_dir), "seeds": seeds, "comparisons": rows},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
