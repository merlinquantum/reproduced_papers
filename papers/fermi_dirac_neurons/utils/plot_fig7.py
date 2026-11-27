"""Fig. 7 reproduction: squared-loss training curves, 10 seeds vs the released CSVs.

Usage (from the repository root)::

    python3 papers/fermi_dirac_neurons/utils/plot_fig7.py \
        papers/fermi_dirac_neurons/outdir/sweeps/paper_seeds_v1

Panels: tanh 2q, tanh 7q, softplus 2q, softplus 7q under the original
(notebook) protocol. Each panel shows the quantum TFIM neuron and the
commuting IM neuron (``_original`` config), the fair baselines B2 and B3
(``_classical`` config, same seeds and data), and the authors' released curves
(dashed).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR.parent))
sys.path.insert(0, str(PROJECT_DIR.parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from fermi_dirac_neurons.lib.data import load_reference_curve  # noqa: E402
from fermi_dirac_neurons.lib.paper_values import FIG7_REFERENCE_CSV  # noqa: E402
from fermi_dirac_neurons.lib.plotting import (  # noqa: E402
    COLOURS,
    ROLE_LABELS,
    SweepReader,
    apply_style,
    band,
    finish,
    model_role,
    save,
)
from fermi_dirac_neurons.utils.compare_reference import default_data_root  # noqa: E402

PANELS = (("tanh", 2), ("tanh", 7), ("softplus", 2), ("softplus", 7))


def main() -> None:
    """Draw the four Fig. 7 panels."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sweep_dir", type=Path)
    args = parser.parse_args()
    reader = SweepReader(args.sweep_dir)
    data_cfg = {"data_root": str(default_data_root())}
    apply_style()
    figure, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for axis, (activation, n_qubits) in zip(axes.flat, PANELS):
        stem = f"fig7_{activation}_{n_qubits}q"
        for config, models in (
            (f"{stem}_original", ("quantum_tfim", "classical_im")),
            (f"{stem}_classical", ("b2_pauli_features", "b3_quantum_linear")),
        ):
            for model in models:
                epochs, curves = reader.histories(config, model)
                role = model_role(model)
                band(axis, epochs, curves, role, ROLE_LABELS[role])
        reference = load_reference_curve(
            data_cfg, FIG7_REFERENCE_CSV[(activation, n_qubits)]
        )
        axis.plot(
            reference["Epoch"],
            reference["Quantum_Loss"],
            color=COLOURS["quantum"],
            linestyle="--",
            linewidth=1.2,
            label="paper CSV: quantum",
        )
        axis.plot(
            reference["Epoch"],
            reference["Classical_Loss"],
            color=COLOURS["commuting"],
            linestyle="--",
            linewidth=1.2,
            label="paper CSV: commuting",
        )
        axis.set_yscale("log")
        axis.set_title(f"{activation}, {n_qubits} qubits (TFIM target)")
        axis.set_xlabel("epoch")
        axis.set_ylabel("training squared loss")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    finish(
        figure,
        handles,
        labels,
        "Fig. 7 reproduction: median over 10 seeds (band: min–max), authors' released curves dashed",
        f"paper-accurate (original notebook protocol, seeds {reader.seeds[0]}–{reader.seeds[-1]}, sweep {reader.sweep_id}); softplus target follows the notebook (D1, D2)",
    )
    for path in save(figure, "fig7_reproduction", args.sweep_dir):
        print(path)


if __name__ == "__main__":
    main()
