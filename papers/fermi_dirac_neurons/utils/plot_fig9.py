"""Fig. 9 reproduction: tanh vs linear vs commuting neurons, 10 seeds vs pixel readings.

Usage (from the repository root)::

    python3 papers/fermi_dirac_neurons/utils/plot_fig9.py \
        papers/fermi_dirac_neurons/outdir/sweeps/paper_seeds_v1 [--variant original]

``--variant`` selects the protocol: ``original`` (Fig. 7 settings, the
configured default), ``sens_eta03`` or ``sens_zeta1`` (the pre-declared
sensitivity variants). Each panel shows the tanh, linear and commuting
neurons (median and min-max band over seeds), B2 from the ``_classical``
config (original variant only, same data), and the paper's value at epoch
10 000 read from the figure by pixel position (hollow diamonds). Fig. 9 has
no released code: settings are inferred (LOG.md D9).
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

from fermi_dirac_neurons.lib.paper_values import FIG9_FINAL_TRAIN_LOSS_READING  # noqa: E402
from fermi_dirac_neurons.lib.plotting import (  # noqa: E402
    COLOURS,
    SweepReader,
    apply_style,
    band,
    finish,
    model_role,
    save,
)

PANELS = (("tfim", 2, "a"), ("tfim", 3, "b"), ("heis", 2, "c"), ("heis", 3, "d"))
VARIANT_TEXT = {
    "original": "Fig. 7 settings (η = 0.1, ζ ~ U[−2, 2], T = 2)",
    "sens_eta03": "sensitivity variant η = 0.3",
    "sens_zeta1": "sensitivity variant ζ ~ U[−1, 1]",
}
LABELS = {
    "quantum": "tanh neuron Tr[tanh(H(θ)/T)ρ]",
    "linear": "quantum linear Tr[H(θ)ρ]",
    "commuting": "commuting neuron (IM / FCIM)",
}


def main() -> None:
    """Draw the four Fig. 9 panels for one protocol variant."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sweep_dir", type=Path)
    parser.add_argument("--variant", default="original", choices=sorted(VARIANT_TEXT))
    args = parser.parse_args()
    reader = SweepReader(args.sweep_dir)
    apply_style()
    figure, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for axis, (family, n_qubits, panel) in zip(axes.flat, PANELS):
        stem = f"fig9_{family}_{n_qubits}q"
        readings = FIG9_FINAL_TRAIN_LOSS_READING[(family, n_qubits)]
        for model in (
            f"quantum_{family}_tanh",
            f"quantum_{family}_linear",
            "classical_ising" if family == "tfim" else "classical_fcim",
        ):
            role = model_role(model)
            epochs, curves = reader.histories(f"{stem}_{args.variant}", model)
            band(axis, epochs, curves, role, LABELS[role])
            axis.plot(
                [epochs[-1]],
                [readings[model]],
                linestyle="none",
                marker="D",
                markersize=8,
                markerfacecolor="none",
                markeredgewidth=1.5,
                color=COLOURS[role],
                label=f"paper at epoch 10 000: {LABELS[role].split(' Tr')[0]}",
            )
        if args.variant == "original":
            epochs, curves = reader.histories(f"{stem}_classical", "b2_pauli_features")
            band(
                axis,
                epochs,
                curves,
                "b2",
                "B2: classical neuron on same Pauli features (paper η, untuned)",
            )
        axis.set_yscale("log")
        axis.set_title(
            f"({panel}) {'TFIM' if family == 'tfim' else 'Heisenberg'}, {n_qubits} qubits"
        )
        axis.set_xlabel("epoch")
        axis.set_ylabel("training squared loss")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    order = sorted(
        range(len(labels)), key=lambda index: labels[index].startswith("paper")
    )
    finish(
        figure,
        [handles[index] for index in order],
        [labels[index] for index in order],
        f"Fig. 9 reproduction ({VARIANT_TEXT[args.variant]}): median over 10 seeds (band: min–max)",
        f"paper-accurate seeds {reader.seeds[0]}–{reader.seeds[-1]} (sweep {reader.sweep_id}), but Fig. 9 settings are inferred: no released code (D9); paper values are pixel readings of the figure",
    )
    for path in save(figure, f"fig9_reproduction_{args.variant}", args.sweep_dir):
        print(path)


if __name__ == "__main__":
    main()
