"""Fourier fingerprints of photonic circuits in one and two input dimensions.

The model encodes the input on four active optical modes plus one reference
mode, evaluates the circuit on a regular grid over the input domain, and takes
the probability that mode 0 contains at least one photon as the measured
signal. The Fourier coefficients of that signal are collected over many random
parameter settings, and the Fourier correlation score (FCC) is the mean
absolute off-diagonal Pearson correlation between the active frequencies.

The one- and two-dimensional experiments differ only in the scale factors, the
photon number, how the input is mapped onto the active modes, and whether the
transform is a 1D or 2D FFT. Everything else, including the four circuit
topologies, is shared.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import merlin as ml
import numpy as np
import perceval as pcvl
import torch
import torch.nn as nn
from merlin.builder import CircuitBuilder

# Scale factors applied to the input before angle encoding, per dimension.
SCALE_FACTORS = {
    1: {
        "linear": [1.0, 2.0, 3.0, 4.0],
        "exponential": [1.0, 2.0, 4.0, 8.0],
        "balanced": [-2.0, -1.0, 1.0, 2.0],
    },
    2: {
        "linear": [1.0, 2.0, 1.0, 2.0],
        "exponential": [1.0, 3.0, 1.0, 3.0],
        "balanced": [-1.0, 1.0, -1.0, 1.0],
    },
}

# Photon number per dimension. The 2D model needs a third photon to carry the
# second coordinate through the interference pattern.
N_PHOTONS = {1: 2, 2: 3}

# In 2D, modes 0 and 1 carry the first coordinate and modes 2 and 3 the second.
COORD_INDICES_2D = [0, 0, 1, 1]

AVAILABLE_CIRCUITS = {
    "circuit_0": 0,
    "circuit_1": 1,
    "circuit_2": 2,
    "circuit_3": 3,
}

# Sampling parameters. One config file per experiment is the convention here,
# so these stay fixed rather than being exposed as config knobs.
N_SAMPLES = 200
N_POINTS_1D = 64
RES_GRID_2D = 32
N_OMEGA_2D = 3
VARIANCE_THRESHOLD = 1e-8
MAX_DISPLAYED_FREQUENCIES_2D = 25


# =====================================================================
# 1. MODEL
# =====================================================================
class PhotonicSpectralModel(nn.Module):
    """Photonic circuit whose output spectrum is the object of study.

    Parameters
    ----------
    dimension:
        1 or 2 input coordinates.
    encoding:
        Name of the scale-factor set, one of ``SCALE_FACTORS[dimension]``.
    circuit_index:
        Which of the four circuit topologies to build.
    """

    def __init__(self, dimension=1, encoding="linear", circuit_index=0):
        super().__init__()
        if dimension not in SCALE_FACTORS:
            raise ValueError(
                f"Invalid dimension: {dimension!r}. Choose from "
                f"{sorted(SCALE_FACTORS)}"
            )
        scales_for_dim = SCALE_FACTORS[dimension]
        if encoding not in scales_for_dim:
            valid_encodings = ", ".join(scales_for_dim)
            raise ValueError(
                f"Invalid encoding: {encoding!r}. Choose from: {valid_encodings}"
            )

        self.dimension = dimension
        self.encoding = encoding
        self.scale_factors = torch.tensor(scales_for_dim[encoding], dtype=torch.float32)
        self.n_photons = N_PHOTONS[dimension]
        self.n_active_modes = len(scales_for_dim[encoding])
        self.n_modes = self.n_active_modes + 1  # active modes + 1 reference mode
        self.measurement_strategy = ml.MeasurementStrategy.probs(
            computation_space=ml.ComputationSpace.FOCK
        )

        self.circuit_index = circuit_index
        self.n_encoding_layers = self._get_num_encoding_layers(circuit_index)
        self.quantum_layer = self._build_quantum_layer(circuit_index)

        # Fock states with at least one photon in mode 0. Derived from the
        # layer's own basis ordering so it stays correct for any mode and
        # photon count, rather than assuming a fixed column range.
        mask = [state[0] >= 1 for state in self.quantum_layer.output_keys]
        self.register_buffer("mode0_mask", torch.tensor(mask, dtype=torch.bool))

    def _build_quantum_layer(self, circuit_index):
        builder = self._build_circuit(circuit_index)
        return ml.QuantumLayer(
            input_size=self.n_active_modes * self.n_encoding_layers,
            builder=builder,
            n_photons=self.n_photons,
            measurement_strategy=self.measurement_strategy,
            dtype=torch.float32,
        )

    def _get_num_encoding_layers(self, circuit_index):
        encoding_layers = {0: 1, 1: 1, 2: 2, 3: 1}
        if circuit_index not in encoding_layers:
            raise ValueError(
                f"Invalid circuit_index: {circuit_index}. "
                f"Choose from {sorted(encoding_layers)}"
            )
        return encoding_layers[circuit_index]

    def _build_circuit(self, circuit_index):
        builder = CircuitBuilder(n_modes=self.n_modes)
        builders = {
            0: self._build_circuit_type_0,
            1: self._build_circuit_type_1,
            2: self._build_circuit_type_2,
            3: self._build_circuit_type_3,
        }
        if circuit_index not in builders:
            raise ValueError(
                f"Invalid circuit_index: {circuit_index}. Choose from {sorted(builders)}"
            )
        builders[circuit_index](builder)
        return builder

    def _add_encoding(self, builder):
        modes = list(range(self.n_active_modes))
        builder.add_angle_encoding(modes=modes, name="data_encoding")

    def _build_circuit_type_0(self, builder):
        # Basic topology: entanglement -> encoding -> entanglement.
        builder.add_entangling_layer(trainable=True, model="mzi", name="init_mzi")
        self._add_encoding(builder)
        builder.add_entangling_layer(trainable=True, model="mzi", name="mid_mzi")

    def _build_circuit_type_1(self, builder):
        # Lighter topology: direct encoding followed by a single entangling layer.
        builder.add_superpositions(targets=(0, 1), trainable_theta=True, name="bs1")
        builder.add_superpositions(targets=(3, 4), trainable_theta=True, name="bs2")
        builder.add_superpositions(targets=(2, 3), trainable_theta=True, name="bs3")
        builder.add_superpositions(targets=(1, 2), trainable_theta=True, name="bs4")
        builder.add_superpositions(targets=(3, 4), trainable_theta=True, name="bs5")
        builder.add_superpositions(targets=(0, 1), trainable_theta=True, name="bs6")
        self._add_encoding(builder)
        builder.add_superpositions(targets=(0, 1), trainable_theta=True, name="bs11")
        builder.add_superpositions(targets=(3, 4), trainable_theta=True, name="bs12")
        builder.add_superpositions(targets=(2, 3), trainable_theta=True, name="bs13")
        builder.add_superpositions(targets=(1, 2), trainable_theta=True, name="bs14")
        builder.add_superpositions(targets=(3, 4), trainable_theta=True, name="bs15")
        builder.add_superpositions(targets=(0, 1), trainable_theta=True, name="bs16")

    def _build_circuit_type_2(self, builder):
        # Deeper topology: two layers before encoding followed by a final mixing layer.
        builder.add_entangling_layer(trainable=True, model="mzi", name="pre_mzi_0")
        self._add_encoding(builder)
        builder.add_entangling_layer(trainable=True, model="mzi", name="pre_mzi_1")
        self._add_encoding(builder)
        builder.add_entangling_layer(trainable=True, model="mzi", name="post_mzi")

    def _build_circuit_type_3(self, builder):
        # Simple naive entanglement: encoding -> entanglement -> encoding.
        builder.add_superpositions(targets=(0, 1), trainable_theta=True, name="bs1")
        builder.add_superpositions(targets=(1, 2), trainable_theta=True, name="bs2")
        builder.add_superpositions(targets=(2, 3), trainable_theta=True, name="bs3")
        builder.add_superpositions(targets=(3, 4), trainable_theta=True, name="bs4")
        self._add_encoding(builder)
        builder.add_superpositions(targets=(3, 4), trainable_theta=True, name="bs5")
        builder.add_superpositions(targets=(2, 3), trainable_theta=True, name="bs6")
        builder.add_superpositions(targets=(1, 2), trainable_theta=True, name="bs7")
        builder.add_superpositions(targets=(0, 1), trainable_theta=True, name="bs8")

    def forward(self, x):
        if self.dimension == 1:
            # Broadcast the scalar input across the active modes.
            encoded_input = x * self.scale_factors
        else:
            # Keep the coordinate-to-mode mapping while applying the scales.
            encoded_input = x[:, COORD_INDICES_2D] * self.scale_factors
        repeated_input = torch.cat([encoded_input] * self.n_encoding_layers, dim=1)
        return self.quantum_layer(repeated_input)

    def mode0_occupancy(self, probs):
        """Probability that output mode 0 contains at least one photon."""
        return probs[:, self.mode0_mask].sum(dim=1)


# =====================================================================
# 2. FOURIER COEFFICIENTS AND PAIRWISE CORRELATIONS
# =====================================================================
def _sample_coefficients(model, n_samples):
    """Yield |c_w| for ``n_samples`` random parameter settings of the model."""
    input_grid, reshape_to = _input_grid(model.dimension)
    coefficients = []

    for _ in range(n_samples):
        with torch.no_grad():
            for param in model.parameters():
                if param.requires_grad:
                    param.data.uniform_(0, 2 * np.pi)
            signal = model.mode0_occupancy(model(input_grid)).numpy()

        if model.dimension == 1:
            spectrum = np.fft.rfft(signal) / signal.size
        else:
            spectrum = np.fft.fftn(signal.reshape(reshape_to)) / signal.size
        coefficients.append(np.abs(spectrum.flatten()))

    return np.array(coefficients)


def _input_grid(dimension):
    """Return the regular evaluation grid and the shape to reshape signals to."""
    if dimension == 1:
        axis = np.linspace(0, 2 * np.pi, N_POINTS_1D, endpoint=False)
        return torch.from_numpy(axis).float().unsqueeze(1), (N_POINTS_1D,)

    axes = [np.linspace(0, 2 * np.pi, RES_GRID_2D, endpoint=False) for _ in range(2)]
    grid_x1, grid_x2 = np.meshgrid(*axes, indexing="ij")
    stacked = np.stack([grid_x1.flatten(), grid_x2.flatten()], axis=-1)
    return torch.from_numpy(stacked).float(), (RES_GRID_2D, RES_GRID_2D)


def _active_frequencies(coefficient_matrix, dimension):
    """Select frequencies that vary across samples, with display labels."""
    variances = np.var(coefficient_matrix, axis=0)
    active = np.where(variances > VARIANCE_THRESHOLD)[0]

    if dimension == 1:
        return active, [f"ω={w}" for w in active]

    # Restrict to Omega_n = {(w1, w2) : |w1| + |w2| <= n_omega}.
    kept, labels = [], []
    for index in active:
        w1, w2 = divmod(int(index), RES_GRID_2D)
        if w1 >= RES_GRID_2D // 2:
            w1 -= RES_GRID_2D
        if w2 >= RES_GRID_2D // 2:
            w2 -= RES_GRID_2D
        if abs(w1) + abs(w2) <= N_OMEGA_2D:
            kept.append(index)
            labels.append(f"({w1},{w2})")
    return np.array(kept, dtype=int), labels


def compute_fingerprint(model, n_samples=N_SAMPLES):
    """Return the correlation fingerprint, FCC score, labels and coefficients."""
    coefficient_matrix = _sample_coefficients(model, n_samples)
    active, labels = _active_frequencies(coefficient_matrix, model.dimension)
    active_coefficients = coefficient_matrix[:, active]

    n_active = len(active)
    if n_active == 0:
        return np.empty((0, 0)), 0.0, labels, active_coefficients
    if n_active == 1:
        return np.ones((1, 1)), 0.0, labels, active_coefficients

    fingerprint = np.atleast_2d(np.corrcoef(active_coefficients, rowvar=False))
    fingerprint = np.nan_to_num(fingerprint, nan=0.0)
    off_diagonal = ~np.eye(n_active, dtype=bool)
    fcc_score = float(np.mean(np.abs(fingerprint[off_diagonal])))
    return fingerprint, fcc_score, labels, active_coefficients


# =====================================================================
# 3. DISPLAY
# =====================================================================
def plot_fingerprint(fingerprint, labels, fcc_score, ax, title=None, max_display=None):
    """Draw the lower-triangular correlation matrix on ``ax``."""
    n_display = len(labels)
    if max_display is not None:
        n_display = min(max_display, n_display)

    if n_display == 0:
        ax.text(0.5, 0.5, "No active frequency", ha="center", va="center")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        return

    shown = np.abs(fingerprint[:n_display, :n_display])
    upper = np.triu(np.ones((n_display, n_display), dtype=bool))
    image = ax.imshow(np.ma.masked_array(shown, upper), cmap="plasma_r", vmin=0, vmax=1)
    ax.figure.colorbar(image, ax=ax)
    ax.set_xticks(range(n_display), labels=labels[:n_display], rotation=60, ha="right")
    ax.set_yticks(range(n_display), labels=labels[:n_display])
    ax.set_title(
        f"{title or 'Fourier Fingerprint'}\nFCC = {fcc_score:.4f}", fontweight="bold"
    )


# =====================================================================
# 4. ENTRY POINT
# =====================================================================
def main(
    dimension=1,
    circuits=None,
    encoding="linear",
    n_samples=N_SAMPLES,
    debug=False,
    rundir: Path | None = None,
    name: str | None = None,
):
    """Evaluate the requested circuits and save or show their fingerprints."""
    if dimension not in SCALE_FACTORS:
        raise ValueError(
            f"Unsupported Fourier fingerprint dimension: {dimension!r}. "
            f"Choose from {sorted(SCALE_FACTORS)}"
        )
    if circuits is None:
        circuits = ["circuit_2"]
    if not isinstance(circuits, list):
        raise TypeError("circuits must be a list of circuit names.")
    if not circuits:
        raise ValueError("circuits must contain at least one circuit.")
    if encoding not in SCALE_FACTORS[dimension]:
        valid_encodings = ", ".join(SCALE_FACTORS[dimension])
        raise ValueError(
            f"Invalid encoding: {encoding!r}. Choose from: {valid_encodings}"
        )

    unknown = [
        name_
        for name_ in circuits
        if not isinstance(name_, str) or name_ not in AVAILABLE_CIRCUITS
    ]
    if unknown:
        valid_names = ", ".join(sorted(AVAILABLE_CIRCUITS))
        raise ValueError(f"Invalid circuits: {unknown}. Choose from: {valid_names}")

    figure, axes = plt.subplots(
        1, len(circuits), figsize=(8 * len(circuits), 7), squeeze=False
    )
    results = {}
    models = {}
    max_display = MAX_DISPLAYED_FREQUENCIES_2D if dimension == 2 else None

    for circuit_name, ax in zip(circuits, axes[0]):
        model = PhotonicSpectralModel(
            dimension=dimension,
            encoding=encoding,
            circuit_index=AVAILABLE_CIRCUITS[circuit_name],
        )
        models[circuit_name] = model

        fingerprint, fcc_score, labels, coefficients = compute_fingerprint(
            model, n_samples=n_samples
        )
        results[circuit_name] = (fingerprint, fcc_score, labels, coefficients)

        print("\n" + "=" * 55)
        print(f" ANALYSIS RESULTS: {circuit_name} ({dimension}D)")
        print("=" * 55)
        print(
            f"Total optical modes        : {model.n_modes} "
            f"({model.n_active_modes} encoded + 1 reference)"
        )
        print(f"Photons                    : {model.n_photons}")
        print(f"Active frequencies         : {len(labels)}")
        print(f"FCC score (correlation)    : {fcc_score:.5f}")
        print("=" * 55)

        plot_fingerprint(
            fingerprint,
            labels,
            fcc_score,
            ax=ax,
            title=f"{circuit_name} - {encoding}",
            max_display=max_display,
        )

    figure.tight_layout()
    if rundir is None:
        plt.show()
    else:
        rundir = Path(rundir)
        rundir.mkdir(parents=True, exist_ok=True)
        filename = name or f"fingerprints_{dimension}d_{encoding}.png"
        if not filename.lower().endswith(".png"):
            filename += ".png"
        figure_path = rundir / filename
        figure.savefig(figure_path, bbox_inches="tight")
        print(f"Matrix figure saved to: {figure_path.resolve()}")
        plt.close(figure)

    if debug:
        for circuit_name in circuits:
            print(f"\n--- Circuit: {circuit_name} ---")
            pcvl.pdisplay(models[circuit_name].quantum_layer.circuit)

    return results


if __name__ == "__main__":
    main(
        dimension=1,
        circuits=["circuit_0", "circuit_1", "circuit_2", "circuit_3"],
        encoding="balanced",
        name="Fig 2(a) - Test configuration",
        rundir=Path(__file__).resolve().parent.parent / "outdir",
    )
