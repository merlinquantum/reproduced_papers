from pathlib import Path

import matplotlib.pyplot as plt
import merlin as ml
import numpy as np
import perceval as pcvl
import torch
import torch.nn as nn
from merlin.builder import CircuitBuilder

# =====================================================================
# 0. CONFIGURATION
# =====================================================================
FACTEURS_ECHELLE_DISPONIBLES = {
    "linear": [1.0, 2.0, 1.0, 2.0],
    "exponential": [1.0, 3.0, 1.0, 3.0],
    "balanced": [-1.0, 1.0, -1.0, 1.0],
}

# =====================================================================
# 1. MERLIN 2D MODEL
# =====================================================================
class PhotonicSpectralModel2D(nn.Module):
    def __init__(
        self,
        facteur_echelle="linear",
        n_photons=3,
        circuit_index=0,
    ):
        super().__init__()
        if facteur_echelle not in FACTEURS_ECHELLE_DISPONIBLES:
            valid_scales = ", ".join(FACTEURS_ECHELLE_DISPONIBLES)
            raise ValueError(
                f"Invalid scale factor: {facteur_echelle!r}. Choose from: {valid_scales}"
            )

        self.facteur_echelle = facteur_echelle
        facteurs = FACTEURS_ECHELLE_DISPONIBLES[facteur_echelle]
        self.facteurs = torch.tensor(facteurs, dtype=torch.float32)
        self.indices_coord = [0, 0, 1, 1]  # Modes 0,1 -> x1; modes 2,3 -> x2.

        self.n_photons = n_photons
        self.n_actifs = 4
        self.n_modes = self.n_actifs + 1  # + 1 reference mode = 5 optical modes
        self.measurement_strategy = ml.MeasurementStrategy.probs(
            computation_space=ml.ComputationSpace.FOCK
        )

        self.circuit_index = circuit_index
        self.n_encoding_layers = self._get_num_encoding_layers(circuit_index)
        self.quantum_layer = self._build_quantum_layer(circuit_index)

    def _build_quantum_layer(self, circuit_index):
        builder = self._build_circuit(circuit_index)
        return ml.QuantumLayer(
            input_size=self.n_actifs * self.n_encoding_layers,
            builder=builder,
            n_photons=self.n_photons,
            measurement_strategy=self.measurement_strategy,
            dtype=torch.float32,
        )

    def _get_num_encoding_layers(self, circuit_index):
        encoding_layers = {
            0: 1,
            1: 1,
            2: 2,
            3: 1,
        }
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
        modes = list(range(self.n_actifs))
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
           builder.add_superpositions(targets=(3, 4), trainable_theta=True, name="bs1")
           builder.add_superpositions(targets=(2, 3), trainable_theta=True, name="bs2")
           builder.add_superpositions(targets=(1, 2), trainable_theta=True, name="bs3")
           builder.add_superpositions(targets=(0, 1), trainable_theta=True, name="bs4")

    def configure_circuit(self, circuit_index=None):
        """
        Dynamically reconfigure the circuit topology.
        """
        if circuit_index is None:
            circuit_index = self.circuit_index

        self.circuit_index = circuit_index
        self.n_encoding_layers = self._get_num_encoding_layers(circuit_index)
        self.quantum_layer = self._build_quantum_layer(circuit_index)

    def forward(self, x_2d):
        # x_2d has shape [batch, 2].
        # Keep the coordinate-to-mode mapping while applying the scale factors.
        x_multimode = x_2d[:, self.indices_coord] * self.facteurs
        repeated_input = torch.cat(
            [x_multimode] * self.n_encoding_layers,
            dim=1,
        )
        return self.quantum_layer(repeated_input)

# =====================================================================
# 2. 2D FOURIER COEFFICIENTS AND CORRELATIONS
# =====================================================================
def calculer_empreinte_fourier_2d(
    model,
    M=200,
    res_grid=32,
    n_omega=3,
):
    """
    Evaluate M configurations on a 2D grid (res_grid x res_grid points)
    and extract the vector spectrum w = (w1, w2).
    """
    print(f"--- 1. 2D grid ({res_grid}x{res_grid} = {res_grid**2} points), {M} samples ---")

    # Create the regular grid [0, 2pi) x [0, 2pi).
    axes = [np.linspace(0, 2 * np.pi, res_grid, endpoint=False) for _ in range(2)]
    grid_x1, grid_x2 = np.meshgrid(*axes, indexing='ij')

    # Input tensor [res_grid**2, 2].
    x_grid = torch.from_numpy(np.stack([grid_x1.flatten(), grid_x2.flatten()], axis=-1)).float()
    coefficients_list = []

    for m in range(M):
        with torch.no_grad():
            for param in model.parameters():
                if param.requires_grad:
                    param.data.uniform_(0, 2 * np.pi)

            probs_out = model(x_grid)
            if m == 0:
                print(probs_out.shape)  # Expected shape: (res_grid**2, 15) in Fock space.
            # Columns 0..4 are the Fock states with at least one photon in mode 0.
            signal_y = probs_out[:, :5].sum(dim=1).numpy()

        # Reshape into a 2D image [32, 32] for the FFT.
        signal_2d = signal_y.reshape((res_grid, res_grid))

        # 2D Fourier transform (FFT).
        fft_coeffs = np.fft.fftn(signal_2d) / (res_grid ** 2)

        # Flatten the 2D frequency tensor into a 1D vector of magnitudes |c_w|.
        coefficients_list.append(np.abs(fft_coeffs.flatten()))

    C_matrix = np.array(coefficients_list)

    print("--- 2. Identifying active frequency pairs (w1, w2) ---")

    # Filter active frequencies (non-zero variance), then restrict to Omega_n.
    # Omega_n = {(w1, w2) : |w1| + |w2| <= n_omega}.
    variances = np.var(C_matrix, axis=0)
    indices_actifs = np.where(variances > 1e-8)[0]

    indices_filtres = []
    freqs_labels = []
    for idx in indices_actifs:
        w1 = idx // res_grid
        w2 = idx % res_grid
        # Convert FFT indices to negative frequencies where appropriate.
        if w1 >= res_grid // 2:
            w1 -= res_grid
        if w2 >= res_grid // 2:
            w2 -= res_grid
        if abs(w1) + abs(w2) <= n_omega:
            indices_filtres.append(idx)
            freqs_labels.append(f"({w1},{w2})")

    indices_actifs = np.array(indices_filtres, dtype=int)
    C_actives = C_matrix[:, indices_actifs]

    # Pearson correlation and FCC.
    n_actives = len(indices_actifs)
    if n_actives == 0:
        fingerprint = np.empty((0, 0))
        score_fcc = 0.0
    elif n_actives == 1:
        fingerprint = np.ones((1, 1))
        score_fcc = 0.0
    else:
        fingerprint = np.corrcoef(C_actives, rowvar=False)
        fingerprint = np.atleast_2d(np.nan_to_num(fingerprint, nan=0.0))
        masque_hors_diag = ~np.eye(n_actives, dtype=bool)
        score_fcc = np.mean(np.abs(fingerprint[masque_hors_diag]))

    return fingerprint, score_fcc, freqs_labels, C_actives

# =====================================================================
# 3. DISPLAY AND PROGRAM ENTRY POINT
# =====================================================================
CIRCUITS_DISPONIBLES = {
    "circuit_0": 0,
    "circuit_1": 1,
    "circuit_2": 2,
    "circuit_3": 3,
}


def afficher_fingerprint_physique_2d(
    fingerprint,
    freqs,
    fcc_val,
    ax,
    nom_circuit=None,
):
    """Display the lower triangular 2D correlation matrix."""
    n_aff = min(25, len(freqs))
    if n_aff == 0:
        ax.text(0.5, 0.5, "No active frequency", ha="center", va="center")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        return

    masque = np.triu(np.ones((n_aff, n_aff), dtype=bool))
    matrice_affichee = np.ma.masked_array(
        np.abs(fingerprint[:n_aff, :n_aff]),
        masque,
    )
    im = ax.imshow(matrice_affichee, cmap="plasma_r", vmin=0, vmax=1)
    ax.set_xticks(
        ticks=range(n_aff),
        labels=freqs[:n_aff],
        rotation=60,
        ha="right",
    )
    ax.set_yticks(ticks=range(n_aff), labels=freqs[:n_aff])
    ax.figure.colorbar(im, ax=ax, label="Pearson correlation |r|")
    title = nom_circuit or "Fourier Fingerprint - 2D model"
    ax.set_title(f"{title}\nFCC = {fcc_val:.4f}", fontweight="bold")
    ax.set_xlabel("Spatial frequency (ω1', ω2')")
    ax.set_ylabel("Spatial frequency (ω1, ω2)")


def main(
    circuits=None,
    facteur_echelle="linear",
    n_photons=3,
    M=150,
    res_grid=32,
    n_omega=3,
    debug=False,
    rundir: Path | None = None,
    name: str | None = None,
):
    """Run the requested 2D circuits and display or save their fingerprints."""
    if circuits is None:
        circuits = ["circuit_0"]
    if not isinstance(circuits, list):
        raise TypeError("circuits must be a list of circuit names.")
    if not circuits:
        raise ValueError("circuits must contain at least one circuit.")
    if facteur_echelle not in FACTEURS_ECHELLE_DISPONIBLES:
        valid_scales = ", ".join(FACTEURS_ECHELLE_DISPONIBLES)
        raise ValueError(
            f"Invalid scale factor: {facteur_echelle!r}. Choose from: {valid_scales}"
        )

    unknown_circuits = [
        circuit for circuit in circuits
        if not isinstance(circuit, str) or circuit not in CIRCUITS_DISPONIBLES
    ]
    if unknown_circuits:
        valid_circuits = ", ".join(sorted(CIRCUITS_DISPONIBLES))
        raise ValueError(
            f"Invalid circuits: {unknown_circuits}. Choose from: {valid_circuits}"
        )

    figure, axes = plt.subplots(
        1,
        len(circuits),
        figsize=(9 * len(circuits), 8),
        squeeze=False,
    )
    resultats = {}
    modeles = {}

    for nom_circuit, ax_fingerprint in zip(circuits, axes[0]):
        model = PhotonicSpectralModel2D(
            facteur_echelle=facteur_echelle,
            n_photons=n_photons,
            circuit_index=CIRCUITS_DISPONIBLES[nom_circuit],
        )
        modeles[nom_circuit] = model

        matrice_r, fcc, labels_w, matrice_c = calculer_empreinte_fourier_2d(
            model,
            M=M,
            res_grid=res_grid,
            n_omega=n_omega,
        )
        resultats[nom_circuit] = (matrice_r, fcc, labels_w, matrice_c)

        print("\n" + "=" * 55)
        print(f" ANALYSIS RESULTS: {nom_circuit}")
        print("=" * 55)
        print("Total optical modes       : 5 (4 encoded + 1 reference)")
        print(f"Active frequencies (ω1, ω2): {len(labels_w)} pairs")
        print(f"FCC score (correlation)   : {fcc:.5f}")
        print("=" * 55)

        afficher_fingerprint_physique_2d(
            matrice_r,
            labels_w,
            fcc,
            ax=ax_fingerprint,
            nom_circuit=f"{nom_circuit} - {facteur_echelle}",
        )

    figure.tight_layout()
    if rundir is None:
        plt.show()
    else:
        rundir = Path(rundir)
        rundir.mkdir(parents=True, exist_ok=True)
        nom_fichier = name or f"matrices_covariance_2d_{facteur_echelle}.png"
        if not nom_fichier.lower().endswith(".png"):
            nom_fichier += ".png"
        chemin_figure = rundir / nom_fichier
        figure.savefig(chemin_figure, bbox_inches="tight")
        print(f"Matrix figure saved to: {chemin_figure.resolve()}")
        plt.close(figure)

    if debug:
        for nom_circuit in circuits:
            print(f"\n--- Circuit: {nom_circuit} ---")
            pcvl.pdisplay(modeles[nom_circuit].quantum_layer.circuit)

    return resultats


if __name__ == "__main__":
    main(
        circuits=["circuit_0", "circuit_1", "circuit_2"],
        facteur_echelle="balanced",
        rundir=Path(__file__).resolve().parent.parent / "outdir",
        name="fourier_fingerprint_2d",
        debug=True
    )
