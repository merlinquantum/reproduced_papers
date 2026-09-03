from pathlib import Path

import matplotlib.pyplot as plt
import merlin as ml
import numpy as np
import perceval as pcvl
import torch
import torch.nn as nn
from merlin.builder import CircuitBuilder

# =====================================================================
# 1. MERLIN 1D MODEL WITH ENCODING
# =====================================================================
FACTEURS_ECHELLE_DISPONIBLES = {
    "exponential": [1.0, 2.0, 4.0, 8.0],
    "linear": [1.0, 2.0, 3.0, 4.0],
    "balanced": [-2.0, -1.0, 1.0, 2.0],
}


class PhotonicSpectralModel(nn.Module):
    def __init__(
        self,
        facteur_echelle="linear",
        n_photons=2,
        circuit_index=0,
    ):
        super().__init__()
        if facteur_echelle not in FACTEURS_ECHELLE_DISPONIBLES:
            noms_echelles_valides = ", ".join(FACTEURS_ECHELLE_DISPONIBLES)
            raise ValueError(
                f"Invalid scale factor: {facteur_echelle!r}. "
                f"Choose from: {noms_echelles_valides}"
            )

        self.facteur_echelle = facteur_echelle
        facteurs = FACTEURS_ECHELLE_DISPONIBLES[facteur_echelle]
        self.facteurs = torch.tensor(facteurs, dtype=torch.float32)
        self.n_photons = n_photons
        self.n_actifs = len(facteurs)
        self.n_modes = self.n_actifs + 1  # 4 encoded modes + 1 reference mode = 5 modes
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
        encoding_layers = {0: 1, 1: 1, 2: 2, 3: 1}
        if circuit_index not in encoding_layers:
            raise ValueError(
                f"Invalid circuit_index: {circuit_index}. Choose from {sorted(encoding_layers)}"
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

    def forward(self, x_1d):
        # Broadcast the 1D input across the 4 weighted waveguides.
        encoded_input = x_1d * self.facteurs
        repeated_input = torch.cat([encoded_input] * self.n_encoding_layers, dim=1)
        return self.quantum_layer(repeated_input)


# =====================================================================
# 2. FOURIER COEFFICIENTS AND PAIRWISE CORRELATIONS (FINGERPRINT)
# =====================================================================
def calculer_empreinte_fourier_1d(model, M=200, n_points=64):
    """
    Sample M model configurations to extract the c_w coefficients,
    compute the Pearson correlation between each pair of frequencies,
    and return the FCC score.
    """
    print(f"--- 1. Sampling {M} random parameter configurations ---")

    # Grille d'échantillonnage régulière dans [0, 2pi)
    x_grid = (
        torch.from_numpy(np.linspace(0, 2 * np.pi, n_points, endpoint=False))
        .float()
        .unsqueeze(1)
    )

    coefficients_list = []

    for m in range(M):
        with torch.no_grad():
            # Uniformly reset the theta parameters in [0, 2pi].
            for param in model.parameters():
                if param.requires_grad:
                    param.data.uniform_(0, 2 * np.pi)

            # Evaluate the optical circuit on the 64 points.
            probs_out = model(x_grid)
            if m == 0:
                print(
                    probs_out.shape
                )  # Display the output tensor shape for verification.
            # In the FOCK basis (5 modes, 2 photons), columns 0..4 are states with n0 >= 1.
            signal_y = (
                probs_out[:, :5].sum(dim=1).numpy()
            )  # P(at least 1 photon in mode 0).

        # Transformée de Fourier rapide réelle (rFFT)
        fft_coeffs = np.fft.rfft(signal_y) / n_points

        # Store the absolute amplitude |c_w| for each frequency.
        coefficients_list.append(np.abs(fft_coeffs))

    # Matrice brute C de forme (M, n_points//2 + 1) -> (200, 33)
    C_matrix = np.array(coefficients_list)

    print("--- 2. Filtrage des fréquences actives et calcul des corrélations ---")

    # Identify the frequencies w that are actually present in this circuit (variance > threshold).
    variances = np.var(C_matrix, axis=0)
    indices_actifs = np.where(variances > 1e-8)[0]

    # Keep only the columns corresponding to active frequencies.
    C_actives = C_matrix[:, indices_actifs]

    # Compute the Pearson correlation matrix r(w, w') between columns.
    n_actives = len(indices_actifs)
    if n_actives == 0:
        fingerprint = np.empty((0, 0))
        score_fcc = 0.0
    else:
        fingerprint = np.atleast_2d(np.corrcoef(C_actives, rowvar=False))
        fingerprint = np.nan_to_num(fingerprint, nan=0.0)

        # With a single frequency, there is no off-diagonal pair.
        if n_actives == 1:
            score_fcc = 0.0
        else:
            masque_hors_diag = ~np.eye(n_actives, dtype=bool)
            score_fcc = np.mean(np.abs(fingerprint[masque_hors_diag]))

    return fingerprint, score_fcc, indices_actifs, C_actives


# =====================================================================
# 3. PAPER-STYLE DISPLAY (TRIANGULAR HEATMAP WITH ACTUAL ω VALUES)
# =====================================================================
def afficher_fingerprint_physique(
    fingerprint, freqs, fcc_val, ax=None, nom_circuit=None
):
    """
    Display the lower triangular correlation matrix,
    labeling the axes with the actual integer frequencies ω.
    """
    masque = np.triu(np.ones_like(fingerprint, dtype=bool))
    matrice_affichee = np.ma.masked_array(np.abs(fingerprint), masque)

    figure_locale = ax is None
    if figure_locale:
        _, ax = plt.subplots(figsize=(8, 7))

    if fingerprint.size:
        im = ax.imshow(matrice_affichee, cmap="plasma_r", vmin=0, vmax=1)
        ax.figure.colorbar(im, ax=ax)
    else:
        ax.text(0.5, 0.5, "No active frequency", ha="center", va="center")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    # Replace column numbers with the actual frequency values w.
    ax.set_xticks(
        range(len(freqs)), labels=[f"ω={w}" for w in freqs], rotation=45, ha="right"
    )
    ax.set_yticks(range(len(freqs)), labels=[f"ω={w}" for w in freqs])

    titre = nom_circuit or "Fourier Fingerprint"
    ax.set_title(f"{titre}\nFCC = {fcc_val:.4f}", fontweight="bold")

    if figure_locale:
        ax.figure.tight_layout()
        plt.show()


# =====================================================================
# 4. PROGRAM ENTRY POINT
# =====================================================================
CIRCUITS_DISPONIBLES = {"circuit_0": 0, "circuit_1": 1, "circuit_2": 2, "circuit_3": 3}


def main(
    circuits=None,
    facteur_echelle="linear",
    M=200,
    n_points=64,
    debug=False,
    rundir: Path | None = None,
    name: str | None = None,
):
    """Test the requested circuits and display their fingerprints."""
    if circuits is None:
        circuits = ["circuit_2"]
    if not isinstance(circuits, list):
        raise TypeError("circuits must be a list of circuit names.")
    if not circuits:
        raise ValueError("circuits must contain at least one circuit.")
    if facteur_echelle not in FACTEURS_ECHELLE_DISPONIBLES:
        noms_echelles_valides = ", ".join(FACTEURS_ECHELLE_DISPONIBLES)
        raise ValueError(
            f"Invalid scale factor: {facteur_echelle!r}. "
            f"Choose from: {noms_echelles_valides}"
        )

    noms_inconnus = [
        nom_circuit
        for nom_circuit in circuits
        if not isinstance(nom_circuit, str) or nom_circuit not in CIRCUITS_DISPONIBLES
    ]
    if noms_inconnus:
        noms_valides = ", ".join(sorted(CIRCUITS_DISPONIBLES))
        raise ValueError(
            f"Invalid circuits: {noms_inconnus}. Choose from: {noms_valides}"
        )

    figure, axes = plt.subplots(
        1,
        len(circuits),
        figsize=(8 * len(circuits), 7),
        squeeze=False,
    )
    resultats = {}
    modeles = {}

    for nom_circuit, ax_fingerprint in zip(circuits, axes[0]):
        model = PhotonicSpectralModel(
            facteur_echelle=facteur_echelle,
            n_photons=2,
            circuit_index=CIRCUITS_DISPONIBLES[nom_circuit],
        )
        modeles[nom_circuit] = model

        matrice_r, fcc, freqs_w, matrice_c = calculer_empreinte_fourier_1d(
            model, M=M, n_points=n_points
        )
        resultats[nom_circuit] = (matrice_r, fcc, freqs_w, matrice_c)

        print("\n" + "=" * 55)
        print(f" ANALYSIS RESULTS: {nom_circuit}")
        print("=" * 55)
        print("Total optical modes        : 5 (4 encoded + 1 reference)")
        print(f"Active frequencies ω       : {len(freqs_w)} harmonics")
        frequence_max = max(freqs_w) if len(freqs_w) else "none"
        print(f"Maximum frequency (ω_max) : {frequence_max}")
        print(f"FCC score (correlation)    : {fcc:.5f}")
        print("=" * 55)

        afficher_fingerprint_physique(
            matrice_r,
            freqs_w,
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
        if name is None:
            nom_fichier = f"matrices_covariance_{facteur_echelle}.png"
        else:
            nom_fichier = name
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
        circuits=["circuit_0", "circuit_1", "circuit_2", "circuit_3"],
        facteur_echelle="balanced",
        name="Fig 2(a) - Test configuration",
        rundir=Path(__file__).resolve().parent.parent / "outdir",
    )
