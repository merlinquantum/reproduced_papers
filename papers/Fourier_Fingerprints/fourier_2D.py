import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import merlin as ml
from merlin.builder import CircuitBuilder
from merlin.utils.combinadics import Combinadics

# =====================================================================
# 0. CONFIGURATION GLOBALE
# =====================================================================
N_PHOTONS = 3
CIRCUIT_INDEX = 0
ENCODING_STRATEGY = 2
MODE_PRESENCE_INDEX = 0  # Premier mode optique
MIN_PHOTONS_IN_MODE = 1  # Presence d'au moins un photon
N_OMEGA = 3

# =====================================================================
# 1. MODÈLE MERLIN 2D [1*x1, 2*x1, 1*x2, 2*x2]
# =====================================================================
class PhotonicSpectralModel2D(nn.Module):
    def __init__(self, n_photons=3, circuit_index=0, encoding_strategy=0):
        super().__init__()
        # On assigne 2 modes pour x1 (facteurs 1 et 2) et 2 modes pour x2 (facteurs 1 et 2)
        self.facteurs = torch.tensor([1.0, 2.0, 1.0, 2.0], dtype=torch.float32)
        self.indices_coord = [0, 0, 1, 1]  # Mode 0,1 -> x1 | Mode 2,3 -> x2

        self.n_photons = n_photons
        self.n_actifs = 4
        self.n_modes = self.n_actifs + 1  # + 1 mode de référence = 5 modes optiques
        self.measurement_strategy = ml.MeasurementStrategy.probs(
            computation_space=ml.ComputationSpace.FOCK
        )

        self.circuit_index = circuit_index
        self.encoding_strategy = encoding_strategy
        self._presence_indices_cache = {}
        self.quantum_layer = self._build_quantum_layer(circuit_index, encoding_strategy)

    def _build_quantum_layer(self, circuit_index, encoding_strategy):
        builder = self._build_circuit(circuit_index, encoding_strategy)
        return ml.QuantumLayer(
            input_size=self.n_actifs,
            builder=builder,
            n_photons=self.n_photons,
            measurement_strategy=self.measurement_strategy,
            dtype=torch.float32,
        )

    def _build_circuit(self, circuit_index, encoding_strategy):
        builder = CircuitBuilder(n_modes=self.n_modes)
        builders = {
            0: self._build_circuit_type_0,
            1: self._build_circuit_type_1,
            2: self._build_circuit_type_2,
        }
        if circuit_index not in builders:
            raise ValueError(f"circuit_index invalide: {circuit_index}. Choisir parmi {sorted(builders)}")
        builders[circuit_index](builder, encoding_strategy)
        return builder

    def _add_encoding(self, builder, encoding_strategy):
        modes = list(range(self.n_actifs))
        if encoding_strategy == 0:
            builder.add_angle_encoding(modes=modes, name="data_encoding")
        elif encoding_strategy == 1:
            builder.add_angle_encoding(
                modes=modes,
                name="data_encoding",
                subset_combinations=True,
                max_order=2,
            )
        elif encoding_strategy == 2:
            builder.add_angle_encoding(modes=modes, name="data_encoding", scale=0.5)
        else:
            raise ValueError(
                f"encoding_strategy invalide: {encoding_strategy}. Choisir parmi [0, 1, 2]"
            )

    def _build_circuit_type_0(self, builder, encoding_strategy):
        # Topologie de base: entanglement -> encodage -> entanglement.
        builder.add_entangling_layer(trainable=True, model="mzi", name="init_mzi")
        self._add_encoding(builder, encoding_strategy)
        builder.add_entangling_layer(trainable=True, model="mzi", name="mid_mzi")

    def _build_circuit_type_1(self, builder, encoding_strategy):
        # Topologie plus légère: encodage direct puis une seule couche d'entanglement.
        self._add_encoding(builder, encoding_strategy)
        builder.add_entangling_layer(trainable=True, model="mzi", name="post_mzi")

    def _build_circuit_type_2(self, builder, encoding_strategy):
        # Topologie plus profonde: deux couches avant encodage puis mélange final.
        builder.add_entangling_layer(trainable=True, model="mzi", name="pre_mzi_0")
        builder.add_entangling_layer(trainable=True, model="mzi", name="pre_mzi_1")
        self._add_encoding(builder, encoding_strategy)
        builder.add_entangling_layer(trainable=True, model="mzi", name="post_mzi")

    def configure_circuit(self, circuit_index=None, encoding_strategy=None):
        """
        Reconfigure dynamiquement la topologie du circuit et/ou la stratégie d'encodage.
        """
        if circuit_index is None:
            circuit_index = self.circuit_index
        if encoding_strategy is None:
            encoding_strategy = self.encoding_strategy

        self.circuit_index = circuit_index
        self.encoding_strategy = encoding_strategy
        self._presence_indices_cache.clear()
        self.quantum_layer = self._build_quantum_layer(circuit_index, encoding_strategy)

    def get_mode_presence_indices(self, mode_index=0, min_photons=1):
        """
        Retourne les indices de colonnes Fock tels que n_mode_index >= min_photons.
        """
        if not (0 <= mode_index < self.n_modes):
            raise ValueError(f"mode_index invalide: {mode_index}. Intervalle valide: [0, {self.n_modes - 1}]")
        if min_photons < 1:
            raise ValueError("min_photons doit etre >= 1")

        cache_key = (mode_index, min_photons)
        if cache_key not in self._presence_indices_cache:
            basis = Combinadics(m=self.n_modes, n=self.n_photons, scheme=ml.ComputationSpace.FOCK)
            n_cols = basis.compute_space_size()
            indices = [
                i for i in range(n_cols)
                if basis.index_to_fock(i)[mode_index] >= min_photons
            ]
            if not indices:
                raise RuntimeError(
                    "Aucune composante Fock ne satisfait le critere de presence."
                )
            self._presence_indices_cache[cache_key] = indices

        return self._presence_indices_cache[cache_key]
        
    def forward(self, x_2d):
        # x_2d est de forme [batch, 2].
        # On distribue x1 sur les 2 premiers modes et x2 sur les 2 suivants avec leurs facteurs
        x_multimode = x_2d[:, self.indices_coord] * self.facteurs
        return self.quantum_layer(x_multimode)

# =====================================================================
# 2. CALCUL DES COEFFICIENTS ET DES CORRÉLATIONS EN 2D
# =====================================================================
def calculer_empreinte_fourier_2d(
    model,
    M=200,
    res_grid=32,
    n_omega=3,
    mode_presence_index=0,
    min_photons_in_mode=1,
):
    """
    Évalue M configurations sur une grille 2D (res_grid x res_grid points)
    et extrait le spectre vectoriel w = (w1, w2).
    """
    print(f"--- 1. Grille 2D ({res_grid}x{res_grid} = {res_grid**2} points) et {M} tirages ---")
    
    # Création de la grille géométrique [0, 2pi) x [0, 2pi)
    axes = [np.linspace(0, 2 * np.pi, res_grid, endpoint=False) for _ in range(2)]
    grid_x1, grid_x2 = np.meshgrid(*axes, indexing='ij')
    
    # Tenseur d'entrée [res_grid**2, 2]
    x_grid = torch.from_numpy(np.stack([grid_x1.flatten(), grid_x2.flatten()], axis=-1)).float()
    presence_indices = model.get_mode_presence_indices(
        mode_index=mode_presence_index,
        min_photons=min_photons_in_mode,
    )
    
    coefficients_list = []
    
    for m in range(M):
        with torch.no_grad():
            for param in model.parameters():
                if param.requires_grad:
                    param.data.uniform_(0, 2 * np.pi)
            
            probs_out = model(x_grid)
            if m == 0:
                print(probs_out.shape)  # Forme attendue: (res_grid**2, 15) en espace Fock
                print(
                    f"Colonnes pour n_mode{mode_presence_index} >= {min_photons_in_mode}: "
                    f"{len(presence_indices)}"
                )
            signal_y = probs_out[:, presence_indices].sum(dim=1).numpy()
            
        # Reshape en image 2D [32, 32] pour la FFT
        signal_2d = signal_y.reshape((res_grid, res_grid))
        
        # Transformée de Fourier 2D (FFT)
        fft_coeffs = np.fft.fftn(signal_2d) / (res_grid ** 2)
        
        # Aplatissement du tenseur de fréquences 2D en vecteur 1D de modules |c_w|
        coefficients_list.append(np.abs(fft_coeffs.flatten()))
        
    C_matrix = np.array(coefficients_list)
    
    print("--- 2. Identification des paires de fréquences actives (w1, w2) ---")
    
    # Filtrage des fréquences actives (variance non nulle) puis restriction a Omega_n.
    # Omega_n = {(w1, w2) : |w1| + |w2| <= n_omega} est equivalent a
    # {-omega, 0, omega | omega1, omega2 in [0, n], omega1 + omega2 <= n}.
    variances = np.var(C_matrix, axis=0)
    indices_actifs = np.where(variances > 1e-8)[0]

    indices_filtres = []
    freqs_labels = []
    for idx in indices_actifs:
        w1 = idx // res_grid
        w2 = idx % res_grid
        # Ajustement pour les fréquences négatives dans la notation FFT standard
        if w1 >= res_grid // 2:
            w1 -= res_grid
        if w2 >= res_grid // 2:
            w2 -= res_grid
        if abs(w1) + abs(w2) <= n_omega:
            indices_filtres.append(idx)
            freqs_labels.append(f"({w1},{w2})")

    indices_actifs = np.array(indices_filtres, dtype=int)
    C_actives = C_matrix[:, indices_actifs]
        
    # Corrélation de Pearson et FCC
    fingerprint = np.corrcoef(C_actives, rowvar=False)
    fingerprint = np.nan_to_num(fingerprint, nan=0.0)
    
    n_actives = len(indices_actifs)
    masque_hors_diag = ~np.eye(n_actives, dtype=bool)
    score_fcc = np.mean(np.abs(fingerprint[masque_hors_diag]))
    
    return fingerprint, score_fcc, freqs_labels, C_actives

# =====================================================================
# 3. EXÉCUTION ET AFFICHAGE
# =====================================================================
model_2d = PhotonicSpectralModel2D(
    n_photons=N_PHOTONS,
    circuit_index=CIRCUIT_INDEX,
    encoding_strategy=ENCODING_STRATEGY,
)
matrice_r, fcc, labels_w, _ = calculer_empreinte_fourier_2d(
    model_2d,
    M=150,
    res_grid=32,
    n_omega=N_OMEGA,
    mode_presence_index=MODE_PRESENCE_INDEX,
    min_photons_in_mode=MIN_PHOTONS_IN_MODE,
)

print("\n" + "="*55)
print(" RÉSULTATS DE L'ANALYSE SPECTRALE MERLIN (2D)")
print("="*55)
print(f"Modes optiques totaux    : 5 (4 encodés + 1 référence)")
print(f"Fréquences (ω1, ω2)      : {len(labels_w)} harmoniques actives")
print(f"Score FCC (Corrélation)  : {fcc:.5f}")
print("="*55)

# Affichage visuel (on limite aux 25 premières harmoniques pour la lisibilité)
n_aff = min(25, len(labels_w))
masque = np.triu(np.ones((n_aff, n_aff), dtype=bool))
matrice_affichee = np.ma.masked_array(np.abs(matrice_r[:n_aff, :n_aff]), masque)

plt.figure(figsize=(9, 8))
im = plt.imshow(matrice_affichee, cmap="plasma_r", vmin=0, vmax=1)
plt.xticks(ticks=range(n_aff), labels=labels_w[:n_aff], rotation=60, ha="right")
plt.yticks(ticks=range(n_aff), labels=labels_w[:n_aff])
plt.colorbar(im, label="Corrélation de Pearson |r|")
plt.title(f"Fourier Fingerprint - Modèle 2D [1x1, 2x1, 1x2, 2x2]\nFCC = {fcc:.4f}", fontweight="bold")
plt.xlabel("Fréquence spatiale (ω1', ω2')")
plt.ylabel("Fréquence spatiale (ω1, ω2)")
plt.tight_layout()
plt.show()