import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import merlin as ml
from merlin.builder import CircuitBuilder

# =====================================================================
# 1. MODÈLE MERLIN 1D AVEC ENCODAGE EXPONENTIEL [1, 2, 4, 8]
# =====================================================================
class PhotonicSpectralModel(nn.Module):
    def __init__(
        self,
        facteurs_echelle=[1.0, 2.0, 4.0, 8.0],
        n_photons=2,
        circuit_index=0,
        encoding_strategy=0,
    ):
        super().__init__()
        self.facteurs = torch.tensor(facteurs_echelle, dtype=torch.float32)
        self.n_photons = n_photons
        self.n_actifs = len(facteurs_echelle)
        self.n_modes = self.n_actifs + 1  # 4 modes encodés + 1 mode de référence = 5 modes
        self.measurement_strategy = ml.MeasurementStrategy.probs(
            computation_space=ml.ComputationSpace.FOCK
        )

        self.circuit_index = circuit_index
        self.encoding_strategy = encoding_strategy
        self.n_encoding_layers = self._get_num_encoding_layers(circuit_index)
        self.quantum_layer = self._build_quantum_layer(circuit_index, encoding_strategy)

    def _build_quantum_layer(self, circuit_index, encoding_strategy):
        builder = self._build_circuit(circuit_index, encoding_strategy)
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
        }
        if circuit_index not in encoding_layers:
            raise ValueError(
                f"circuit_index invalide: {circuit_index}. Choisir parmi {sorted(encoding_layers)}"
            )
        return encoding_layers[circuit_index]

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
        self._add_encoding(builder, encoding_strategy)
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
        self.n_encoding_layers = self._get_num_encoding_layers(circuit_index)
        self.quantum_layer = self._build_quantum_layer(circuit_index, encoding_strategy)
        
    def forward(self, x_1d):
        # Diffusion (broadcast) de l'entrée 1D sur les 4 guides d'ondes pondérés
        encoded_input = x_1d * self.facteurs
        repeated_input = torch.cat([encoded_input] * self.n_encoding_layers, dim=1)
        return self.quantum_layer(repeated_input)

# =====================================================================
# 2. CALCUL DES COEFFICIENTS ET DES CORRÉLATIONS 2 À 2 (FINGERPRINT)
# =====================================================================
def calculer_empreinte_fourier_1d(model, M=200, n_points=64):
    """
    Échantillonne M configurations du modèle pour extraire les coefficients c_w,
    calculer la corrélation de Pearson entre chaque paire de fréquences,
    et retourner le score FCC.
    """
    print(f"--- 1. Échantillonnage de {M} configurations aléatoires de paramètres ---")
    
    # Grille d'échantillonnage régulière dans [0, 2pi)
    x_grid = torch.from_numpy(np.linspace(0, 2 * np.pi, n_points, endpoint=False)).float().unsqueeze(1)
    
    coefficients_list = []
    
    for m in range(M):
        with torch.no_grad():
            # Réinitialisation uniforme des paramètres theta dans [0, 2pi]
            for param in model.parameters():
                if param.requires_grad:
                    param.data.uniform_(0, 2 * np.pi)
            
            # Évaluation du circuit optique sur les 64 points
            probs_out = model(x_grid)
            if m == 0:
                print(probs_out.shape)  # Affiche la forme du tenseur de sortie pour vérification
            # En base FOCK (5 modes, 2 photons), les colonnes 0..4 sont les états avec n0 >= 1.
            signal_y = probs_out[:, :5].sum(dim=1).numpy()  # P(au moins 1 photon dans le mode 0)
            
        # Transformée de Fourier rapide réelle (rFFT)
        fft_coeffs = np.fft.rfft(signal_y) / n_points
        
        # On stocke l'amplitude absolue |c_w| pour chaque fréquence
        coefficients_list.append(np.abs(fft_coeffs))
        
    # Matrice brute C de forme (M, n_points//2 + 1) -> (200, 33)
    C_matrix = np.array(coefficients_list)
    
    print("--- 2. Filtrage des fréquences actives et calcul des corrélations ---")
    
    # Identifier les fréquences w qui existent réellement dans ce circuit (variance > seuil)
    variances = np.var(C_matrix, axis=0)
    indices_actifs = np.where(variances > 1e-8)[0]

    # On ne garde que les colonnes des fréquences actives
    C_actives = C_matrix[:, indices_actifs]
    
    # Calcul de la matrice de corrélation de Pearson r(w, w') entre les colonnes
    fingerprint = np.corrcoef(C_actives, rowvar=False)
    fingerprint = np.nan_to_num(fingerprint, nan=0.0)
    
    # Calcul du score FCC : Moyenne des valeurs absolues hors diagonale principale
    n_actives = len(indices_actifs)
    masque_hors_diag = ~np.eye(n_actives, dtype=bool)
    score_fcc = np.mean(np.abs(fingerprint[masque_hors_diag]))
    
    return fingerprint, score_fcc, indices_actifs, C_actives

# =====================================================================
# 3. EXÉCUTION DU DIAGNOSTIC SUR LE MODÈLE EXPONENTIEL
# =====================================================================
model_exp = PhotonicSpectralModel(
    facteurs_echelle=[1.0, 2.0, 4.0, 8.0],
    n_photons=2,
    circuit_index=2,
    encoding_strategy=0,
)
matrice_r, fcc, freqs_w, matrice_c = calculer_empreinte_fourier_1d(model_exp, M=200, n_points=64)

print("\n" + "="*55)
print(" RÉSULTATS DE L'ANALYSE SPECTRALE MERLIN (1D EXP)")
print("="*55)
print(f"Modes optiques totaux      : 5 (4 encodés + 1 référence)")
print(f"Fréquences ω actives       : {len(list(freqs_w))} harmoniques")
print(f"Fréquence maximale (ω_max) : {max(freqs_w)}")
print(f"Score FCC (Corrélation)    : {fcc:.5f}")
print("="*55)

# =====================================================================
# 4. AFFICHAGE STYLE PAPIER (HEATMAP TRIANGULAIRE AVEC VRAIS ω)
# =====================================================================
def afficher_fingerprint_physique(fingerprint, freqs, fcc_val):
    """
    Affiche la matrice de corrélation triangulaire inférieure, 
    en étiquetant les axes avec les fréquences entières ω réelles.
    """
    masque = np.triu(np.ones_like(fingerprint, dtype=bool))
    matrice_affichee = np.ma.masked_array(np.abs(fingerprint), masque)
    
    plt.figure(figsize=(8, 7))
    im = plt.imshow(matrice_affichee, cmap="plasma_r", vmin=0, vmax=1)
    
    # Remplacement des numéros de colonnes par les vraies valeurs de fréquences w
    plt.xticks(ticks=range(len(freqs)), labels=[f"ω={w}" for w in freqs], rotation=45, ha="right")
    plt.yticks(ticks=range(len(freqs)), labels=[f"ω={w}" for w in freqs])
    
    plt.colorbar(im, label="Corrélation de Pearson |r|")
    plt.title(f"Fourier Fingerprint - Encodage Exponentiel [1, 2, 4, 8]\nFCC = {fcc_val:.4f}", fontweight="bold")
    plt.xlabel("Fréquence spatiale ω'")
    plt.ylabel("Fréquence spatiale ω")
    plt.tight_layout()
    plt.show()

afficher_fingerprint_physique(matrice_r, freqs_w, fcc)