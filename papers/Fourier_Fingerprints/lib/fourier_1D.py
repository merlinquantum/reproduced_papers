import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import merlin as ml
import perceval as pcvl
from merlin.builder import CircuitBuilder

# =====================================================================
# 1. MODÈLE MERLIN 1D
# =====================================================================
class PhotonicSpectralModel(nn.Module):
    def __init__(
        self,
        facteurs_echelle=None,
        facteurs_echelle_type="exponential",
        n_facteurs=4,
        facteur_base=1.0,
        n_photons=2,
        circuit_index=0,
        encoding_strategy=0,
        afficher_circuit=True,
    ):
        super().__init__()
        self.facteurs_mode = str(facteurs_echelle_type).lower()
        if self.facteurs_mode not in {"linear", "exponential"}:
            raise ValueError(
                "facteurs_echelle_type invalide. Choisir parmi ['linear', 'exponential']"
            )
        facteurs = self._resolve_facteurs_echelle(
            facteurs_echelle=facteurs_echelle,
            facteurs_echelle_type=self.facteurs_mode,
            n_facteurs=n_facteurs,
            facteur_base=facteur_base,
        )
        self.facteurs = torch.tensor(facteurs, dtype=torch.float32)
        self.n_photons = n_photons
        self.n_actifs = len(facteurs)
        self.n_modes = self.n_actifs + 1  # 4 modes encodés + 1 mode de référence = 5 modes
        self.measurement_strategy = ml.MeasurementStrategy.probs(
            computation_space=ml.ComputationSpace.FOCK
        )
        self.afficher_circuit = afficher_circuit

        self.circuit_index = circuit_index
        self.encoding_strategy = encoding_strategy
        self.n_encoding_layers = self._get_num_encoding_layers(circuit_index)
        self.quantum_layer = self._build_quantum_layer(circuit_index, encoding_strategy)

    def _resolve_facteurs_echelle(
        self,
        facteurs_echelle,
        facteurs_echelle_type,
        n_facteurs,
        facteur_base,
    ):
        if facteurs_echelle is not None:
            if len(facteurs_echelle) == 0:
                raise ValueError("facteurs_echelle ne peut pas être vide")
            return [float(v) for v in facteurs_echelle]

        if n_facteurs < 1:
            raise ValueError("n_facteurs doit être >= 1")
        if facteur_base <= 0:
            raise ValueError("facteur_base doit être > 0")

        mode = facteurs_echelle_type
        if mode == "linear":
            return [facteur_base * (i + 1) for i in range(n_facteurs)]
        if mode == "exponential":
            return [facteur_base * (2 ** i) for i in range(n_facteurs)]

        raise ValueError(
            "facteurs_echelle_type invalide. Choisir parmi ['linear', 'exponential']"
        )

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

    def _add_butterfly(self, builder, name_prefix="butterfly"):
        # Butterfly fixe pour 5 modes:
        # (0,1) et (3,4) en parallele, puis (1,2), puis (2,3), puis (3,4) et (0,1).
        if self.n_modes < 5:
            raise ValueError("_add_butterfly requiert au moins 5 modes")

        builder.add_entangling_layer(modes=[0, 1], trainable=True, model="mzi", name=f"{name_prefix}_0_1_a")
        builder.add_entangling_layer(modes=[3, 4], trainable=True, model="mzi", name=f"{name_prefix}_3_4_a")

        builder.add_entangling_layer(modes=[1, 2], trainable=True, model="mzi", name=f"{name_prefix}_1_2")
        builder.add_entangling_layer(modes=[2, 3], trainable=True, model="mzi", name=f"{name_prefix}_2_3")

        builder.add_entangling_layer(modes=[3, 4], trainable=True, model="mzi", name=f"{name_prefix}_3_4_b")
        builder.add_entangling_layer(modes=[0, 1], trainable=True, model="mzi", name=f"{name_prefix}_0_1_b")

    

    def _show_circuit(self, builder):
        if not self.afficher_circuit:
            return
        print("\n--- Topologie du circuit optique ---")
        # Compatible avec differentes versions de Merlin.
        if hasattr(builder, "show") and callable(builder.show):
            builder.show()
        elif hasattr(builder, "draw") and callable(builder.draw):
            builder.draw()
        else:
            print(builder)
        pcvl.pdisplay(builder.to_pcvl_circuit())

    def _build_circuit_type_0(self, builder, encoding_strategy):
        # Topologie de base: entanglement -> encodage -> entanglement.
        builder.add_entangling_layer(trainable=True, model="mzi", name="init_mzi")
        self._add_encoding(builder, encoding_strategy)
        builder.add_entangling_layer(trainable=True, model="mzi", name="mid_mzi")
        self._show_circuit(builder)

    def _build_circuit_type_1(self, builder, encoding_strategy):
        self._add_butterfly(builder, name_prefix="pre_butterfly")
        self._add_encoding(builder, encoding_strategy)
        self._add_butterfly(builder, name_prefix="post_butterfly")
        self._show_circuit(builder)

    def _build_circuit_type_2(self, builder, encoding_strategy):
        # Topologie plus profonde: deux couches avant encodage puis mélange final.
        builder.add_entangling_layer(trainable=True, model="mzi", name="pre_mzi_0")
        self._add_encoding(builder, encoding_strategy)
        builder.add_entangling_layer(trainable=True, model="mzi", name="pre_mzi_1")
        self._add_encoding(builder, encoding_strategy)
        builder.add_entangling_layer(trainable=True, model="mzi", name="post_mzi")
        self._show_circuit(builder)

    def _build_circuit_type_3(self, builder, encoding_strategy):
        # Topologie plus profonde: deux couches avant encodage puis mélange final.
        builder.add_entangling_layer(trainable=True, model="mzi", name="pre_mzi_0")
        self._add_encoding(builder, encoding_strategy)
        builder.add_entangling_layer(trainable=True, model="mzi", name="pre_mzi_1")
        self._add_encoding(builder, encoding_strategy)
        builder.add_entangling_layer(trainable=True, model="mzi", name="post_mzi")
        self._add_encoding(builder, encoding_strategy)
        builder.add_entangling_layer(trainable=True, model="mzi", name="post_mzi")
        self._show_circuit(builder)

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
        # Diffusion (broadcast) de l'entrée 1D sur les guides d'ondes pondérés.
        # Pour plusieurs couches d'encodage, on continue la suite de facteurs
        # (linéaire ou exponentielle) au lieu de répéter le même bloc.
        facteurs_ref = self.facteurs.to(device=x_1d.device, dtype=x_1d.dtype)
        if self.n_encoding_layers == 1:
            return self.quantum_layer(x_1d * facteurs_ref)

        if self.facteurs_mode == "linear":
            if self.n_actifs > 1:
                step = facteurs_ref[1] - facteurs_ref[0]
            else:
                step = facteurs_ref[0]
            start = facteurs_ref[0]
            full_factors = start + step * torch.arange(
                self.n_actifs * self.n_encoding_layers,
                device=x_1d.device,
                dtype=x_1d.dtype,
            )
        else:
            if self.n_actifs > 1 and facteurs_ref[0] != 0:
                ratio = facteurs_ref[1] / facteurs_ref[0]
            else:
                ratio = torch.tensor(2.0, device=x_1d.device, dtype=x_1d.dtype)
            full_factors = facteurs_ref[0] * torch.pow(
                ratio,
                torch.arange(
                    self.n_actifs * self.n_encoding_layers,
                    device=x_1d.device,
                    dtype=x_1d.dtype,
                ),
            )

        repeated_input = x_1d * full_factors.unsqueeze(0)
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
    indices_actifs = np.where(variances > 1e-10)[0]

    # On ne garde que les colonnes des fréquences actives.
    C_actives = C_matrix[:, indices_actifs]
    n_actives = len(indices_actifs)

    if n_actives <= 1:
        raise RuntimeError(
            f"Circuit insuffisamment expressif: {n_actives} fréquence active détectée "
            "(seuil variance > 1e-10). FCC non défini pour <= 1 fréquence active. "
            "Essayez une topologie plus profonde (ex: circuit_index=2), un autre observable, "
            "ou davantage d'échantillons."
        )

    # Calcul de la matrice de corrélation de Pearson r(w, w') entre les colonnes.
    fingerprint = np.corrcoef(C_actives, rowvar=False)
    fingerprint = np.nan_to_num(fingerprint, nan=0.0)

    # Calcul du score FCC : moyenne des valeurs absolues hors diagonale principale.
    masque_hors_diag = ~np.eye(n_actives, dtype=bool)
    score_fcc = np.mean(np.abs(fingerprint[masque_hors_diag]))

    return fingerprint, score_fcc, indices_actifs, C_actives

# =====================================================================
# 4. AFFICHAGE STYLE PAPIER (HEATMAP TRIANGULAIRE AVEC VRAIS ω)
# =====================================================================
def afficher_fingerprint_physique(fingerprint, freqs, fcc_val, output_path=None):
    """
    Affiche ou enregistre la matrice de corrélation triangulaire inférieure,
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
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def executer_fourier_fingerprint_1d(output_dir, circuit_index, encoding="linear"):
    """
    Fonction globale d'exécution du script 1D.

    Args:
        output_dir (str | Path): Dossier de sortie pour enregistrer les graphiques.
        circuit_index (int): Indice de topologie du circuit à utiliser.

    Returns:
        dict: Résultats numériques principaux (FCC, fréquences actives, matrice de corrélation).
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    n_modes = 5
    n_photons = 2

    n_facteurs = n_modes - 1
    model_exp = PhotonicSpectralModel(
        facteurs_echelle_type=encoding,
        n_facteurs=n_facteurs,
        n_photons=n_photons,
        circuit_index=circuit_index,
        encoding_strategy=0,
        afficher_circuit=False,
    )

    matrice_r, fcc, freqs_w, matrice_c = calculer_empreinte_fourier_1d(
        model_exp,
        M=200,
        n_points=128,
    )

    terminal_log_path = output_path / "terminal.txt"
    logs = [
        "\n" + "=" * 55,
        " RÉSULTATS DE L'ANALYSE SPECTRALE MERLIN (1D EXP)",
        "=" * 55,
        f"Modes optiques totaux      : {n_modes} ({n_facteurs} encodés + 1 référence)",
        f"Fréquences ω actives       : {len(list(freqs_w))} harmoniques",
        f"Fréquence maximale (ω_max) : {max(freqs_w)}",
        f"Score FCC (Corrélation)    : {fcc:.5f}",
        "=" * 55,
    ]
    with terminal_log_path.open("a", encoding="utf-8") as terminal_log:
        for line in logs:
            print(line)
            terminal_log.write(line + "\n")

    figure_path = output_path / f"fingerprint_1D_circuit_{circuit_index}_modes_{n_modes}_photons_{n_photons}.png"
    afficher_fingerprint_physique(matrice_r, freqs_w, fcc, output_path=figure_path)
    print(f"Figure enregistrée : {figure_path}")

    return {
        "fcc": float(fcc),
        "frequences_actives": [int(w) for w in freqs_w],
        "fingerprint": matrice_r,
        "coefficients_actifs": matrice_c,
        "figure_path": str(figure_path),
    }

if __name__ == "__main__":
    executer_fourier_fingerprint_1d(
        output_dir="outdir",
        circuit_index=2,
        encoding="exponential"
    )