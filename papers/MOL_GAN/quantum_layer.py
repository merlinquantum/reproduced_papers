import torch
import torch.nn as nn
from merlin.algorithms import QuantumLayer
from merlin.builder import CircuitBuilder


class PhotonicRewardModule(nn.Module):
    def __init__(
        self,
        pca_components,
        pca_mean,
        num_atoms=9,
        num_bonds=4,
        num_atom_types=5,
        nb_modes=6,
        nb_photons=2,
    ):
        super().__init__()

        in_dim = (num_atoms * num_atoms * num_bonds) + (num_atoms * num_atom_types)

        self.pca_layer = nn.Linear(in_dim, nb_modes)
        self.pca_layer.weight.data = torch.tensor(pca_components, dtype=torch.float32)

        bias = (
            -torch.tensor(pca_mean, dtype=torch.float32) @ self.pca_layer.weight.data.T
        )
        self.pca_layer.bias.data = bias

        self.pca_layer.requires_grad_(False)
        self.normalization = nn.Sigmoid()

        builder = CircuitBuilder(n_modes=nb_modes)
        builder.add_entangling_layer(trainable=True, name="U1")
        builder.add_angle_encoding(modes=list(range(nb_modes)), name="input")
        builder.add_rotations(trainable=True, name="Theta")

        self.quantum_layer = QuantumLayer(
            input_size=nb_modes, builder=builder, n_photons=nb_photons
        )

        # 3. MAPPING CLASSIQUE AVAL
        q_out_dim = (
            self.quantum_layer.output_size
            if hasattr(self.quantum_layer, "output_size")
            else nb_modes
        )
        self.mapping = nn.Sequential(
            nn.Linear(q_out_dim, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid()
        )

    def forward(self, adj_tensor, node_matrix):
        # A. Concaténation des tenseurs du graphe
        flat_adj = adj_tensor.flatten(start_dim=1)
        flat_nodes = node_matrix.flatten(start_dim=1)
        x = torch.cat([flat_adj, flat_nodes], dim=1)

        # B. Réduction en angles [0, 2pi]
        # (CORRECTION 2 : L'ancienne ligne self.encoder a été supprimée)
        pca_out = self.pca_layer(x)
        angles = self.normalization(pca_out) * (2 * torch.pi)

        # C. Passage dans l'interféromètre MerLin
        q_out = self.quantum_layer(angles)

        # D. Prédiction finale du Reward Quantique RQ
        r_q = self.mapping(q_out)
        return r_q
