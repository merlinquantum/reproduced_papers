import numpy as np
import pytest
import torch
import torch.nn as nn

# A AJUSTER : Remplacez 'nom_de_votre_fichier' par le nom du fichier Python
# qui contient la classe PhotonicRewardModule
from lib.quantum_layer import PhotonicRewardModule


@pytest.fixture
def setup_data():
    """Génère les tenseurs factices (dummy data) pour les tests."""
    batch_size = 2
    num_atoms = 9
    num_bonds = 4
    num_atom_types = 5
    nb_modes = 6
    in_dim = (num_atoms * num_atoms * num_bonds) + (num_atoms * num_atom_types)  # 369

    # Tenseurs d'entrée (Graphe)
    adj_tensor = torch.rand(batch_size, num_atoms, num_atoms, num_bonds)
    node_matrix = torch.rand(batch_size, num_atoms, num_atom_types)

    # Paramètres ACP simulés (statiques)
    pca_components = np.random.randn(nb_modes, in_dim)
    pca_mean = np.random.randn(in_dim)

    # Cible factice (Reward attendu)
    target = torch.rand(batch_size, 1)

    return adj_tensor, node_matrix, target, pca_components, pca_mean


@pytest.fixture
def model(setup_data):
    """Instancie le modèle avec les paramètres ACP générés."""
    _, _, _, pca_components, pca_mean = setup_data
    return PhotonicRewardModule(
        pca_components=pca_components,
        pca_mean=pca_mean,
        num_atoms=9,
        num_bonds=4,
        num_atom_types=5,
        nb_modes=6,
    )


def test_forward_shape_and_bounds(model, setup_data):
    adj_tensor, node_matrix, _, _, _ = setup_data
    batch_size = adj_tensor.shape[0]

    # Mode évaluation
    model.eval()
    with torch.no_grad():
        output = model(adj_tensor, node_matrix)

    # Vérification des dimensions
    assert output.shape == (batch_size, 1), (
        f"Format attendu: {(batch_size, 1)}, obtenu: {output.shape}"
    )

    # Vérification des bornes (la dernière couche est une Sigmoïde, donc dans [0, 1])
    assert torch.all(output >= 0.0) and torch.all(output <= 1.0), (
        "La sortie n'est pas bornée entre 0 et 1."
    )


def test_backward_and_gradients(model, setup_data):
    adj_tensor, node_matrix, target, _, _ = setup_data

    model.train()
    output = model(adj_tensor, node_matrix)

    criterion = nn.MSELoss()
    loss = criterion(output, target)
    loss.backward()

    # Vérification stricte des gradients
    for name, param in model.named_parameters():
        if param.requires_grad:
            # Les couches entraînables (Circuit quantique + Mapping) DOIVENT avoir un gradient
            assert param.grad is not None, (
                f"Le paramètre entraînable '{name}' n'a pas reçu de gradient."
            )
            assert torch.sum(torch.abs(param.grad)) > 0, (
                f"Le gradient de '{name}' est nul."
            )
        else:
            # La couche ACP NE DOIT PAS avoir de gradient
            assert param.grad is None, (
                f"Erreur : le paramètre gelé '{name}' a reçu un gradient !"
            )
            assert "pca" in name.lower(), f"Un paramètre inattendu a été gelé : {name}"


def test_optimizer_step(model, setup_data):
    adj_tensor, node_matrix, target, _, _ = setup_data

    # On filtre les paramètres pour ne donner à l'optimiseur que ceux qui s'entraînent
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=0.01)

    # Sauvegarde des poids avant l'étape d'optimisation
    weights_before = {name: param.clone() for name, param in model.named_parameters()}

    model.train()
    optimizer.zero_grad()
    output = model(adj_tensor, node_matrix)
    loss = nn.MSELoss()(output, target)
    loss.backward()
    optimizer.step()

    # Vérification de la mise à jour des poids
    for name, param in model.named_parameters():
        if param.requires_grad:
            # Les poids du circuit et du mapping doivent avoir changé
            assert not torch.equal(weights_before[name], param), (
                f"Les poids de '{name}' n'ont pas été mis à jour."
            )
        else:
            # Les poids de l'ACP doivent rester rigoureusement identiques
            assert torch.equal(weights_before[name], param), (
                f"Erreur : les poids gelés de '{name}' ont été modifiés !"
            )
