# import sys
# from pathlib import Path

# # Ajoute le dossier contenant ce fichier de test au sys.path
# sys.path.insert(0, str(Path(__file__).parent.resolve()))

# import pytest
# import torch
# import torch.nn as nn
# from quantum_layer import PhotonicRewardModule


# # Importer la classe définie précédemment (adapte le nom du module)
# # from models import PhotonicRewardModule

# @pytest.fixture
# def model_and_data():
#     """Initialise le modèle et crée des tenseurs synthétiques de graphe moléculaire."""
#     batch_size = 4
#     num_atoms = 9
#     num_bonds = 4
#     num_atom_types = 5
#     nb_modes = 6
#     nb_photons = 2

#     model = PhotonicRewardModule(
#         num_atoms=num_atoms,
#         num_bonds=num_bonds,
#         num_atom_types=num_atom_types,
#         nb_modes=nb_modes,
#         nb_photons=nb_photons
#     )

#     # Simulation d'un batch de graphes sortant du générateur
#     adj_tensor = torch.randn(batch_size, num_atoms, num_atoms, num_bonds, requires_grad=True)
#     node_matrix = torch.randn(batch_size, num_atoms, num_atom_types, requires_grad=True)

#     # Simulation d'un reward RDKit cible Rc dans [0, 1]
#     target_rc = torch.rand(batch_size, 1)

#     return model, adj_tensor, node_matrix, target_rc


# def test_forward_shape_and_bounds(model_and_data):
#     """Vérifie que la sortie est un tenseur (batch_size, 1) borné entre 0 et 1."""
#     model, adj_tensor, node_matrix, _ = model_and_data

#     r_q = model(adj_tensor, node_matrix)

#     # 1. Vérification des dimensions
#     assert r_q.shape == (4, 1), f"Forme attendue (4, 1), obtenu {r_q.shape}"

#     # 2. Vérification des bornes du reward (Sigmoid)
#     assert torch.all(r_q >= 0.0) and torch.all(r_q <= 1.0), "RQ doit être dans l'intervalle [0, 1]"
#     assert not torch.isnan(r_q).any(), "La sortie contient des NaN"


# def test_backward_and_gradients(model_and_data):
#     """Vérifie le calcul de la loss et la présence de gradients non nuls à chaque étage."""
#     model, adj_tensor, node_matrix, target_rc = model_and_data

#     # 1. Forward
#     r_q = model(adj_tensor, node_matrix)

#     # 2. Loss d'alignement |RQ - RC|
#     loss = torch.mean(torch.abs(r_q - target_rc))

#     # 3. Backward
#     loss.backward()

#     # 4. Vérification que le gradient remonte jusqu'aux entrées du graphe
#     assert adj_tensor.grad is not None, "Le gradient n'atteint pas le tenseur d'adjacence"
#     assert node_matrix.grad is not None, "Le gradient n'atteint pas la matrice de nœuds"
#     assert torch.norm(adj_tensor.grad) > 0, "Le gradient de l'adjacence est nul"

#     # 5. Vérification des gradients sur les paramètres internes du modèle
#     # A. Encodeur amont
#     for name, param in model.encoder.named_parameters():
#         assert param.grad is not None, f"Pas de gradient pour {name} (encodeur)"
#         assert torch.norm(param.grad) > 0, f"Gradient nul pour {name} (encodeur)"

#     # B. Couche quantique MerLin
#     for name, param in model.quantum_layer.named_parameters():
#         assert param.grad is not None, f"Pas de gradient pour {name} (quantum layer)"
#         assert torch.norm(param.grad) > 0, f"Gradient nul pour {name} (quantum layer)"

#     # C. Mapping aval
#     for name, param in model.mapping.named_parameters():
#         assert param.grad is not None, f"Pas de gradient pour {name} (mapping)"
#         assert torch.norm(param.grad) > 0, f"Gradient nul pour {name} (mapping)"


# def test_optimizer_step(model_and_data):
#     """Vérifie que les poids du modèle se mettent bien à jour après une itération."""
#     model, adj_tensor, node_matrix, target_rc = model_and_data
#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

#     # Sauvegarde des poids avant mise à jour
#     params_before = [p.clone().detach() for p in model.parameters()]

#     # Étape d'optimisation
#     optimizer.zero_grad()
#     r_q = model(adj_tensor, node_matrix)
#     loss = torch.mean(torch.abs(r_q - target_rc))
#     loss.backward()
#     optimizer.step()

#     # Vérification que les poids ont changé
#     params_changed = any(
#         not torch.equal(p_before, p_after)
#         for p_before, p_after in zip(params_before, model.parameters())
#     )
#     assert params_changed, "Les paramètres du modèle n'ont pas été modifiés par l'optimiseur"


import numpy as np
import pytest
import torch
import torch.nn as nn

# A AJUSTER : Remplacez 'nom_de_votre_fichier' par le nom du fichier Python
# qui contient la classe PhotonicRewardModule
from quantum_layer import PhotonicRewardModule


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
