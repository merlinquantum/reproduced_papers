import pytest
import torch
import torch.nn.functional as F

# A AJUSTER : Importez vos classes depuis votre fichier (ex: gan_models.py)
from WGAN import Discriminator, Generator


@pytest.fixture
def gan_config():
    """Paramètres d'architecture calqués sur ton code source."""
    return {
        "z_dim": 8,
        "g_conv_dims": [128, 256, 512],
        "d_conv_dims": ([128, 64], 128, [128, 64]),
        "N_nodes": 9,
        "N_bonds": 5,
        "N_atoms": 5,
        "batch_size": 4,
        "tau": 1.0,  # Température Gumbel
    }


@pytest.fixture
def models(gan_config):
    """Instancie le Générateur et le Discriminateur."""
    G = Generator(
        gan_config["g_conv_dims"],
        gan_config["z_dim"],
        gan_config["N_nodes"],
        gan_config["N_bonds"],
        gan_config["N_atoms"],
        0.0,
    )
    D = Discriminator(
        gan_config["d_conv_dims"],
        m_dim=gan_config["N_atoms"],
        b_dim=gan_config["N_bonds"] - 1,
        dropout_rate=0.0,
    )
    return G, D


def test_forward_shapes(models, gan_config):
    """Vérifie que les dimensions circulent correctement de G vers D."""
    G, D = models
    z = torch.randn(gan_config["batch_size"], gan_config["z_dim"])

    # 1. Forward Générateur
    edges_logits, nodes_logits = G(z)

    assert edges_logits.shape == (
        gan_config["batch_size"],
        gan_config["N_nodes"],
        gan_config["N_nodes"],
        gan_config["N_bonds"],
    )
    assert nodes_logits.shape == (
        gan_config["batch_size"],
        gan_config["N_nodes"],
        gan_config["N_atoms"],
    )

    # 2. Gumbel-Softmax
    fake_adj = F.gumbel_softmax(edges_logits, tau=gan_config["tau"], hard=False, dim=-1)
    fake_nodes = F.gumbel_softmax(
        nodes_logits, tau=gan_config["tau"], hard=False, dim=-1
    )

    # 3. Forward Discriminateur
    d_out = D(fake_adj, None, fake_nodes)
    assert d_out.shape == (gan_config["batch_size"], 1), (
        f"Sortie D inattendue : {d_out.shape}"
    )


def test_discriminator_gradients(models, gan_config):
    """Vérifie que la loss WGAN met à jour D mais NE TOUCHE PAS G."""
    G, D = models
    batch_size = gan_config["batch_size"]

    real_adj = torch.randn(
        batch_size, gan_config["N_nodes"], gan_config["N_nodes"], gan_config["N_bonds"]
    )
    real_nodes = torch.randn(batch_size, gan_config["N_nodes"], gan_config["N_atoms"])
    z = torch.randn(batch_size, gan_config["z_dim"])

    edges_logits, nodes_logits = G(z)
    fake_adj = F.gumbel_softmax(edges_logits, tau=gan_config["tau"], hard=True, dim=-1)
    fake_nodes = F.gumbel_softmax(
        nodes_logits, tau=gan_config["tau"], hard=True, dim=-1
    )

    # Étape critique : .detach() empêche le gradient d'aller vers le Générateur
    real_val = D(real_adj, None, real_nodes)
    fake_val = D(fake_adj.detach(), None, fake_nodes.detach())

    d_loss = -torch.mean(real_val) + torch.mean(fake_val)
    d_loss.backward()

    # Vérifications des gradients
    for name, param in D.named_parameters():
        assert param.grad is not None, f"Gradient manquant pour D: {name}"
        assert torch.norm(param.grad) > 0, f"Gradient nul pour D: {name}"

    for name, param in G.named_parameters():
        assert param.grad is None, (
            f"Erreur fatale : G a reçu un gradient pendant l'entraînement de D ({name})"
        )


def test_generator_gradients(models, gan_config):
    """Vérifie que le gradient remonte correctement de D vers G."""
    G, D = models
    batch_size = gan_config["batch_size"]

    D.zero_grad()
    G.zero_grad()

    z = torch.randn(batch_size, gan_config["z_dim"])
    edges_logits, nodes_logits = G(z)

    # hard=False est indispensable ici pour laisser passer le gradient
    fake_adj = F.gumbel_softmax(edges_logits, tau=gan_config["tau"], hard=False, dim=-1)
    fake_nodes = F.gumbel_softmax(
        nodes_logits, tau=gan_config["tau"], hard=False, dim=-1
    )

    fake_val = D(fake_adj, None, fake_nodes)
    g_loss = -torch.mean(fake_val)
    g_loss.backward()

    # Vérification que le Générateur apprend
    for name, param in G.named_parameters():
        assert param.grad is not None, f"Gradient manquant pour G: {name}"
        assert torch.norm(param.grad) > 0, f"Gradient nul pour G: {name}"


def test_micro_convergence(models, gan_config):
    """Effectue 3 itérations complètes d'entraînement pour traquer les valeurs aberrantes (NaN)."""
    G, D = models
    opt_G = torch.optim.Adam(G.parameters(), lr=1e-4)
    opt_D = torch.optim.Adam(D.parameters(), lr=1e-4)
    batch_size = gan_config["batch_size"]

    real_adj = torch.randn(
        batch_size, gan_config["N_nodes"], gan_config["N_nodes"], gan_config["N_bonds"]
    )
    real_nodes = torch.randn(batch_size, gan_config["N_nodes"], gan_config["N_atoms"])

    # On fige le bruit d'entrée pour tester l'adaptation des poids sur un batch unique
    z = torch.randn(batch_size, gan_config["z_dim"])

    for _ in range(3):
        # 1. Update D
        opt_D.zero_grad()
        edges, nodes = G(z)
        fake_adj = F.gumbel_softmax(edges, tau=gan_config["tau"], hard=True, dim=-1)
        fake_nodes = F.gumbel_softmax(nodes, tau=gan_config["tau"], hard=True, dim=-1)

        d_loss = -torch.mean(D(real_adj, None, real_nodes)) + torch.mean(
            D(fake_adj.detach(), None, fake_nodes.detach())
        )
        d_loss.backward()
        opt_D.step()

        # 2. Update G
        opt_G.zero_grad()
        edges, nodes = G(z)
        fake_adj_g = F.gumbel_softmax(edges, tau=gan_config["tau"], hard=False, dim=-1)
        fake_nodes_g = F.gumbel_softmax(
            nodes, tau=gan_config["tau"], hard=False, dim=-1
        )

        g_loss = -torch.mean(D(fake_adj_g, None, fake_nodes_g))
        g_loss.backward()
        opt_G.step()

    # La métrique critique : l'entraînement n'a pas explosé
    assert not torch.isnan(d_loss), (
        "Explosion numérique (NaN) dans la loss du Discriminateur."
    )
    assert not torch.isnan(g_loss), (
        "Explosion numérique (NaN) dans la loss du Générateur."
    )
