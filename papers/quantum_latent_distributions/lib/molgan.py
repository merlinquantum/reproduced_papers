"""MolGAN-style QM9 generation (paper Table II).

Appendix E: *"the MolGAN generator was modified by feeding affine transforms of
the latent code z to all layers of the feedforward network... The activation
functions used in the generator were also changed to LeakyReLU. Moreover,
instead of using 3 hidden layers, we used 5 hidden layers with sizes 64, 176,
288, 400 and 512."* The discriminator is *"a relational graph convolutional
network."*

Optional dependencies (rdkit, fcd-torch, torch-geometric) are imported lazily so
that the other three experiments run without them.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from lib.gan import WGANGPConfig
from torch import nn

logger = logging.getLogger(__name__)

__all__ = [
    "MAX_ATOMS",
    "N_ATOM_TYPES",
    "N_BOND_TYPES",
    "MolGANGenerator",
    "RelationalGCNCritic",
    "evaluate_molecules",
    "graphs_to_smiles",
    "load_qm9_dense",
    "train_molgan",
]

MAX_ATOMS = 9
N_ATOM_TYPES = 5  # C, N, O, F, padding
N_BOND_TYPES = 5  # none, single, double, triple, aromatic
GENERATOR_HIDDEN = (64, 176, 288, 400, 512)


class MolGANGenerator(nn.Module):
    """Generator emitting a dense bond tensor and an atom-type matrix.

    Hidden widths are non-decreasing, which is the paper's design constraint for
    Theorem 1 to apply, and an affine copy of ``z`` is added at every layer.

    Parameters
    ----------
    latent_dim : int
        Dimension of the latent code.
    tau : float
        Gumbel-softmax temperature. Default value is 1.0.
    """

    def __init__(self, latent_dim: int, tau: float = 1.0):
        super().__init__()
        self.tau = tau
        dims = (latent_dim, *GENERATOR_HIDDEN)
        self.blocks = nn.ModuleList(
            nn.Linear(dims[i], dims[i + 1]) for i in range(len(GENERATOR_HIDDEN))
        )
        self.skips = nn.ModuleList(
            nn.Linear(latent_dim, width, bias=False) for width in GENERATOR_HIDDEN
        )
        self.act = nn.LeakyReLU(0.2)
        self.edge_head = nn.Linear(
            GENERATOR_HIDDEN[-1], MAX_ATOMS * MAX_ATOMS * N_BOND_TYPES
        )
        self.node_head = nn.Linear(GENERATOR_HIDDEN[-1], MAX_ATOMS * N_ATOM_TYPES)

    def forward(
        self, z: torch.Tensor, hard: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate a batch of relaxed molecular graphs.

        Parameters
        ----------
        z : torch.Tensor
            Latent batch of shape ``(batch, latent_dim)``.
        hard : bool
            If True, use straight-through (discrete) Gumbel-softmax samples.
            Default value is False.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Bond tensor ``(batch, 9, 9, 5)`` and node tensor ``(batch, 9, 5)``.
        """
        hidden = z
        for block, skip in zip(self.blocks, self.skips):
            hidden = self.act(block(hidden) + skip(z))
        edges = self.edge_head(hidden).view(-1, MAX_ATOMS, MAX_ATOMS, N_BOND_TYPES)
        edges = (edges + edges.transpose(1, 2)) / 2  # molecular graphs are undirected
        nodes = self.node_head(hidden).view(-1, MAX_ATOMS, N_ATOM_TYPES)
        edges = nn.functional.gumbel_softmax(edges, tau=self.tau, hard=hard, dim=-1)
        nodes = nn.functional.gumbel_softmax(nodes, tau=self.tau, hard=hard, dim=-1)
        return edges, nodes


class RelationalGCNCritic(nn.Module):
    """Relational graph convolutional critic over the dense molecular graph.

    Parameters
    ----------
    hidden : tuple[int, ...]
        Widths of the relational convolution layers. Default value is (128, 64).
    agg : int
        Width of the gated graph-level aggregation. Default value is 128.
    """

    def __init__(self, hidden: tuple[int, ...] = (128, 64), agg: int = 128):
        super().__init__()
        self.relations = nn.ModuleList()
        width = N_ATOM_TYPES
        for out_width in hidden:
            self.relations.append(
                nn.ModuleList(nn.Linear(width, out_width) for _ in range(N_BOND_TYPES))
            )
            width = out_width
        self.self_loops = nn.ModuleList(
            nn.Linear(N_ATOM_TYPES if i == 0 else hidden[i - 1], out_width)
            for i, out_width in enumerate(hidden)
        )
        self.agg_gate = nn.Linear(width, agg)
        self.agg_value = nn.Linear(width, agg)
        self.head = nn.Sequential(nn.Linear(agg, 128), nn.Tanh(), nn.Linear(128, 1))

    def forward(self, edges: torch.Tensor, nodes: torch.Tensor) -> torch.Tensor:
        """Score a batch of molecular graphs.

        Parameters
        ----------
        edges : torch.Tensor
            Bond tensor of shape ``(batch, 9, 9, 5)``.
        nodes : torch.Tensor
            Node tensor of shape ``(batch, 9, 5)``.

        Returns
        -------
        torch.Tensor
            Critic scores of shape ``(batch, 1)``.
        """
        hidden = nodes
        for layer, loop in zip(self.relations, self.self_loops):
            message = loop(hidden)
            for relation in range(1, N_BOND_TYPES):  # relation 0 is "no bond"
                message = message + torch.einsum(
                    "bij,bjf->bif", edges[..., relation], layer[relation](hidden)
                )
            hidden = torch.tanh(message)
        gated = torch.sigmoid(self.agg_gate(hidden)) * self.agg_value(hidden)
        return self.head(gated.sum(dim=1))


def load_qm9_dense(cfg: dict[str, Any]):
    """Load QM9 into the dense 9-atom / 5-bond-type MolGAN representation.

    Parameters
    ----------
    cfg : dict
        Resolved configuration; ``data_root`` selects the download location.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, list[str]]
        Bond tensor, node tensor, and the reference SMILES list.
    """
    from torch_geometric.datasets import QM9
    from torch_geometric.utils import to_dense_adj

    root = f"{cfg['data_root']}/quantum_latent_distributions/qm9"
    dataset = QM9(root)
    atom_map = {6: 0, 7: 1, 8: 2, 9: 3}  # C N O F; index 4 is padding
    edge_list, node_list = [], []
    for molecule in dataset:
        if molecule.num_nodes > MAX_ATOMS:
            continue
        adjacency = to_dense_adj(
            molecule.edge_index,
            edge_attr=molecule.edge_attr,
            max_num_nodes=MAX_ATOMS,
        )[0]
        edges = torch.zeros(MAX_ATOMS, MAX_ATOMS, N_BOND_TYPES)
        edges[..., 1:] = adjacency
        edges[..., 0] = 1 - adjacency.sum(-1)
        nodes = torch.zeros(MAX_ATOMS, N_ATOM_TYPES)
        nodes[:, 4] = 1.0
        for i, atomic_number in enumerate(molecule.z.tolist()):
            if atomic_number in atom_map:
                nodes[i] = 0.0
                nodes[i, atom_map[atomic_number]] = 1.0
        edge_list.append(edges)
        node_list.append(nodes)

    edges = torch.stack(edge_list)
    nodes = torch.stack(node_list)
    reference = cfg["dataset"].get("reference_smiles", 20_000)
    train_smiles = [
        s for s in graphs_to_smiles(edges[:reference], nodes[:reference]) if s
    ]
    logger.info("QM9: %d graphs, %d reference SMILES", len(edges), len(train_smiles))
    return edges, nodes, train_smiles


def graphs_to_smiles(edges: torch.Tensor, nodes: torch.Tensor) -> list[str | None]:
    """Decode dense graphs to SMILES; invalid molecules decode to ``None``.

    Parameters
    ----------
    edges : torch.Tensor
        Bond tensor of shape ``(batch, 9, 9, 5)``.
    nodes : torch.Tensor
        Node tensor of shape ``(batch, 9, 5)``.

    Returns
    -------
    list[str | None]
        One entry per graph.
    """
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    atoms = ["C", "N", "O", "F", None]
    bond_types = [
        None,
        Chem.BondType.SINGLE,
        Chem.BondType.DOUBLE,
        Chem.BondType.TRIPLE,
        Chem.BondType.AROMATIC,
    ]
    edge_index = edges.argmax(-1).cpu().numpy()
    node_index = nodes.argmax(-1).cpu().numpy()

    decoded: list[str | None] = []
    for k in range(len(edge_index)):
        molecule = Chem.RWMol()
        positions = {}
        for i, atom_type in enumerate(node_index[k]):
            if atoms[atom_type] is not None:
                positions[i] = molecule.AddAtom(Chem.Atom(atoms[atom_type]))
        for i in positions:
            for j in positions:
                if i < j and bond_types[edge_index[k, i, j]] is not None:
                    molecule.AddBond(
                        positions[i], positions[j], bond_types[edge_index[k, i, j]]
                    )
        try:
            built = molecule.GetMol()
            Chem.SanitizeMol(built)
            decoded.append(Chem.MolToSmiles(built))
        except Exception:  # noqa: BLE001 - RDKit raises many sanitisation errors
            decoded.append(None)
    return decoded


def train_molgan(cfg, latent, edges, nodes, seed: int) -> MolGANGenerator:
    """Train a MolGAN generator/critic pair with WGAN-GP.

    Parameters
    ----------
    cfg : dict
        Resolved configuration.
    latent : merlin.LatentDistribution
        Latent distribution under test.
    edges : torch.Tensor
        Real bond tensors.
    nodes : torch.Tensor
        Real node tensors.
    seed : int
        Seed used for this repeat (already applied by the caller).

    Returns
    -------
    MolGANGenerator
        The trained generator.
    """
    device = torch.device(cfg.get("device", "cpu"))
    training = cfg["training"]
    wgan = WGANGPConfig(
        iterations=training["iterations"],
        batch_size=cfg["dataset"]["batch_size"],
        lr=training["lr"],
        optimizer=training.get("optimizer", "adam"),
        betas=tuple(training["betas"]),
        n_critic=training["n_critic"],
        gp_weight=training["gp_weight"],
        device=cfg.get("device", "cpu"),
    )
    edges, nodes = edges.to(device), nodes.to(device)
    generator = MolGANGenerator(cfg["latent"]["dim"]).to(device)
    critic = RelationalGCNCritic().to(device)
    opt_g = torch.optim.Adam(
        generator.parameters(), lr=wgan.lr, betas=tuple(wgan.betas)
    )
    opt_d = torch.optim.Adam(critic.parameters(), lr=wgan.lr, betas=tuple(wgan.betas))

    def gradient_penalty(real_e, real_n, fake_e, fake_n):
        alpha = torch.rand(real_e.size(0), 1, 1, 1, device=device)
        mixed_e = (alpha * real_e + (1 - alpha) * fake_e).requires_grad_(True)
        alpha_n = alpha.squeeze(-1)
        mixed_n = (alpha_n * real_n + (1 - alpha_n) * fake_n).requires_grad_(True)
        score = critic(mixed_e, mixed_n)
        grads = torch.autograd.grad(
            score, [mixed_e, mixed_n], torch.ones_like(score), create_graph=True
        )
        flat = torch.cat([g.reshape(g.size(0), -1) for g in grads], dim=1)
        return ((flat.norm(2, dim=1) - 1) ** 2).mean()

    for step in range(1, wgan.iterations + 1):
        for _ in range(wgan.n_critic):
            idx = torch.randint(0, edges.size(0), (wgan.batch_size,), device=device)
            real_e, real_n = edges[idx], nodes[idx]
            with torch.no_grad():
                fake_e, fake_n = generator(
                    latent.sample(wgan.batch_size, device=device)
                )
            d_loss = (
                critic(fake_e, fake_n).mean()
                - critic(real_e, real_n).mean()
                + wgan.gp_weight * gradient_penalty(real_e, real_n, fake_e, fake_n)
            )
            opt_d.zero_grad(set_to_none=True)
            d_loss.backward()
            opt_d.step()

        fake_e, fake_n = generator(latent.sample(wgan.batch_size, device=device))
        g_loss = -critic(fake_e, fake_n).mean()
        opt_g.zero_grad(set_to_none=True)
        g_loss.backward()
        opt_g.step()

        if step % max(wgan.iterations // 10, 1) == 0:
            logger.info(
                "molgan step %d d=%.3f g=%.3f",
                step,
                d_loss.detach().item(),
                g_loss.detach().item(),
            )
    return generator


def evaluate_molecules(cfg, generator, latent, train_smiles) -> dict[str, float]:
    """Compute the paper's Table II metrics for a trained generator.

    Parameters
    ----------
    cfg : dict
        Resolved configuration.
    generator : MolGANGenerator
        Trained generator.
    latent : merlin.LatentDistribution
        Latent distribution used during training.
    train_smiles : list[str]
        Reference SMILES for FCD and novelty.

    Returns
    -------
    dict
        ``fcd`` (lower is better), ``valid_and_unique`` and ``novel`` counts.
    """
    from fcd_torch import FCD

    device = torch.device(cfg.get("device", "cpu"))
    n_generate = cfg["evaluation"]["n_samples"]
    with torch.no_grad():
        edges, nodes = generator(latent.sample(n_generate, device=device), hard=True)
    generated = [s for s in graphs_to_smiles(edges, nodes) if s]
    unique = sorted(set(generated))
    novel = set(unique) - set(train_smiles)
    fcd = FCD(device=str(device))
    return {
        "fcd": float(fcd(unique, list(train_smiles)[: len(unique)])),
        "valid_and_unique": len(unique),
        "novel": len(novel),
    }
