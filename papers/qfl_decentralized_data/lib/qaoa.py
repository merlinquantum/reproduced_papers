"""QAOA MaxCut task (experiment E3, paper Sec. III-B, Eqs. 3-6).

Data are random 3-regular graphs, one graph per sample. Every client holds some of the
graphs and all clients train the same ``(gamma, beta)`` parameters; the server
averages them. The loss is ``1 - <C> / C_max`` per graph, where ``<C>`` is the
expected cut and ``C_max`` the brute-force optimum, so it is 0 at the optimum. The
paper plots an un-normalised loss whose scale cannot be recovered from the text.
"""

from __future__ import annotations

from typing import Any

import networkx as nx
import torch
from torch import nn

from . import statevector as sv
from .data import dirichlet_partition, iid_partition
from .task import Task, TensorBatch


def cut_diagonal(graph: nx.Graph, n_nodes: int) -> torch.Tensor:
    """Cut value of every bit string; wire 0 is the most significant bit.

    ``C(z) = sum over edges {j, k} of (z_j xor z_k)``, which equals the paper's
    ``H_C = sum (1 - Z_j Z_k) / 2`` on computational basis states (Eq. 6).
    """
    indices = torch.arange(2**n_nodes)
    bits = [(indices >> (n_nodes - 1 - wire)) & 1 for wire in range(n_nodes)]
    cost = torch.zeros(2**n_nodes)
    for node_a, node_b in graph.edges:
        cost += (bits[node_a] ^ bits[node_b]).float()
    return cost


def generate_graph_batch(
    n_graphs: int, n_nodes: int, degree: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(cost_diagonals, max_cuts)`` for ``n_graphs`` random regular graphs."""
    diagonals = []
    for index in range(n_graphs):
        graph = nx.random_regular_graph(degree, n_nodes, seed=seed * 100003 + index)
        diagonals.append(cut_diagonal(graph, n_nodes))
    stacked = torch.stack(diagonals)
    return stacked, stacked.max(dim=1).values


class QAOAModel(nn.Module):
    """Depth-``P`` QAOA returning the expected cut of each graph in a batch.

    Parameters
    ----------
    n_nodes : int
        Number of qubits (graph nodes).
    layers : int
        Number of QAOA layers ``P``.
    init_value : float
        Initial value of every ``gamma`` and ``beta`` (the paper uses 0.01).
    dtype : torch.dtype
        Real dtype of the parameters.
    """

    def __init__(
        self, n_nodes: int, layers: int, init_value: float, dtype: torch.dtype
    ) -> None:
        super().__init__()
        self.n_nodes = n_nodes
        self.gamma = nn.Parameter(torch.full((layers,), init_value, dtype=dtype))
        self.beta = nn.Parameter(torch.full((layers,), init_value, dtype=dtype))

    def state(self, diagonal: torch.Tensor) -> torch.Tensor:
        """Final state ``prod_j exp(-i beta_j H_M) exp(-i gamma_j H_C) |+>^n``."""
        batch = diagonal.shape[0]
        amplitude = 2.0 ** (-self.n_nodes / 2)
        flat = torch.full(
            (batch, 2**self.n_nodes),
            amplitude,
            dtype=sv.complex_dtype_for(diagonal.dtype),
            device=diagonal.device,
        )
        for gamma, beta in zip(self.gamma, self.beta):
            flat = flat * torch.exp(-1j * gamma * diagonal)
            shaped = flat.reshape(batch, *([2] * self.n_nodes))
            mixer = sv.rx_matrix(2 * beta)  # exp(-i beta X) = RX(2 beta)
            for wire in range(self.n_nodes):
                shaped = sv.apply_one_qubit(shaped, mixer, wire)
            flat = shaped.reshape(batch, -1)
        return flat

    def forward(self, diagonal: torch.Tensor) -> torch.Tensor:
        """Expected cut ``<C>`` for each graph, shape ``(batch,)``."""
        amplitudes = self.state(diagonal)
        probabilities = amplitudes.real**2 + amplitudes.imag**2
        return (probabilities * diagonal).sum(dim=1)

    def optimal_probability(
        self, diagonal: torch.Tensor, max_cut: torch.Tensor
    ) -> torch.Tensor:
        """Probability mass on maximum-cut bit strings, shape ``(batch,)``."""
        amplitudes = self.state(diagonal)
        probabilities = amplitudes.real**2 + amplitudes.imag**2
        optimal = (diagonal >= max_cut[:, None] - 1e-6).to(probabilities.dtype)
        return (probabilities * optimal).sum(dim=1)


def build_qaoa_task(
    cfg: dict[str, Any], dtype: torch.dtype, device: torch.device
) -> Task:
    """Build the QAOA MaxCut task.

    Parameters
    ----------
    cfg : dict
        Resolved configuration; uses ``dataset.{n_nodes, degree, n_train_graphs,
        n_test_graphs, graph_seed}``, ``model.layers`` and ``model.init_value``.
    dtype : torch.dtype
        Real dtype of cost diagonals and parameters.
    device : torch.device
        Device for evaluation.

    Returns
    -------
    Task
        ``evaluate`` reports ``test_approximation_ratio`` and
        ``test_optimal_probability`` on fresh graphs, plus ``train_graphs_loss``.
    """
    dataset, model_cfg, federation = cfg["dataset"], cfg["model"], cfg["federation"]
    n_nodes = int(dataset["n_nodes"])
    seed = int(dataset["graph_seed"])
    train_diag, train_max = generate_graph_batch(
        int(dataset["n_train_graphs"]), n_nodes, int(dataset["degree"]), seed
    )
    # Test graphs use a different seed stream, so they are fresh draws.
    test_diag, test_max = generate_graph_batch(
        int(dataset["n_test_graphs"]), n_nodes, int(dataset["degree"]), seed + 1
    )
    train_diag, train_max = train_diag.to(dtype), train_max.to(dtype)
    test_diag, test_max = test_diag.to(dtype).to(device), test_max.to(dtype).to(device)

    generator = torch.Generator().manual_seed(seed)
    n_train = len(train_max)
    if federation["mode"] == "federated":
        n_clients = int(federation["n_clients"])
        if federation["partition"] == "noniid":
            parts = dirichlet_partition(
                torch.zeros(n_train, dtype=torch.long),
                n_clients,
                float(federation["dirichlet_alpha"]),
                generator,
                int(federation["min_client_size"]),
            )
        elif federation["partition"] == "iid":
            parts = iid_partition(n_train, n_clients, generator)
        else:
            raise ValueError(
                f"Unknown federation.partition {federation['partition']!r}"
            )
    else:
        parts = [torch.arange(n_train)]
    client_data = [(train_diag[part], train_max[part]) for part in parts]

    model = QAOAModel(
        n_nodes, int(model_cfg["layers"]), float(model_cfg["init_value"]), dtype
    ).to(device)

    def loss(model: nn.Module, batch: TensorBatch) -> torch.Tensor:
        diagonal, max_cut = batch
        return (1.0 - model(diagonal) / max_cut).mean()

    def evaluate(model: nn.Module) -> dict[str, float]:
        train_expected = model(train_diag.to(device))
        return {
            "train_graphs_loss": float(
                (1.0 - train_expected / train_max.to(device)).mean()
            ),
            "test_approximation_ratio": float((model(test_diag) / test_max).mean()),
            "test_optimal_probability": float(
                model.optimal_probability(test_diag, test_max).mean()
            ),
        }

    def final_report(model: nn.Module) -> dict[str, Any]:
        return {
            "gamma": model.gamma.detach().cpu().tolist(),
            "beta": model.beta.detach().cpu().tolist(),
            **evaluate(model),
        }

    return Task(
        model=model,
        client_data=client_data,
        loss=loss,
        evaluate=evaluate,
        primary_metric="test_approximation_ratio",
        final_report=final_report,
        summary={
            "n_nodes": n_nodes,
            "degree": int(dataset["degree"]),
            "train_graphs": n_train,
            "test_graphs": len(test_max),
            "client_sizes": [len(part) for part in parts],
        },
    )
