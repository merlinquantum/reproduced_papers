"""VQE for molecular hydrogen (experiment E4, paper Sec. III-C, Eqs. 8-9 and 20).

Each sample is one bond length with its own two-qubit Hamiltonian. The ansatz is the
one-parameter UCC state ``exp(-i t X0 Y1) |01> = cos(t) |01> - sin(t) |10>``.

A single shared angle cannot be optimal for every bond length, yet the paper's Fig. 6b
shows the variational parameter varying with bond length. The paper does not say how
the shared model produces that curve, so this reproduction makes the angle a function of
the bond length, ``t(R) = sum_k w_k T_k(x)`` with Chebyshev polynomials ``T_k`` of the
rescaled bond length ``x`` in [-1, 1]. The weights ``w_k`` are the federated parameters
(assumption recorded in LOG.md).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from .data import MAX_PARTITION_ATTEMPTS, iid_partition, sample_dirichlet
from .h2 import PAULI_ORDER, exact_energy_and_angle, h2_coefficients
from .task import Task, TensorBatch

_X = np.array([[0.0, 1.0], [1.0, 0.0]])
_Z = np.diag([1.0, -1.0])
_I = np.eye(2)
_Y = np.array([[0, -1j], [1j, 0]])
# Real 4x4 matrices for II, ZI, IZ, ZZ, XX, YY in the |q0 q1> basis (Y⊗Y is real).
PAULI_MATRICES = np.stack(
    [
        np.kron(_I, _I),
        np.kron(_Z, _I),
        np.kron(_I, _Z),
        np.kron(_Z, _Z),
        np.kron(_X, _X),
        np.kron(_Y, _Y).real,
    ]
)
assert len(PAULI_MATRICES) == len(PAULI_ORDER)


def chebyshev_features(
    bond_length: torch.Tensor, low: float, high: float, degree: int
) -> torch.Tensor:
    """``(batch, degree + 1)`` Chebyshev polynomials of the rescaled bond length."""
    x = 2 * (bond_length - low) / (high - low) - 1
    columns = [torch.ones_like(x), x]
    for _ in range(2, degree + 1):
        columns.append(2 * x * columns[-1] - columns[-2])
    return torch.stack(columns[: degree + 1], dim=1)


class VQEModel(nn.Module):
    """UCC ansatz whose angle is a Chebyshev series in the bond length.

    Parameters
    ----------
    degree : int
        Highest Chebyshev degree; there are ``degree + 1`` weights.
    dtype : torch.dtype
        Real dtype of the weights.
    """

    def __init__(self, degree: int, dtype: torch.dtype) -> None:
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(degree + 1, dtype=dtype))
        self.register_buffer("paulis", torch.from_numpy(PAULI_MATRICES).to(dtype))

    def angle(self, features: torch.Tensor) -> torch.Tensor:
        """UCC angle ``t(R)`` for each sample."""
        return features @ self.weights

    def forward(
        self, features: torch.Tensor, coefficients: torch.Tensor
    ) -> torch.Tensor:
        """Energy ``<psi(t)| H |psi(t)>`` per sample, shape ``(batch,)``."""
        angle = self.angle(features)
        zero = torch.zeros_like(angle)
        state = torch.stack([zero, torch.cos(angle), -torch.sin(angle), zero], dim=1)
        hamiltonian = torch.einsum("bk,kij->bij", coefficients, self.paulis)
        return torch.einsum("bi,bij,bj->b", state, hamiltonian, state)


def build_vqe_task(
    cfg: dict[str, Any], dtype: torch.dtype, device: torch.device
) -> Task:
    """Build the H2 VQE task over a grid of bond lengths.

    Parameters
    ----------
    cfg : dict
        Resolved configuration; uses ``dataset.{n_bond_lengths, bond_length_min,
        bond_length_max, data_seed}``, ``model.degree`` and ``federation.*``.
    dtype : torch.dtype
        Real dtype of coefficients and weights.
    device : torch.device
        Device for evaluation.

    Returns
    -------
    Task
        ``evaluate`` reports ``mean_energy_error`` and ``max_energy_error`` (hartree to
        the exact ground state) and ``mean_angle_error`` (radian, modulo pi) over all
        bond lengths.
    """
    dataset, model_cfg, federation = cfg["dataset"], cfg["model"], cfg["federation"]
    low, high = float(dataset["bond_length_min"]), float(dataset["bond_length_max"])
    bond_lengths = np.linspace(low, high, int(dataset["n_bond_lengths"]))
    coefficients = np.array([h2_coefficients(float(r)) for r in bond_lengths])
    exact = [exact_energy_and_angle(tuple(row)) for row in coefficients]
    exact_energy = torch.tensor([e for e, _ in exact], dtype=dtype, device=device)
    exact_angle = torch.tensor([a for _, a in exact], dtype=dtype, device=device)

    radii = torch.tensor(bond_lengths, dtype=dtype)
    degree = int(model_cfg["degree"])
    features = chebyshev_features(radii, low, high, degree)
    coefficient_tensor = torch.tensor(coefficients, dtype=dtype)

    generator = torch.Generator().manual_seed(int(dataset["data_seed"]))
    n_samples = len(bond_lengths)
    if federation["mode"] == "federated":
        n_clients = int(federation["n_clients"])
        if federation["partition"] == "noniid":
            # Contiguous bond-length ranges per client with unequal sizes: each client
            # sees a different part of the dissociation curve (non-i.i.d. and unbalanced).
            concentration = torch.full(
                (n_clients,), float(federation["dirichlet_alpha"])
            )
            for _ in range(MAX_PARTITION_ATTEMPTS):
                proportions = sample_dirichlet(concentration, generator)
                cuts = (proportions.cumsum(0)[:-1] * n_samples).long()
                parts = torch.tensor_split(torch.arange(n_samples), cuts.tolist())
                if min(len(part) for part in parts) >= int(
                    federation["min_client_size"]
                ):
                    break
            else:
                raise RuntimeError(
                    f"No Dirichlet bond-length split over {n_clients} clients gives every "
                    f"client at least {federation['min_client_size']} bond lengths"
                )
        elif federation["partition"] == "iid":
            parts = iid_partition(n_samples, n_clients, generator)
        else:
            raise ValueError(
                f"Unknown federation.partition {federation['partition']!r}"
            )
    else:
        parts = [torch.arange(n_samples)]
    client_data = [(features[part], coefficient_tensor[part]) for part in parts]

    model = VQEModel(degree, dtype).to(device)
    all_features, all_coefficients = features.to(device), coefficient_tensor.to(device)

    def loss(model: nn.Module, batch: TensorBatch) -> torch.Tensor:
        batch_features, batch_coefficients = batch
        return model(batch_features, batch_coefficients).mean()

    def evaluate(model: nn.Module) -> dict[str, float]:
        energy = model(all_features, all_coefficients)
        error = energy - exact_energy
        angle_gap = torch.remainder(model.angle(all_features) - exact_angle, np.pi)
        angle_error = torch.minimum(angle_gap, np.pi - angle_gap)
        return {
            "mean_energy_error": float(error.abs().mean()),
            "max_energy_error": float(error.abs().max()),
            "mean_angle_error": float(angle_error.mean()),
        }

    def final_report(model: nn.Module) -> dict[str, Any]:
        return {
            "bond_lengths_angstrom": bond_lengths.tolist(),
            "energies": model(all_features, all_coefficients).detach().cpu().tolist(),
            "exact_energies": exact_energy.cpu().tolist(),
            "angles": model.angle(all_features).detach().cpu().tolist(),
            "exact_angles": exact_angle.cpu().tolist(),
            **evaluate(model),
        }

    return Task(
        model=model,
        client_data=client_data,
        loss=loss,
        evaluate=evaluate,
        primary_metric="mean_energy_error",
        final_report=final_report,
        summary={
            "n_bond_lengths": n_samples,
            "bond_length_range_angstrom": [low, high],
            "client_sizes": [len(part) for part in parts],
        },
    )
