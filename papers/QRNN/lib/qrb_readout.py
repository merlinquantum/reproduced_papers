from __future__ import annotations

from dataclasses import dataclass

import merlin as ml
import torch
from merlin.core.partial_measurement import PartialMeasurement
from merlin.core.state_vector import StateVector


@dataclass(frozen=True)
class ReadoutSpec:
    """Readout specification for partial measurement outcomes."""

    k: int
    computation_space: ml.ComputationSpace


@dataclass(frozen=True)
class DualRailReadoutSpec:
    """Backward-compatible readout spec (dual-rail outcomes)."""

    k: int


def partial_measurement_to_probabilities(
    pm: PartialMeasurement,
    *,
    spec: ReadoutSpec,
) -> torch.Tensor:
    """Convert a PartialMeasurement into a probability vector in the chosen basis.

    Parameters
    ----------
    pm:
        Merlin PartialMeasurement over the first 2k modes (in detector order).
    spec:
        ReadoutSpec(k=..., computation_space=...).

    Returns
    -------
    torch.Tensor
        Tensor of shape (batch, out_dim) where out_dim depends on the computation space.

    Notes
    -----
    - Branch outcomes are tuples of length 2k with per-mode photon counts.
    - We use Combinadics for a canonical ordering:
        - dual_rail: Combinadics("dual_rail", n=k, m=2k) with out_dim=2**k
        - unbunched: Combinadics("unbunched", n=k, m=2k) with out_dim=binom(2k, k)
    - Multiple branches can share the same outcome; probabilities are accumulated.
    """

    k = int(spec.k)
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    computation_space = ml.ComputationSpace.coerce(spec.computation_space)
    if computation_space is ml.ComputationSpace.DUAL_RAIL:
        combinator = ml.Combinadics("dual_rail", n=k, m=2 * k)
        out_dim = 2**k
    elif computation_space is ml.ComputationSpace.UNBUNCHED:
        combinator = ml.Combinadics("unbunched", n=k, m=2 * k)
        out_dim = int(combinator.compute_space_size())
    else:
        raise ValueError(
            "This QRNN readout only supports computation_space in {dual_rail, unbunched}"
        )

    # Determine batch size from the first branch.
    if not pm.branches:
        raise ValueError("PartialMeasurement has no branches")

    prob0 = pm.branches[0].probability
    if prob0.ndim == 0:
        batch = 1
    else:
        batch = int(prob0.shape[0])

    probs = torch.zeros(batch, out_dim, dtype=prob0.dtype, device=prob0.device)

    for branch in pm.branches:
        outcome = branch.outcome
        if len(outcome) != 2 * k:
            raise ValueError(
                f"Expected outcome length {2 * k}, got {len(outcome)} (measured_modes={pm.measured_modes})"
            )
        idx = int(combinator.fock_to_index(outcome))
        p = branch.probability
        if p.ndim == 0:
            probs[0, idx] += p
        else:
            probs[:, idx] += p

    # Ensure normalization is well-behaved.
    total = probs.sum(dim=1, keepdim=True)
    safe = torch.where(total == 0, torch.ones_like(total), total)
    probs = probs / safe
    return probs


def partial_measurement_to_dual_rail_probabilities(
    pm: PartialMeasurement,
    *,
    spec: DualRailReadoutSpec,
) -> torch.Tensor:
    """Compatibility wrapper for the dual-rail readout."""

    return partial_measurement_to_probabilities(
        pm,
        spec=ReadoutSpec(
            k=int(spec.k), computation_space=ml.ComputationSpace.DUAL_RAIL
        ),
    )


def project_fock_amplitudes_to_dual_rail(
    state: StateVector,
    *,
    n_qubits: int,
) -> torch.Tensor:
    """Project a (remaining) Fock-basis StateVector onto the dual-rail subspace.

    Parameters
    ----------
    state:
        Merlin StateVector on 2*n_qubits modes with n_qubits photons (FOCK basis ordering).
    n_qubits:
        Number of logical qubits, i.e. photons in dual-rail.

    Returns
    -------
    torch.Tensor
        Complex amplitude tensor of shape (2**n_qubits,) or (batch, 2**n_qubits).

    Notes
    -----
    This is a convenience for early prototypes. It assumes the conditional state lives
    mostly in the dual-rail subspace (no bunching); any amplitude outside the subspace
    is discarded.
    """

    nq = int(n_qubits)
    if nq < 0:
        raise ValueError(f"n_qubits must be non-negative, got {nq}")

    if state.n_modes != 2 * nq:
        raise ValueError(f"Expected n_modes={2 * nq}, got {state.n_modes}")
    if state.n_photons != nq:
        raise ValueError(f"Expected n_photons={nq}, got {state.n_photons}")

    fock_basis = ml.Combinadics("fock", n=nq, m=2 * nq)
    dual_basis = ml.Combinadics("dual_rail", n=nq, m=2 * nq)

    dual_dim = 2**nq
    tensor = state.tensor

    if tensor.ndim == 1:
        out = torch.zeros(dual_dim, dtype=tensor.dtype, device=tensor.device)
        for dual_idx in range(dual_dim):
            counts = dual_basis.index_to_fock(dual_idx)
            fock_idx = fock_basis.fock_to_index(counts)
            out[dual_idx] = tensor[fock_idx]
        norm = torch.linalg.vector_norm(out)
        return out if norm == 0 else out / norm

    if tensor.ndim == 2:
        batch = tensor.shape[0]
        out = torch.zeros(batch, dual_dim, dtype=tensor.dtype, device=tensor.device)
        for dual_idx in range(dual_dim):
            counts = dual_basis.index_to_fock(dual_idx)
            fock_idx = fock_basis.fock_to_index(counts)
            out[:, dual_idx] = tensor[:, fock_idx]
        norm = torch.linalg.vector_norm(out, dim=1, keepdim=True)
        safe = torch.where(norm == 0, torch.ones_like(norm), norm)
        projected = out / safe
        if batch == 1:
            return projected[0]
        return projected

    raise ValueError(f"Unsupported StateVector tensor shape: {tuple(tensor.shape)}")


def kron_states(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Kronecker product for 1D (or batched 2D) vectors."""

    if left.ndim == 1 and right.ndim == 1:
        return torch.kron(left, right)

    if left.ndim == 2 and right.ndim == 2:
        if left.shape[0] != right.shape[0]:
            raise ValueError("Batch sizes must match for batched kron")
        # (B, A) ⊗ (B, C) -> (B, A*C)
        return (left[:, :, None] * right[:, None, :]).reshape(left.shape[0], -1)

    raise ValueError(
        f"Unsupported shapes for kron: left={tuple(left.shape)} right={tuple(right.shape)}"
    )
