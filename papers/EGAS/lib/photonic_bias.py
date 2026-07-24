"""Photonic continuous parameter refinement.

The current photonic encoder represents each data-driven PS gate phase as:

PS(theta_k * r + phi_k(x)

where ``theta_k`` is fed from ``x[..., data_idx]`` and ``phi_k(x)`` is produced by the encoder's
small bias MLP (``encoder.bias``). Refinement trains this bias network for a fixed circuit
structure.
"""

from __future__ import annotations

import merlin as ml
import numpy as np
import torch

from .egas import pairwise_energy
from .photonic_circuits import create_quantum_module
from .statevec import fidelity_matrix


def _bce_pair_loss(states, labels, eps=1e-3):
    F = fidelity_matrix(states)
    Fbar = F.clamp(eps, 1 - eps)
    same = (labels.unsqueeze(0) == labels.unsqueeze(1)).double()
    S = states.shape[0]
    off = ~torch.eye(S, dtype=torch.bool, device=states.device)
    bce = -(same * torch.log(Fbar) + (1 - same) * torch.log(1 - Fbar))
    return bce[off].mean()


def _parameter_l2(parameters):
    if not parameters:
        return torch.tensor(0.0)

    all_params = torch.cat([p.double().flatten() for p in parameters])
    return (all_params**2).mean()


def refine_bias(
    seq,
    X,
    y,
    n_modes,
    num_features,
    *,
    num_photons=2,
    computation_space=ml.ComputationSpace.UNBUNCHED,
    epochs=100,
    batch_samples=25,
    lr=5e-4,
    grad_clip=2.0,
    l2_bias=1e-6,
    hidden=32,
    gain=10.0,
    seed=0,
    device="cpu",
    avg_last=10,
):
    """Train the encoder's bias MLP for a fixed circuit `seq`

    Returns ``(encoder, E_before, E_after)``. The returned encoder is updated in place by
    optimising ``encoder.bias`` (which produces the per-PS phase offsets ``phi``).
    ``hidden`` and ``gain`` configure the bias MLP via ``encoder.reset_bias``.e.
    """
    torch.manual_seed(seed)
    Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    yt = torch.as_tensor(y, dtype=torch.long, device=device)

    encoder = create_quantum_module(
        seq,
        num_features=num_features,
        n_modes=n_modes,
        num_photons=num_photons,
        computation_space=computation_space,
    ).to(X.device)
    encoder.reset_bias(hidden=hidden, gain=gain)
    encoder = encoder.to(device)

    def _states_for_energy(inputs: torch.Tensor) -> torch.Tensor:
        """Embed every sample used to evaluate the pairwise-energy objective."""
        if len(encoder.ps_data_indices) == 0:
            # A circuit without data-dependent phases produces one fixed state.
            state = encoder(torch.zeros(1, 0, dtype=torch.float32, device=device))
            return state.expand(len(inputs), -1)
        return encoder(inputs)

    trainable_parameters = [p for p in encoder.bias.parameters() if p.requires_grad]
    if len(trainable_parameters) == 0:
        with torch.no_grad():
            states = _states_for_energy(Xt)
            E_before = pairwise_energy(states, yt).item()
        return encoder, E_before, E_before

    opt = torch.optim.RMSprop(encoder.bias.parameters(), lr=lr)

    with torch.no_grad():
        states = _states_for_energy(Xt)
        E_before = pairwise_energy(states, yt).item()

    rng = np.random.default_rng(seed)
    n = len(X)
    recent = []
    for ep in range(epochs):
        idx = rng.choice(n, size=min(batch_samples, n), replace=False)
        Xb, yb = Xt[idx], yt[idx]
        # Evaluate with a single sample and replicate if no input indices
        Xb_input = (
            torch.zeros(1, 0, dtype=torch.float32, device=device)
            if len(encoder.ps_data_indices) == 0
            else Xb
        )
        states = encoder(Xb_input)
        if len(encoder.ps_data_indices) == 0:
            states = states.repeat(len(Xb), 1)
        loss = _bce_pair_loss(states, yb)
        trainable_parameters = [p for p in encoder.bias.parameters() if p.requires_grad]
        reg = l2_bias * _parameter_l2(trainable_parameters)
        opt.zero_grad()
        (loss + reg).backward()
        torch.nn.utils.clip_grad_norm_(trainable_parameters, grad_clip)
        opt.step()
        if ep >= epochs - avg_last:
            with torch.no_grad():
                states = _states_for_energy(Xt)
                recent.append(pairwise_energy(states, yt).item())
    E_after = float(np.mean(recent)) if recent else E_before
    return encoder, E_before, E_after
