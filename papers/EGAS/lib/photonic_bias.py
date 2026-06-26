"""Photonic continuous parameter refinement.

The photonic encoder already gives every data-driven PS gate a trainable phase offset:

    PS(theta_k * r + phi_k)

where ``theta_k`` is fed from ``x[..., data_idx]`` and ``phi_k`` is registered as a MerLin
trainable parameter.  Refinement therefore trains the encoder's ``phi_k`` parameters directly
instead of adding a separate bias MLP.
"""

from __future__ import annotations

import merlin as ml
import numpy as np
import torch

from .egas import pairwise_energy
from .statevec import fidelity_matrix
from .photonic_circuits import create_quantum_module


def _bce_pair_loss(states, labels, eps=1e-3):
    F = fidelity_matrix(states)
    Fbar = F.clamp(eps, 1 - eps)
    same = (labels.unsqueeze(0) == labels.unsqueeze(1)).double()
    S = states.shape[0]
    off = ~torch.eye(S, dtype=torch.bool, device=states.device)
    bce = -(same * torch.log(Fbar) + (1 - same) * torch.log(1 - Fbar))
    return bce[off].mean()


def _parameter_l2(parameters):
    reg = None
    for param in parameters:
        term = (param.double() ** 2).mean()
        reg = term if reg is None else reg + term
    return reg if reg is not None else torch.tensor(0.0)


def refine_bias(
    seq,
    X,
    y,
    n_modes,
    *,
    num_photons=2,
    computation_space=ml.ComputationSpace.UNBUNCHED,
    epochs=100,
    batch_samples=25,
    lr=5e-4,
    grad_clip=2.0,
    l2_bias=1e-6,
    hidden=None,
    gain=None,
    seed=0,
    device="cpu",
    avg_last=10,
):
    """Train encoder phase offsets for a fixed circuit `seq`.

    Returns ``(encoder, E_before, E_after)``.  The returned encoder has its MerLin
    trainable PS offsets refined in place.  ``hidden`` and ``gain`` are accepted for
    compatibility with the MLP-bias path and are intentionally unused here.
    """
    torch.manual_seed(seed)
    Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    yt = torch.as_tensor(y, dtype=torch.long, device=device)

    encoder = create_quantum_module(
        seq,
        n_modes=n_modes,
        num_photons=num_photons,
        computation_space=computation_space,
    ).to(device)
    trainable_parameters = [p for p in encoder.parameters() if p.requires_grad]
    if not trainable_parameters:
        with torch.no_grad():
            E_before = pairwise_energy(encoder(Xt), yt).item()
        return encoder, E_before, E_before

    opt = torch.optim.RMSprop(trainable_parameters, lr=lr)

    with torch.no_grad():
        E_before = pairwise_energy(encoder(Xt), yt).item()

    rng = np.random.default_rng(seed)
    n = len(X)
    recent = []
    for ep in range(epochs):
        idx = rng.choice(n, size=min(batch_samples, n), replace=False)
        Xb, yb = Xt[idx], yt[idx]
        states = encoder(Xb)
        loss = _bce_pair_loss(states, yb)
        reg = l2_bias * _parameter_l2(trainable_parameters)
        opt.zero_grad()
        (loss + reg).backward()
        torch.nn.utils.clip_grad_norm_(trainable_parameters, grad_clip)
        opt.step()
        if ep >= epochs - avg_last:
            with torch.no_grad():
                recent.append(pairwise_energy(encoder(Xt), yt).item())
    E_after = float(np.mean(recent)) if recent else E_before
    return encoder, E_before, E_after
