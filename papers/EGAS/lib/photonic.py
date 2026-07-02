"""MerLin photonic counterpart of the EGAS quantum embedding (Phase 4).

The paper is gate-based: EGAS searches a discrete gate-circuit embedding and evaluates it via
a fidelity quantum-kernel SVM, ``K_ij = |<psi(x_i)|psi(x_j)>|^2``.  The photonic counterpart
preserves that scientific role:

* a *photonic* embedding maps each input to a multi-photon state on a linear-optical mesh
  (angle encoding + trainable interferometer),
* the same fidelity kernel is computed photonically with MerLin's ``FidelityKernel``
  (``|<s|U^dag(x2) U(x1)|s>|^2`` via SLOS), and
* downstream classification uses the identical precomputed-kernel SVM (C=0.05).

Two variants:
* **fixed**   — random/initial mesh (data-agnostic photonic embedding).
* **trained** — mesh parameters optimised with the EGAS pairwise-fidelity surrogate
  (Eq. 4): same-class overlaps up, cross-class overlaps down. This is the continuous,
  photonic analogue of EGAS's discrete architecture search.

Hardware-aware settings: >=2 photons (single-photon linear optics is a trivial classical
baseline and is not used), UNBUNCHED computation space (threshold detectors), analytic SLOS
(shots=None).
"""

from __future__ import annotations

import merlin as ml
import numpy as np
import torch
from sklearn.svm import SVC

C_SVM = 0.05


def build_feature_map(n_modes: int, n_layers: int = 2, scale: float = 1.0):
    """Angle-encoding photonic feature map with trainable entangling meshes."""
    b = ml.CircuitBuilder(n_modes=n_modes)
    b.add_entangling_layer()
    b.add_angle_encoding(scale=scale)
    for _ in range(n_layers - 1):
        b.add_entangling_layer()
    ip = b.input_parameter_prefixes
    tp = b.trainable_parameter_prefixes
    ip = ip() if callable(ip) else ip
    tp = tp() if callable(tp) else tp
    return ml.FeatureMap(
        builder=b, input_size=n_modes, input_parameters=ip, trainable_parameters=tp
    )


def default_input_state(n_modes: int, n_photons: int):
    """Spread n_photons across modes (alternating), ensuring >=2 photons."""
    assert n_photons >= 2, "photonic reproduction requires >=2 photons"
    state = [0] * n_modes
    placed = 0
    i = 0
    while placed < n_photons and i < n_modes:
        state[i] = 1
        placed += 1
        i += 2
    # if not enough room with stride 2, fill remaining low modes
    j = 0
    while placed < n_photons:
        if state[j] == 0:
            state[j] = 1
            placed += 1
        j += 1
    return state


def make_kernel(n_modes, n_photons, n_layers=2, scale=1.0, device="cpu"):
    fm = build_feature_map(n_modes, n_layers=n_layers, scale=scale)
    state = default_input_state(n_modes, n_photons)
    kern = ml.FidelityKernel(
        fm,
        input_state=state,
        n_photons=n_photons,
        computation_space="unbunched",
        device=device,
    )
    return kern, state


def _pair_fidelity_loss(K, y, eps=1e-4):
    same = (y.unsqueeze(0) == y.unsqueeze(1)).double()
    off = ~torch.eye(len(y), dtype=torch.bool)
    return (same - K.double()).abs()[off].mean()


def train_photonic_embedding(
    kern, X_train, y_train, *, epochs=120, batch=36, lr=0.05, seed=0
):
    """Optimise the photonic mesh parameters with the EGAS pairwise-fidelity surrogate."""
    torch.manual_seed(seed)
    params = [p for p in kern.parameters() if p.requires_grad]
    if not params:
        return kern  # nothing trainable
    opt = torch.optim.Adam(params, lr=lr)
    Xt = torch.as_tensor(X_train, dtype=torch.float32)
    yt = torch.as_tensor(y_train, dtype=torch.long)
    rng = np.random.default_rng(seed)
    n = len(X_train)
    for _ in range(epochs):
        idx = rng.choice(n, size=min(batch, n), replace=False)
        K = kern.forward(Xt[idx])
        loss = _pair_fidelity_loss(K, yt[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    return kern


def photonic_qksvm_accuracy(kern, X_train, y_train, X_test, y_test):
    Xtr = torch.as_tensor(X_train, dtype=torch.float32)
    Xte = torch.as_tensor(X_test, dtype=torch.float32)
    with torch.no_grad():
        K_tr = kern.forward(Xtr)
        K_te = kern.forward(Xte, Xtr)
    K_tr = K_tr.detach().cpu().numpy()
    K_te = K_te.detach().cpu().numpy()
    svc = SVC(kernel="precomputed", C=C_SVM)
    svc.fit(K_tr, y_train)
    return float((svc.predict(K_te) == y_test).mean())
