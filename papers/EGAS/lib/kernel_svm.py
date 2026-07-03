"""Quantum-kernel SVM evaluation and baselines (Section IV.B, Appendix A.3).

QKSVM uses a precomputed fidelity kernel K_ij = F_Phi(x_i, x_j) with SVM regularisation
C = 0.05.  Baselines: classical linear SVM and RBF SVM (C=0.05, gamma=0.125) on z-scored
input features, the vanilla ZZ feature-map kernel, and NQE (ZZ map preceded by a trainable
neural preprocessing network trained with a fidelity loss).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .circuits import embed_states, zz_feature_states
from .statevec import fidelity_matrix

C_SVM = 0.05


def _kernel(states_a, states_b):
    return fidelity_matrix(states_a, states_b).cpu().numpy()


def qksvm_accuracy(
    seq_or_fn, X_train, y_train, X_test, y_test, n_qubits, bias=None, device="cpu"
):
    """Train QKSVM on a token sequence (or a callable X->states) and return test accuracy."""
    Xtr = torch.as_tensor(X_train, dtype=torch.float64, device=device)
    Xte = torch.as_tensor(X_test, dtype=torch.float64, device=device)
    if callable(seq_or_fn):
        st_tr, st_te = seq_or_fn(Xtr), seq_or_fn(Xte)
    else:
        st_tr = embed_states(seq_or_fn, Xtr, n_qubits, bias=bias)
        st_te = embed_states(seq_or_fn, Xte, n_qubits, bias=bias)
    with torch.no_grad():
        K_tr = _kernel(st_tr, st_tr)
        K_te = _kernel(st_te, st_tr)
    svc = SVC(kernel="precomputed", C=C_SVM)
    svc.fit(K_tr, y_train)
    return float((svc.predict(K_te) == y_test).mean())


def zz_accuracy(X_train, y_train, X_test, y_test, n_qubits, n_layers=1, device="cpu"):
    def fn(X):
        return zz_feature_states(X, n_qubits, n_layers=n_layers)

    return qksvm_accuracy(fn, X_train, y_train, X_test, y_test, n_qubits, device=device)


def classical_svm_accuracy(X_train, y_train, X_test, y_test, kind="linear"):
    """Compute SVM accuracy on test set. Supports binary and multiclass.

    For multiclass, uses one-vs-rest (OvR) strategy automatically.
    """
    scaler = StandardScaler().fit(X_train)
    Xtr, Xte = scaler.transform(X_train), scaler.transform(X_test)
    n_classes = len(np.unique(y_train))
    # Use OvR for multiclass, auto for binary
    decision_function_shape = "ovr" if n_classes > 2 else "ovr"
    if kind == "linear":
        svc = SVC(
            kernel="linear", C=C_SVM, decision_function_shape=decision_function_shape
        )
    else:
        svc = SVC(
            kernel="rbf",
            C=C_SVM,
            gamma=0.125,
            decision_function_shape=decision_function_shape,
        )
    svc.fit(Xtr, y_train)
    return float((svc.predict(Xte) == y_test).mean())


# ---------------------------------------------------------------------------
# NQE: trainable classical preprocessing feeding a fixed ZZ feature map (ref [19]).
# ---------------------------------------------------------------------------
class NQENet(nn.Module):
    def __init__(self, n_in, n_qubits, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_qubits),
        )
        self.n_qubits = n_qubits

    def forward(self, X):
        return self.net(X.to(torch.float64))


def train_nqe(
    X_train,
    y_train,
    n_qubits,
    *,
    epochs=80,
    batch_samples=25,
    lr=1e-3,
    seed=0,
    device="cpu",
):
    """Train NQE preprocessing with a pairwise fidelity (BCE) loss, then return the embedding fn."""
    torch.manual_seed(seed)
    Xt = torch.as_tensor(X_train, dtype=torch.float64, device=device)
    yt = torch.as_tensor(y_train, dtype=torch.long, device=device)
    net = NQENet(X_train.shape[1], n_qubits).double().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    n = len(X_train)
    for _ in range(epochs):
        idx = rng.choice(n, size=min(batch_samples, n), replace=False)
        Xb, yb = Xt[idx], yt[idx]
        feats = net(Xb)
        states = zz_feature_states(feats, n_qubits, n_layers=1)
        F = fidelity_matrix(states).clamp(1e-3, 1 - 1e-3)
        same = (yb.unsqueeze(0) == yb.unsqueeze(1)).double()
        off = ~torch.eye(len(idx), dtype=torch.bool, device=device)
        loss = -(same * torch.log(F) + (1 - same) * torch.log(1 - F))[off].mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    def embed_fn(X):
        return zz_feature_states(net(X), n_qubits, n_layers=1)

    return embed_fn


def nqe_accuracy(
    X_train, y_train, X_test, y_test, n_qubits, *, seed=0, device="cpu", **kw
):
    fn = train_nqe(X_train, y_train, n_qubits, seed=seed, device=device, **kw)
    return qksvm_accuracy(fn, X_train, y_train, X_test, y_test, n_qubits, device=device)
