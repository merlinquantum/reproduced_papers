"""Photonic quantum-kernel SVM evaluation.

QKSVM uses a precomputed fidelity kernel K_ij = F_Phi(x_i, x_j) with SVM regularisation
C = 0.05.  The photonic embedding is supplied as an already-built model/callable that maps
input features to output amplitudes/states.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .statevec import fidelity_matrix

C_SVM = 0.05


def _kernel(states_a, states_b):
    return fidelity_matrix(states_a, states_b).cpu().numpy()


def qksvm_accuracy(photonic_model, X_train, y_train, X_test, y_test, device="cpu"):
    """Train QKSVM from an already-built photonic model and return test accuracy."""
    Xtr = torch.as_tensor(X_train, dtype=torch.float32, device=device)
    Xte = torch.as_tensor(X_test, dtype=torch.float32, device=device)

    if isinstance(photonic_model, torch.nn.Module):
        photonic_model = photonic_model.to(device)
        photonic_model.eval()

    with torch.no_grad():
        st_tr = photonic_model(Xtr)
        st_te = photonic_model(Xte)
        K_tr = _kernel(st_tr, st_tr)
        K_te = _kernel(st_te, st_tr)
    svc = SVC(kernel="precomputed", C=C_SVM)
    svc.fit(K_tr, y_train)
    return float((svc.predict(K_te) == y_test).mean())


def classical_svm_accuracy(X_train, y_train, X_test, y_test, kind="linear"):
    """Compute SVM accuracy on test set. Supports binary and multiclass.

    For multiclass, uses one-vs-rest (OvR) strategy automatically.
    """
    scaler = StandardScaler().fit(X_train)
    Xtr, Xte = scaler.transform(X_train), scaler.transform(X_test)
    # Use OvR for multiclass, auto for binary
    decision_function_shape = "ovr"
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
