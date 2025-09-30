"""
===================================================================================================
Re‑uploading Quantum Classifiers (Merlin + Perceval)
— 1D Fisher (DeepLDA‑style) Training + Paper‑Style LDA Head
===================================================================================================

This module implements the photonic data re‑uploading architecture with a training objective
inspired by **Deep Linear Discriminant Analysis** (DeepLDA) and an LDA‑style classifier that
matches the paper's decision rule.

The pipeline is intentionally minimal and paper‑consistent:

    Input x → Quantum Circuit (PS/BS/MZI re‑uploading) → 1D feature x_feat = p10
                                              ↓
                              Train with 1D Fisher/DeepLDA loss
                                              ↓
                          Freeze circuit → LDA head (means‑based rule)

Why 1D? With a single photon over two modes (lossless setting), the measured probabilities obey
$$
p_{01} + p_{10} = 1.
$$
Thus the learned hidden representation is effectively one‑dimensional. We use
$$
x_{\\mathrm{feat}} := p_{10} \\in [0,1]
$$
as the topmost feature.

---------------------------------------------------------------------------------------------------
KEY ASSUMPTIONS
---------------------------------------------------------------------------------------------------

1. Single‑photon, two‑mode measurement ⇒ \\(p_{01}+p_{10}=1\\) ⇒ the representation is **1D**.
2. Loss is the 1D log‑ratio Fisher objective above (DeepLDA‑inspired), applied to \\(x=p_{10}\\).
3. Classifier head is means‑based LDA in 1D (no extra learnable head).
4. In Merlin, gradients flow via autodiff; in Perceval, via COBYLA or Adam+PSR.
5. Constructor fixes architecture only; training hyperparameters belong to `fit(...)`.

For more details about the loss function and the classifier head see the associated jupyternotebook

---------------------------------------------------------------------------------------------------
BACKENDS
---------------------------------------------------------------------------------------------------

• Merlin (autograd)
  The quantum layer is exposed as a PyTorch module producing probabilities \\([p_{01},p_{10}]\\).
  We take \\(x=p_{10}\\) and optimize \\(\\mathcal{L}_{\\text{Fisher}}\\) via Adam using PyTorch
  automatic differentiation end‑to‑end.

• Perceval (simulation)
  We extract \\(x=p_{10}\\) by sampling the circuit and train with either:
  - COBYLA (gradient‑free), or
  - Adam + Parameter‑Shift Rule (PSR): analytic gradients from shifted parameter evaluations.

Both backends share the same loss and the same LDA head.

---------------------------------------------------------------------------------------------------
API (architecture‑only __init__, training hyperparams in fit)
---------------------------------------------------------------------------------------------------

    # Merlin
    model = MerlinReuploadingClassifier(
        dimension=X_train.shape[1],
        num_layers=NUM_LAYERS,
    )
    model.fit(
        X_train, y_train,
        track_history=True,
        max_epochs=...,
        learning_rate=...,
        batch_size=...,
        patience=...,
        tau=...,
    )

    # Perceval
    model = PercevalReuploadingClassifier(
        dimension=X_train.shape[1],
        num_layers=NUM_LAYERS,
    )
    model.fit(
        X_train, y_train,
        track_history=True,
        optimizer="cobyla"  # or "adam_psr"
        max_epochs=...,
        tau=...,
        n_shots=...,
        # if optimizer == "adam_psr": lr=..., psr_batch_size=..., etc.
    )

Common helpers:
- `predict(X)`, `predict_proba(X)`, `score(X, y)`
- `get_quantum_features(X)` → returns the learned 1D feature \\(x=p_{10}\\)
- `training_history_` (if `track_history=True`) stores the per‑epoch loss.

---------------------------------------------------------------------------------------------------
REFERENCES
---------------------------------------------------------------------------------------------------

- Dorfer, Kelz, Widmer — *Deep Linear Discriminant Analysis*, ICLR 2016 (workshop).
- Kingma, Ba — *Adam: A Method for Stochastic Optimization*, ICLR 2015.
- Schuld et al. — *Evaluating analytic gradients on quantum hardware* (Parameter‑Shift Rule).

===================================================================================================
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

# Quantum backends
import merlin as ml  # Merlin autograd-compatible backend
import numpy as np
import perceval as pcvl  # Perceval simulator

# Torch
import torch
import torch.nn as nn
import torch.optim as optim
from perceval.algorithm import Sampler
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score

# -------------------------------------------------------------------------------------------------
# Fisher (DeepLDA-style) Loss — 1D-friendly (works for any D, but we use D=1)
# -------------------------------------------------------------------------------------------------


def fisher_logratio_loss(
    features: torch.Tensor,  # (N, D) — D=1 in our use
    labels: torch.Tensor,  # (N,)
    tau: float = 1.0,
    eps: float = 1e-12,
    scheme: str = "macro",  # "macro" (default) or "micro"
) -> torch.Tensor:
    """
    Scale-invariant, class-imbalance-robust Fisher/DeepLDA-style loss.

      L = -[ log(Sb + eps) - tau * log(Sw + eps) ]

    Sw = mean_c Var(X_c)         (unbiased, per class; equal weight per class)
    Sb = Var_c(mean(X_c))        (variance of class means; equal weight per class)
    """
    if features.dim() != 2 or labels.dim() != 1 or features.size(0) != labels.size(0):
        raise ValueError("Expected features (N,D) and labels (N,) with matching N.")
    device = features.device
    dtype = features.dtype

    classes = torch.unique(labels)
    if classes.numel() < 2:
        return torch.zeros((), device=device, dtype=dtype)

    # --- Per-class unbiased variance; handle small classes gracefully
    class_vars = []
    class_means = []
    for c in classes:
        Xc = features[labels == c]  # (n_c, D)
        if Xc.shape[0] >= 2:
            var_c = torch.var(Xc, dim=0, unbiased=True)  # (D,)
        elif Xc.shape[0] == 1:
            var_c = torch.zeros(features.shape[1], device=device, dtype=dtype)
        else:
            # no samples for this class in the batch; skip
            continue
        mu_c = (
            torch.mean(Xc, dim=0)
            if Xc.shape[0] > 0
            else torch.zeros(features.shape[1], device=device, dtype=dtype)
        )
        class_vars.append(var_c)
        class_means.append(mu_c)

    if len(class_means) < 2:
        # not enough classes present in this minibatch
        return torch.zeros((), device=device, dtype=dtype)

    # Sw = mean of per-class variances (sum over dims then average classes)
    Sw = torch.stack([v.sum() for v in class_vars]).mean()

    # Sb = variance of class means (unweighted, across classes)
    M = torch.stack(class_means, dim=0)  # (C_eff, D)
    mu_bar = M.mean(dim=0)  # (D,)
    Sb = (
        ((M - mu_bar) ** 2).sum(dim=1).mean()
    )  # mean over classes of squared L2 distance

    return -(torch.log(Sb + eps) - tau * torch.log(Sw + eps))


def fisher_logratio_loss_numpy(
    x: np.ndarray,  # (N, D), use D=1
    y: np.ndarray,  # (N,)
    tau: float = 1.0,
    eps: float = 1e-12,
) -> float:
    """
    NumPy twin of the PyTorch loss, with the same normalization choices.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    y = np.asarray(y)
    classes = np.unique(y)
    if classes.size < 2:
        return 0.0

    class_vars = []
    class_means = []
    for c in classes:
        Xc = x[y == c]  # (n_c, D)
        if Xc.shape[0] >= 2:
            var_c = Xc.var(axis=0, ddof=1)  # unbiased
        elif Xc.shape[0] == 1:
            var_c = np.zeros(x.shape[1], dtype=float)
        else:
            continue
        mu_c = Xc.mean(axis=0) if Xc.shape[0] > 0 else np.zeros(x.shape[1], dtype=float)
        class_vars.append(var_c)
        class_means.append(mu_c)

    if len(class_means) < 2:
        return 0.0

    Sw = np.mean([v.sum() for v in class_vars])  # mean over classes (sum over dims)
    M = np.vstack(class_means)  # (C_eff, D)
    mu_bar = M.mean(axis=0)
    Sb = np.mean(np.sum((M - mu_bar) ** 2, axis=1))  # mean over classes

    return float(-(np.log(Sb + eps) - tau * np.log(Sw + eps)))


# -------------------------------------------------------------------------------------------------
# Paper-Style LDA Head in 1D (no learnable params)
# -------------------------------------------------------------------------------------------------


class LDA1DHead:
    """
    Means-based LDA rule in 1D:
        d_c(x) = m_c * x - 0.5 * m_c^2
    where m_c is the per-class mean of the training features.
    """

    def fit(self, x_train: np.ndarray, y_train: np.ndarray):
        x = np.asarray(x_train).reshape(-1)
        y = np.asarray(y_train)
        self.classes_ = np.unique(y)
        self.means_ = np.array([x[y == c].mean() for c in self.classes_], dtype=float)
        return self

    def scores(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x).reshape(-1, 1)  # (N,1)
        m = self.means_.reshape(1, -1)  # (1,C)
        return m * x - 0.5 * (m**2)  # (N,C)

    def predict(self, x: np.ndarray) -> np.ndarray:
        d = self.scores(x)
        return self.classes_[np.argmax(d, axis=1)]

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        d = self.scores(x)
        p = 1.0 / (1.0 + np.exp(-d))
        return p / (np.sum(p, axis=1, keepdims=True) + 1e-12)


# -------------------------------------------------------------------------------------------------
# Photonic Circuit Model for Perceval (2 modes, p10 feature)
# -------------------------------------------------------------------------------------------------

# Assume "pcvl" and "np" are already imported in your module


class ReUploadCircuitModelTwoMode:
    """
    Two-mode re-uploading photonic circuit with pluggable block designs.

    design: two chars in 'A,B,C', e.g. 'AA', 'BC', ...
      - first char:  Data block type (A|B|C)
      - second char: Trainable/variable block type (A|B|C)

    Data blocks (consume up to two data features per block):
      A: PS_0(x_i) ; BS_0() ; PS_0(x_{i+1}) ; BS_0()
      B: PS_0(x_i) ; BS_0(x_{i+1})
      C: PS_0(x_i) ; PS_1(x_{i+1}) ; BS_0()

    Var blocks (consume 2 trainable params per block):
      A: PS_0(θ_k) ; BS_0() ; PS_0(φ_k) ; BS_0()
      B: PS_0(θ_k) ; BS_0(φ_k)
      C: PS_0(θ_k) ; PS_1(φ_k) ; BS_0()

    """

    def __init__(self, dimension: int, num_layers: int, design: str = "AA"):
        if dimension <= 0:
            raise ValueError(f"dimension must be positive, got {dimension}")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        if not (
            isinstance(design, str)
            and len(design) == 2
            and all(c in "ABC" for c in design)
        ):
            raise ValueError(
                "design must be a two-character string with letters in 'ABC', e.g. 'AA', 'BC'."
            )

        self.dimension = dimension
        self.num_layers = num_layers
        self.design = design.upper()
        self.input_state = pcvl.BasicState([0, 1])

        # One block handles up to two data features
        self.num_rotation_blocks_per_layer = math.ceil(self.dimension / 2)

        # Parameters
        self.num_data_params = self.dimension
        self.data_params = [
            pcvl.Parameter(f"x_{i}") for i in range(self.num_data_params)
        ]

        # 2 trainable params per block, regardless of A|B|C (see patterns below)
        self.num_var_params = 2 * self.num_rotation_blocks_per_layer * self.num_layers
        self.var_params = [
            pcvl.Parameter(f"var_{i}") for i in range(self.num_var_params)
        ]

        # Build circuit
        self.circuit = self._build_circuit()

    # ---------- Block pattern definitions ----------
    # Each function returns the new (data_idx, var_idx) after adding its gates.

    def _data_A(self, circuit, data_idx: int, var_idx: int) -> tuple[int, int]:
        # A: PS_0(x_i) ; BS() ; PS_0(x_{i+1}) ; BS()   (if i+1 exists), else PS_0(x_i) ; BS()
        circuit.add(0, pcvl.PS(self.data_params[data_idx]), merge=True)
        circuit.add(0, pcvl.BS(), merge=True)
        if data_idx + 1 < self.dimension:
            circuit.add(0, pcvl.PS(self.data_params[data_idx + 1]), merge=True)
            circuit.add(0, pcvl.BS(), merge=True)
            data_idx += 2
        else:
            data_idx += 1
        return data_idx, var_idx

    def _data_B(self, circuit, data_idx: int, var_idx: int) -> tuple[int, int]:
        # B: PS_0(x_i) ; BS_0(x_{i+1})   (if i+1 exists), else PS_0(x_i) ; BS_0()
        circuit.add(0, pcvl.PS(self.data_params[data_idx]), merge=True)
        if data_idx + 1 < self.dimension:
            circuit.add(0, pcvl.BS(self.data_params[data_idx + 1]), merge=True)
            data_idx += 2
        else:
            circuit.add(0, pcvl.BS(), merge=True)
            data_idx += 1
        return data_idx, var_idx

    def _data_C(self, circuit, data_idx: int, var_idx: int) -> tuple[int, int]:
        # C: PS_0(x_i) ; PS_1(x_{i+1}) ; BS()   (if i+1 exists), else PS_0(x_i) ; BS()
        circuit.add(0, pcvl.PS(self.data_params[data_idx]), merge=True)
        if data_idx + 1 < self.dimension:
            circuit.add(1, pcvl.PS(self.data_params[data_idx + 1]), merge=True)
            data_idx += 2
        else:
            data_idx += 1
        circuit.add(0, pcvl.BS(), merge=True)
        return data_idx, var_idx

    def _var_A(self, circuit, data_idx: int, var_idx: int) -> tuple[int, int]:
        # A: PS_0(θ_k) ; BS() ; PS_0(φ_k) ; BS()
        circuit.add(0, pcvl.PS(self.var_params[var_idx]), merge=True)
        circuit.add(0, pcvl.BS(), merge=True)
        circuit.add(0, pcvl.PS(self.var_params[var_idx + 1]), merge=True)
        circuit.add(0, pcvl.BS(), merge=True)
        var_idx += 2
        return data_idx, var_idx

    def _var_B(self, circuit, data_idx: int, var_idx: int) -> tuple[int, int]:
        # B: PS_0(θ_k) ; BS_0(φ_k)   (two trainable params)
        circuit.add(0, pcvl.PS(self.var_params[var_idx]), merge=True)
        circuit.add(
            0, pcvl.BS(self.var_params[var_idx + 1]), merge=True
        )  # <--- adjust signature if needed
        var_idx += 2
        return data_idx, var_idx

    def _var_C(self, circuit, data_idx: int, var_idx: int) -> tuple[int, int]:
        # C: PS_0(θ_k) ; PS_1(φ_k) ; BS()
        circuit.add(0, pcvl.PS(self.var_params[var_idx]), merge=True)
        circuit.add(1, pcvl.PS(self.var_params[var_idx + 1]), merge=True)
        circuit.add(0, pcvl.BS(), merge=True)
        var_idx += 2
        return data_idx, var_idx

    @property
    def _data_block(self) -> Callable:
        return {"A": self._data_A, "B": self._data_B, "C": self._data_C}[self.design[0]]

    @property
    def _var_block(self) -> Callable:
        return {"A": self._var_A, "B": self._var_B, "C": self._var_C}[self.design[1]]

    # ---------- Circuit builder ----------
    def _build_circuit(self) -> Any:
        circuit = pcvl.Circuit(2)
        var_idx = 0

        for _ in range(self.num_layers):
            data_idx = 0
            for _ in range(self.num_rotation_blocks_per_layer):
                # Data block (consumes up to 2 features)
                data_idx, var_idx = self._data_block(circuit, data_idx, var_idx)
                # Trainable block (always consumes 2 trainable params)
                data_idx, var_idx = self._var_block(circuit, data_idx, var_idx)
            circuit.barrier()

        # Sanity checks (optional; helps catch mistakes early)
        if var_idx != self.num_var_params:
            raise RuntimeError(
                f"Var params consumed ({var_idx}) != allocated ({self.num_var_params})."
            )
        # data_idx may be == dimension or dimension+1 depending on tail handling inside blocks;
        # we enforce that all features from 0..dimension-1 were used at least once.
        if data_idx < self.dimension:
            raise RuntimeError(
                f"Data features consumed ({data_idx}) < dimension ({self.dimension})."
            )

        return circuit

    # ---------- Parameter update helpers ----------
    def update_data_params(self, data_vector: np.ndarray) -> None:
        if len(data_vector) != self.dimension:
            raise ValueError(
                f"Expected {self.dimension} features, got {len(data_vector)}"
            )
        encoded = data_vector
        for i, p in enumerate(self.data_params):
            p.set_value(float(encoded[i]))

    def update_var_params(self, var_vector: np.ndarray) -> None:
        if len(var_vector) != self.num_var_params:
            raise ValueError(
                f"Expected {self.num_var_params} parameters, got {len(var_vector)}"
            )
        for i, p in enumerate(self.var_params):
            p.set_value(float(var_vector[i]))


# -------------------------------------------------------------------------------------------------
# Merlin Backend — autograd
# -------------------------------------------------------------------------------------------------


class MerlinReuploadingClassifier(BaseEstimator, ClassifierMixin):
    """
    __init__(dimension, num_layers, design)
      - architecture ONLY

    fit(...):
      - max_epochs, learning_rate, batch_size, patience, tau, track_history (training hyperparams)

    Pipeline:
      Stage 1: train circuit params with Fisher loss on x=p10 (1D)
      Stage 2: freeze circuit, fit 1D LDA head on training features
    """

    def __init__(
        self, dimension: int, num_layers: int = 1, design: str = "AA", alpha=np.pi / 4
    ):
        if dimension <= 0 or num_layers <= 0:
            raise ValueError("dimension and num_layers must be positive.")
        self.dimension = dimension
        self.num_layers = num_layers
        self.alpha = alpha

        # Build circuit and Merlin quantum layer
        circuit_model = ReUploadCircuitModelTwoMode(
            self.dimension, self.num_layers, design
        )
        circuit_model.update_var_params(
            np.random.uniform(0, np.pi, size=circuit_model.num_var_params)
        )
        quantum_layer = ml.QuantumLayer(
            input_size=self.dimension,
            output_size=2,  # [p01, p10]
            circuit=circuit_model.circuit,
            trainable_parameters=["var"],
            input_parameters=["x"],
            input_state=circuit_model.input_state,
            output_mapping_strategy=ml.OutputMappingStrategy.NONE,
        )

        class QuantumModule(nn.Module):
            def __init__(self, layer, alpha):
                super().__init__()
                self.layer = layer
                self.alpha = alpha

            def forward(self, x):
                encoded_x = x * self.alpha
                probs = self.layer(encoded_x)  # (N,2)
                p10 = probs[..., 1]  # (N,)
                return p10.unsqueeze(-1)  # (N,1)

        self.quantum_model = QuantumModule(quantum_layer, self.alpha)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.quantum_model.to(self.device)

        self.classifier_ = LDA1DHead()
        self.is_fitted_ = False
        self.training_history_ = None
        self.best_params_ = None
        self.convergence_epoch_ = None

    def _create_loader(self, X: torch.Tensor, y: torch.Tensor, batch_size: int):
        ds = torch.utils.data.TensorDataset(X, y)
        return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        track_history: bool = True,
        *,
        max_epochs: int = 1000,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        patience: int = 50,
        tau: float = 1.0,
        convergence_tolerance: float = 1e-6,
    ) -> MerlinReuploadingClassifier:
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        y_t = torch.tensor(y, dtype=torch.long, device=self.device)
        loader = self._create_loader(X_t, y_t, batch_size=batch_size)

        self.training_history_ = {"loss": [], "epochs": 0} if track_history else None
        opt = optim.Adam(self.quantum_model.parameters(), lr=learning_rate)

        best_loss = float("inf")
        patience_ctr = 0
        avg_loss = None

        self.quantum_model.train()
        for epoch in range(max_epochs):
            epoch_losses = []
            for bx, by in loader:
                opt.zero_grad(set_to_none=True)
                feats = self.quantum_model(bx)  # (B,1)
                loss = fisher_logratio_loss(feats, by, tau=tau)
                loss.backward()
                opt.step()
                epoch_losses.append(float(loss.item()))

            avg_loss = float(np.mean(epoch_losses))
            if track_history:
                self.training_history_["loss"].append(avg_loss)
                self.training_history_["epochs"] = epoch + 1

            if avg_loss < best_loss - convergence_tolerance:
                best_loss = avg_loss
                self.best_params_ = {
                    n: p.clone() for n, p in self.quantum_model.named_parameters()
                }
                patience_ctr = 0
            else:
                patience_ctr += 1

            if patience_ctr >= patience:
                self.convergence_epoch_ = epoch + 1
                break

        # Stage 2: LDA 1D head on training features
        self.quantum_model.eval()
        with torch.no_grad():
            feats_train = self.quantum_model(X_t).cpu().numpy().reshape(-1)
        self.classifier_.fit(feats_train, y)
        self.is_fitted_ = True
        return self

    # Inference
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted_:
            raise ValueError("Call fit() first.")
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            feats = self.quantum_model(X_t).cpu().numpy().reshape(-1)
        return self.classifier_.predict(feats)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted_:
            raise ValueError("Call fit() first.")
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            feats = self.quantum_model(X_t).cpu().numpy().reshape(-1)
        return self.classifier_.predict_proba(feats)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return accuracy_score(y, self.predict(X))

    def get_quantum_features(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted_:
            raise ValueError("Call fit() first.")
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            return self.quantum_model(X_t).cpu().numpy().reshape(-1, 1)


# ------------------------------------------------------------------------------------------------------------
# Perceval Backend — Adam (PSR), COBYLA (gradient-free), L‑BFGS‑B (gradient-free), Nelder‑Mead (gradient-free)
# ------------------------------------------------------------------------------------------------------------


@dataclass
class COBYLAConfig:
    tol: float = 1e-6
    rhobeg: float = np.pi / 2
    maxiter: int | None = None


@dataclass
class LBFGSBConfig:
    maxiter: int = 15_000
    maxfun: int = 15_000
    iprint: int = -1
    maxls: int = 20
    ftol: float = np.finfo(float).eps
    gtol: float = 1e-5
    maxcor: int = 10
    max_eval: int | None = None


@dataclass
class NelderMeadConfig:
    maxiter: int | None = None
    maxfev: int | None = None
    xatol: float = 1e-4
    fatol: float = 1e-4
    adaptive: bool = False


@dataclass
class AdamPSRConfig:
    lr: float = 1e-2
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    shift: float = np.pi / 2
    batch_size: int = 128


class PercevalReuploadingClassifier(BaseEstimator, ClassifierMixin):
    """
    __init__(dimension, num_layers, design, alpha)
      - architecture
      - mapping from data to phases

    fit(...):
      - optimizer
      - max_epochs, tau
      - n_shots, [optimizer params]
      - track_history

    Pipeline:
      Stage 1: train circuit params with Fisher loss on x=p10 (1D)
      Stage 2: freeze circuit, fit 1D LDA head on training features
    """

    def __init__(
        self,
        dimension: int,
        num_layers: int = 1,
        design: str = "AA",
        *,
        alpha: float = np.pi / 4,
        noise_model: Any | None = None,
    ) -> None:
        if dimension <= 0 or num_layers <= 0:
            raise ValueError("dimension and num_layers must be positive.")
        self.dimension = dimension
        self.num_layers = num_layers
        self.alpha = alpha
        self.noise_model = noise_model

        self.quantum_model = ReUploadCircuitModelTwoMode(dimension, num_layers, design)
        self.processor = pcvl.Processor("SLOS", 2, noise=noise_model)

        self.classifier_ = LDA1DHead()
        self.is_fitted_ = False
        self.training_history_: dict[str, Any] | None = None
        self.best_params_: np.ndarray | None = None

    # ----- utilities -----
    def _set_theta(self, theta: np.ndarray):
        self.quantum_model.update_var_params(theta)

    def _feature_p10(self, X: np.ndarray, *, n_shots: float | int) -> np.ndarray:
        feats: list[list[float]] = []
        self.processor.set_circuit(self.quantum_model.circuit)
        self.processor.with_input(self.quantum_model.input_state)
        sampler = Sampler(self.processor)
        for x_s in X:
            self.quantum_model.update_data_params(x_s * self.alpha)
            if np.isinf(n_shots):
                probs = sampler.prob_distribution()
                p10 = probs.get(str(pcvl.BasicState([1, 0])), 0.0)
            else:
                res = sampler.samples(int(n_shots))
                cnts: dict[str, int] = {}
                for o in res["results"]:
                    k = str(o)
                    cnts[k] = cnts.get(k, 0) + 1
                p10 = cnts.get(str(pcvl.BasicState([1, 0])), 0) / n_shots
            feats.append([p10])
        return np.asarray(feats, dtype=float)

    def _objective(self, theta: np.ndarray, X, y, tau, n_shots):
        self._set_theta(theta)
        return fisher_logratio_loss_numpy(
            self._feature_p10(X, n_shots=n_shots), y, tau=tau
        )

    # ----- optimizers -----
    def _train_cobyla(self, X, y, *, tau, n_shots, cfg: COBYLAConfig, track):
        P = self.quantum_model.num_var_params
        init = np.random.uniform(0, 2 * np.pi, size=P)
        self.training_history_["epochs"] = 0

        def fun(theta):
            return self._objective(theta, X, y, tau, n_shots)

        def cb(xk):
            if track:
                self.training_history_["loss"].append(fun(xk))
                self.training_history_["epochs"] += 1

        res = minimize(
            fun,
            init,
            method="COBYLA",
            tol=cfg.tol,
            callback=cb,
            options={
                "rhobeg": cfg.rhobeg,
                "maxiter": cfg.maxiter,
                "disp": False,
            },
        )
        self.best_params_ = res.x
        self._set_theta(res.x)

    def _train_lbfgsb(self, X, y, *, tau, n_shots, cfg, track):
        P = self.quantum_model.num_var_params
        init = np.random.uniform(0, 2 * np.pi, size=P)
        self.training_history_["epochs"] = 0

        def fun(theta):
            return self._objective(theta, X, y, tau, n_shots)

        def cb(xk):
            if track:
                self.training_history_["loss"].append(fun(xk))
                self.training_history_["epochs"] += 1

        res = minimize(
            fun,
            init,
            method="L-BFGS-B",
            jac=False,
            callback=cb,
            options={
                "maxiter": cfg.maxiter,
                "maxfun": cfg.maxfun,
                "iprint": cfg.iprint,
                "maxls": cfg.maxls,
                "ftol": cfg.ftol,
                "gtol": cfg.gtol,
                "maxcor": cfg.maxcor,
                "max_eval": cfg.max_eval,
            },
        )
        self.best_params_ = res.x
        self._set_theta(res.x)

    def _train_nelder(self, X, y, *, tau, n_shots, cfg, track):
        P = self.quantum_model.num_var_params
        init = np.random.uniform(0, 2 * np.pi, size=P)
        self.training_history_["epochs"] = 0

        def fun(theta):
            return self._objective(theta, X, y, tau, n_shots)

        def cb(xk):
            if track:
                self.training_history_["loss"].append(fun(xk))
                self.training_history_["epochs"] += 1

        res = minimize(
            fun,
            init,
            method="Nelder-Mead",
            callback=cb,
            options={
                "maxiter": cfg.maxiter,
                "maxfev": cfg.maxfev,
                "xatol": cfg.xatol,
                "fatol": cfg.fatol,
                "adaptive": cfg.adaptive,
            },
        )
        self.best_params_ = res.x
        self._set_theta(res.x)

    def _psr_gradient(self, theta, X, y, tau, n_shots, shift):
        P = len(theta)
        grad = np.zeros(P, dtype=float)
        for i in range(P):
            e = np.zeros(P)
            e[i] = 1.0
            loss_plus = self._objective(theta + shift * e, X, y, tau, n_shots)
            loss_minus = self._objective(theta - shift * e, X, y, tau, n_shots)
            grad[i] = 0.5 * (loss_plus - loss_minus) / np.sin(shift)
        return grad

    def _train_adam_psr(
        self, X, y, *, tau, n_shots, max_epochs, cfg: AdamPSRConfig, track
    ):
        """
        Train the variable parameters with Adam + Parameter-Shift Rule (PSR) **via
        `scipy.optimize.minimize(method="adam")`** instead of a hand-rolled loop.
        All Adam hyper-parameters come from `cfg`.

        Parameters
        ----------
        X, y          : training data / labels (NumPy)
        tau           : Fisher-loss temperature
        n_shots       : number of shots (``np.inf`` ⇒ analytic probabilities)
        max_epochs    : optimiser iterations (`maxiter`)
        cfg           : AdamPSRConfig with lr, beta1, beta2, eps, shift, batch_size
        track         : if True, record loss history in `self.training_history_`
        """
        P = self.quantum_model.num_var_params
        init_theta = np.random.uniform(0, 2 * np.pi, size=P)

        # ——— objective and analytic PSR gradient ———
        def fun(theta: np.ndarray) -> float:
            return self._objective(theta, X, y, tau, n_shots)

        def grad(theta: np.ndarray) -> np.ndarray:
            # mini-batch PSR gradient (supports cfg.batch_size)
            if cfg.batch_size < len(X):
                idx = np.random.choice(len(X), cfg.batch_size, replace=False)
                X_b, y_b = X[idx], y[idx]
            else:
                X_b, y_b = X, y
            return self._psr_gradient(theta, X_b, y_b, tau, n_shots, cfg.shift)

        # ——— callback for loss history ———
        if track:
            self.training_history_["epochs"] = 0

            def cb(theta):
                self.training_history_["loss"].append(float(fun(theta)))
                self.training_history_["epochs"] += 1
        else:
            cb = None

        # ——— run SciPy “adam” optimiser ———
        res = minimize(
            fun,
            init_theta,
            method="adam",
            jac=grad,
            callback=cb,
            options={
                "lr": cfg.lr,
                "beta1": cfg.beta1,
                "beta2": cfg.beta2,
                "eps": cfg.eps,
                "maxiter": max_epochs,
            },
        )

        self.best_params_ = res.x
        self._set_theta(res.x)

    # ——— public API ———
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        optimizer: str = "cobyla",  # {cobyla|l_bfgs_b|nelder_mead|adam_psr}
        cfg=COBYLAConfig,
        tau: float = 1.0,
        n_shots: float | int = 2000,
        track_history: bool = True,
    ) -> PercevalReuploadingClassifier:
        # init history store
        if track_history:
            self.training_history_ = {"loss": [], "epochs": 0}
        else:
            self.training_history_ = {"loss": [], "epochs": 0}

        optim = optimizer.lower()
        if optim == "cobyla":
            self._train_cobyla(
                X, y, tau=tau, n_shots=n_shots, cfg=cfg, track=track_history
            )
        elif optim in {"l_bfgs_b", "lbfgsb"}:
            self._train_lbfgsb(
                X, y, tau=tau, n_shots=n_shots, cfg=cfg, track=track_history
            )
        elif optim == "nelder_mead":
            self._train_nelder(
                X, y, tau=tau, n_shots=n_shots, cfg=cfg, track=track_history
            )
        elif optim == "adam_psr":
            self._train_adam_psr(
                X, y, tau=tau, n_shots=n_shots, cfg=cfg, track=track_history
            )
        else:
            raise ValueError(
                "optimizer must be one of {'cobyla','l_bfgs_b','nelder_mead','adam_psr'}"
            )

        # LDA head
        feats_train = self._feature_p10(X, n_shots=n_shots).reshape(-1)
        self.classifier_.fit(feats_train, y)
        self.is_fitted_ = True
        return self

    # Inference
    def _feats(self, X: np.ndarray, n_shots: float | int) -> np.ndarray:
        """Internal: flattened (N,) vector of quantum feature *p10* with given shot budget."""
        return self._feature_p10(X, n_shots=n_shots).reshape(-1)

    def predict(self, X: np.ndarray, *, n_shots: float | int = 2000) -> np.ndarray:
        """Class labels for **X** using *n_shots* (or `np.inf` for analytic)."""
        if not self.is_fitted_:
            raise RuntimeError("Call fit() first.")
        return self.classifier_.predict(self._feats(X, n_shots))

    def predict_proba(
        self, X: np.ndarray, *, n_shots: float | int = 2000
    ) -> np.ndarray:
        """Class probabilities for **X** using the same shot logic as :meth:`predict`."""
        if not self.is_fitted_:
            raise RuntimeError("Call fit() first.")
        return self.classifier_.predict_proba(self._feats(X, n_shots))

    def score(
        self, X: np.ndarray, y: np.ndarray, *, n_shots: float | int = 2000
    ) -> float:
        """Accuracy on (X, y) at the specified shot count."""
        return accuracy_score(y, self.predict(X, n_shots=n_shots))

    def get_quantum_features(
        self, X: np.ndarray, *, n_shots: float | int = 2000
    ) -> np.ndarray:
        """Return the learned 1‑D quantum feature :math:`x = p_{10}` for each sample."""
        if not self.is_fitted_:
            raise RuntimeError("Call fit() first.")
        return self._feature_p10(X, n_shots=n_shots)
