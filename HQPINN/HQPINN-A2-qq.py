# ============================================================
#  HQ_PINN: Hybrid Quantum–Quantum Physics-Informed Neural Network
#  ------------------------------------------------------------
#  Solves the damped oscillator ODE:
#
#      m u''(t) + μ u'(t) + k u(t) = 0
#
#  using a PINN built from two parallel quantum branches.
#  Each branch is a multi-layer quantum ansatz evaluated by a
#  PennyLane QNode with a PyTorch interface.
# ============================================================

import os
from datetime import datetime

import numpy as np

import torch
import torch.nn as nn
from torch.autograd import grad

import pennylane as qml
from pennylane import numpy as pnp

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Use a non-interactive backend for automated PDF export
matplotlib.use("Agg")


# ============================================================
#  Physical parameters and training hyperparameters
# ============================================================

# Damped oscillator constants
m = 1.0
mu = 4.0
k = 400.0

# Loss weights
lambda1 = 1e-1
lambda2 = 1e-4

# Optimizer hyperparameters
# lr = 0.002
lr = 0.05
n_epochs = 2800
plot_every = 100

# Quantum model architecture
n_qubits = 3
n_layers = 3


# ============================================================
#  Training data: time samples t ∈ [0, 1]
# ============================================================

# Use PennyLane NumPy for consistency with the QNode interface
t_train = pnp.linspace(0.0, 1.0, 200)

# Convert to PyTorch tensor with shape (N, 1)
t_train_torch = torch.tensor(t_train, dtype=torch.float32).reshape(-1, 1)


# ============================================================
#  Quantum device and random seeds
# ============================================================

# Qubit simulator
dev = qml.device("default.qubit", wires=n_qubits)

# Deterministic runs
torch.manual_seed(0)
np.random.seed(0)


# ============================================================
#  Quantum layers: ansatz and feature encoding
# ============================================================


def ansatz_layer(theta: torch.Tensor) -> None:
    """
    Single ansatz layer.

    Parameters
    ----------
    theta : (n_qubits, 3) tensor-like
        For each qubit i:
          theta[i, 0] : RZ rotation angle
          theta[i, 1] : RX rotation angle
          theta[i, 2] : RZ rotation angle

    Structure
    ---------
    - Local rotations on each qubit
    - Followed by a ring of CNOTs: i -> (i+1) mod n_qubits
    """
    # Local single-qubit rotations
    for i in range(n_qubits):
        qml.RZ(theta[i, 0], wires=i)
        qml.RX(theta[i, 1], wires=i)
        qml.RZ(theta[i, 2], wires=i)

    # Entangling structure: circular chain of CNOT gates
    for i in range(n_qubits):
        qml.CNOT(wires=[i, (i + 1) % n_qubits])


def feature_layer(phi: torch.Tensor) -> None:
    """
    Feature map: angle encoding of a classical input into RY rotations.

    Parameters
    ----------
    phi : (n_qubits,) tensor-like
        For qubit i, apply RY(phi[i]).
    """
    for i in range(n_qubits):
        qml.RY(phi[i], wires=i)


# ============================================================
#  Quantum circuit (QNode)
# ============================================================


@qml.qnode(dev, interface="torch")
def quantum_block(phi: torch.Tensor, thetas: torch.Tensor) -> torch.Tensor:
    """
    Full quantum block used as the building block for each branch.

    Circuit structure
    -----------------
    A typical sequence:
        ansatz(theta_0) →
        feature(phi)    →
        ansatz(theta_1) →
        feature(phi)    →
        ansatz(theta_2)

    Parameters
    ----------
    phi : (n_qubits,) torch.Tensor
        Feature vector for angle encoding on each qubit.

    thetas : (n_layers, n_qubits, 3) torch.Tensor
        Trainable parameters for each ansatz layer.

    Returns
    -------
    torch.Tensor (scalar)
        Expectation value ⟨Z₀⟩ on qubit 0.
    """
    # Layer 1
    ansatz_layer(thetas[0])
    feature_layer(phi)

    # Layer 2
    ansatz_layer(thetas[1])
    feature_layer(phi)

    # Layer 3
    ansatz_layer(thetas[2])

    # Observable: Pauli-Z on qubit 0
    return qml.expval(qml.PauliZ(0))


# ============================================================
#  Quantum branch (PyTorch module)
# ============================================================


class QuantumBranch(nn.Module):
    """
    Single quantum branch: multi-layer QNode wrapped as a PyTorch module.

    Architecture
    ------------
    - Input: scalar time t
    - Feature map: φ(t) ∈ R^3
    - Quantum circuit: quantum_block(φ(t), θ)
    - Output: scalar u_q(t)

    Note
    ----
    The whole branch is differentiable end-to-end through
    the PyTorch–PennyLane interface.
    """

    def __init__(self) -> None:
        super().__init__()

        # Trainable parameters: (n_layers, n_qubits, 3)
        # Small initialization to keep the circuit close to identity,
        # which stabilizes early training.
        self.theta = nn.Parameter(torch.randn(n_layers, n_qubits, 3) * 0.01)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the quantum branch on a batch of time samples.
        """
        if t.dim() == 2:
            t = t.squeeze(-1)

        # Feature-map scaling factors
        # φ(t) = [π t, 2π t, 3π t]
        scale = np.pi
        phi = torch.stack(
            [scale * t, 2 * scale * t, 3 * scale * t],
            dim=1,
        )

        outputs = []
        # Loop over batch dimension; each call builds and executes the QNode.
        for i in range(phi.size(0)):
            out_i = quantum_block(phi[i], self.theta)
            outputs.append(out_i)

        # Stack to shape (N,) then unsqueeze to (N, 1)
        return torch.stack(outputs).unsqueeze(-1)


# ============================================================
#  Full QQ-PINN model: sum of two quantum branches
# ============================================================


class QQ_PINN(nn.Module):
    """
    Physics-Informed model: sum of two independent quantum branches.

    u(t) = u_q1(t) + u_q2(t)
    """

    def __init__(self) -> None:
        super().__init__()
        self.branch1 = QuantumBranch()
        self.branch2 = QuantumBranch()

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        t : (N, 1) or (N,) torch.Tensor

        Returns
        -------
        (N, 1) torch.Tensor
            u(t) = u_q1(t) + u_q2(t)
        """
        return self.branch1(t) + self.branch2(t)


# ============================================================
#  Autograd utilities: first and second derivatives w.r.t. t
# ============================================================


def derivative(u: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Compute du/dt using PyTorch autograd.
    """
    return grad(
        outputs=u,
        inputs=t,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
    )[0]


def second_derivative(u: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Second derivative d²u/dt².
    """
    du_dt = derivative(u, t)
    return derivative(du_dt, t)


# ============================================================
#  Physics-informed loss function
# ============================================================


def loss_fn(model: nn.Module, t: torch.Tensor):
    """
    Compute the three components of the PINN loss:

      1. Initial condition on u(0)
      2. Initial condition on u'(0)
      3. PDE residual over the time grid

    Hard points
    -----------
    1. t must be a fresh tensor with requires_grad=True for each call,
       otherwise the autograd graph becomes tangled across iterations.

    2. Initial condition derivative:
       - We use a dedicated scalar time t0 = 0 with requires_grad=True,
         run the model on it, then differentiate w.r.t. t0.
    """
    # Fresh differentiable copy of t for derivative computations
    t = t.clone().detach().requires_grad_(True)

    # Forward pass over training grid
    u = model(t)  # u(t)
    du = derivative(u, t)  # u'(t)
    d2u = second_derivative(u, t)  # u''(t)

    # PDE residual: m u'' + μ u' + k u
    f = m * d2u + mu * du + k * u

    # Initial conditions at t = 0
    t0 = torch.zeros((1, 1), dtype=t.dtype, device=t.device).requires_grad_(True)
    u0 = model(t0)
    du0 = derivative(u0, t0)

    # IC: u(0) = 1
    loss_ic_u = (u0 - 1.0) ** 2

    # IC: u'(0) = 0
    loss_ic_du = du0**2

    # PDE residual loss over the full grid
    loss_f = torch.mean(f**2)

    # Return scalars
    return (
        loss_ic_u.squeeze(),
        loss_ic_du.squeeze(),
        loss_f.squeeze(),
    )


# ============================================================
#  Training loop and PDF logging
# ============================================================

model = QQ_PINN()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Output directory and timestamped PDF path
out_dir = "HQPINN/results"
os.makedirs(out_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
pdf_path = os.path.join(out_dir, f"HQPINN-A2-qq_{timestamp}.pdf")

with PdfPages(pdf_path) as pdf:
    for epoch in range(n_epochs):

        optimizer.zero_grad()

        # Compute physics-informed losses
        lic_u, lic_du, lf = loss_fn(model, t_train_torch)

        # Total weighted loss
        loss = lic_u + lambda1 * lic_du + lambda2 * lf

        # Backpropagation through the quantum circuits and autograd graph
        loss.backward()
        optimizer.step()

        # Diagnostics and plotting
        if epoch % plot_every == 0:
            print(
                f"Epoch {epoch:4d} | "
                f"Loss = {loss.item():.4e} | "
                f"IC_u = {lic_u:.4e} | "
                f"IC_du = {lic_du:.4e} | "
                f"PDE = {lf:.4e}"
            )

            # Diagnostic grid with gradients enabled
            t_diag = t_train_torch.clone().detach().requires_grad_(True)
            u_diag = model(t_diag)
            du_diag = derivative(u_diag, t_diag)
            d2u_diag = second_derivative(u_diag, t_diag)

            print(
                "||u||:",
                u_diag.abs().mean().item(),
                "\n||u'||:",
                du_diag.abs().mean().item(),
                "\n||u''||:",
                d2u_diag.abs().mean().item(),
            )

            # Convert to NumPy for plotting (no_grad to detach from graph)
            with torch.no_grad():
                t_np = t_diag.squeeze().cpu().numpy()
                u_np = u_diag.squeeze().cpu().numpy()
                du_np = du_diag.squeeze().cpu().numpy()
                d2u_np = d2u_diag.squeeze().cpu().numpy()

            # --------------------------------------------------------
            # Page 1: u(t), u'(t), u''(t)
            # --------------------------------------------------------
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            ax1.plot(t_np, u_np, label="u(t)")
            ax1.plot(t_np, du_np, label="u'(t)")
            ax1.plot(t_np, d2u_np, label="u''(t)")
            ax1.legend()
            ax1.set_xlabel("t")
            ax1.set_title(f"Diagnostics: u, u', u'' — epoch {epoch}")
            ax1.grid(True)
            fig1.tight_layout()
            pdf.savefig(fig1, bbox_inches="tight")
            plt.close(fig1)

            # --------------------------------------------------------
            # Page 2: comparison with exact solution
            # --------------------------------------------------------

            # Exact solution of the damped oscillator with u(0)=1, u'(0)=0
            omega = np.sqrt(k - (mu / 2.0) ** 2)

            def u_exact(t_array: np.ndarray) -> np.ndarray:
                return np.exp(-mu * t_array / 2.0) * (
                    np.cos(omega * t_array)
                    + (mu / (2.0 * omega)) * np.sin(omega * t_array)
                )

            with torch.no_grad():
                u_pred = model(t_train_torch).cpu().numpy().flatten()
            u_ex = u_exact(t_np)

            fig2, ax2 = plt.subplots(figsize=(10, 6))
            ax2.plot(t_np, u_pred, label="PINN (quantum-quantum)")
            ax2.plot(t_np, u_ex, "--", label="Exact")
            ax2.legend()
            ax2.set_xlabel("t")
            ax2.set_ylabel("u(t)")
            ax2.set_title(f"Prediction vs Exact — epoch {epoch}")
            ax2.grid(True)
            fig2.tight_layout()
            pdf.savefig(fig2, bbox_inches="tight")
            plt.close(fig2)

# End of file
