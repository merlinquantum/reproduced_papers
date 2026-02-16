# ============================================================
#  MerLin-based QQ-PINN for the damped oscillator
#  ------------------------------------------------------------
#  ODE:
#      m u''(t) + μ u'(t) + k u(t) = 0
#
#  Model:
#    - Two parallel photonic quantum branches (MerLin QuantumLayer)
#    - Dual-rail encoding for 3 logical qubits (6 modes)
#
#  Main differences vs PennyLane version:
#    - Quantum circuit built with Perceval
#    - Trainable parameters and inputs handled via MerLin's QuantumLayer
# ============================================================

import os
from datetime import datetime

import numpy as np

import torch
import torch.nn as nn
from torch.autograd import grad

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

matplotlib.use("Agg")  # non-interactive backend for batch PDF export

# MerLin / Perceval imports
import merlin as ML
from merlin import LexGrouping, QuantumLayer
from merlin.core.merlin_processor import MerlinProcessor

import perceval as pcvl
from perceval import PS, BS

# ============================================================
#  Physical and training hyperparameters
# ============================================================

m = 1.0
mu = 4.0
k = 400.0

lambda1 = 1e-1
lambda2 = 1e-4

# lr = 0.05 # Diverge
lr = 0.002

n_epochs = 2000
plot_every = 100


# ============================================================
#  Quantum architecture
# ============================================================

n_qubits = 3

# ============================================================
#  Training data: t ∈ [0, 1]
# ============================================================

t_train = np.linspace(0.0, 1.0, 200)
t_train_torch = torch.tensor(t_train, dtype=torch.float32).reshape(-1, 1)

torch.manual_seed(0)
np.random.seed(0)


# ============================================================
#  Perceval building blocks: entangling and ansatz/feature layers
# ============================================================


def entangling_chain_all_modes(n_qubits: int) -> pcvl.Circuit:
    """
    Linear (non-circular) entangling chain across all 2 * n_qubits modes.

    Structure
    ---------
    - n_modes = 2 * n_qubits (dual-rail encoding)
    - Apply BS.H between (m, m+1) for m = 0 .. n_modes - 2

    Note
    ----
    - This is a simple, hardware-motivated entangling layer.
    - It is applied *after* the per-qubit rotations in ansatz_layer.
    """
    n_modes = 2 * n_qubits
    circ = pcvl.Circuit(n_modes)

    for m in range(n_modes - 1):
        # BS.H acts on the pair (m, m+1)
        circ // (m, BS.H())

    return circ


def ansatz_layer(prefix: str) -> pcvl.Circuit:
    """
    Perceval implementation of an ansatz layer in dual-rail encoding.

    Dual-rail encoding
    ------------------
    - Logical qubit i ↦ spatial modes (2*i, 2*i+1).

    Parameters (symbolic)
    ---------------------
    For each logical qubit i we introduce 3 parameters:
      theta_{prefix}_{i}_0 : "RZ-like" rotation
      theta_{prefix}_{i}_1 : "RX-like" rotation
      theta_{prefix}_{i}_2 : "RZ-like" rotation

    Implementation
    --------------
    - RZ is implemented via a phase shifter on one rail (relative phase).
    - RX is implemented via BS.Rx on the dual-rail pair.
    - Followed by a global entangling chain across all modes.

    Hard point
    ----------
    - Parameters are *symbolic* (pcvl.P). MerLin will later bind
      all symbols whose name contains "theta" as trainable parameters.
    """
    circ = pcvl.Circuit(2 * n_qubits)

    for i in range(n_qubits):
        m0 = 2 * i
        m1 = 2 * i + 1

        theta_z1 = pcvl.P(f"theta_{prefix}_{i}_0")
        theta_x = pcvl.P(f"theta_{prefix}_{i}_1")
        theta_z2 = pcvl.P(f"theta_{prefix}_{i}_2")

        # Approximate RZ via relative phase on one rail
        circ // (m1, PS(theta_z1))

        # Approximate RX via beam-splitter rotation on the dual-rail pair
        circ // (m0, BS.Rx(theta_x))

        # Second RZ
        circ // (m1, PS(theta_z2))

    # Add entangling chain across all dual-rail modes
    return circ // entangling_chain_all_modes(n_qubits)


def feature_layer(prefix: str) -> pcvl.Circuit:
    """
    Feature map layer implemented as BS.Ry rotations.

    Parameters (symbolic)
    ---------------------
    For each logical qubit i, we introduce:
      phi_{prefix}_{i}

    Hard point
    ----------
    - These 'phi_*' parameters are *inputs* (not trainable).
      MerLin treats all symbols that contain "phi" as input parameters
      fed by the classical feature tensor at each forward pass.
    """
    circ = pcvl.Circuit(2 * n_qubits)

    for i in range(n_qubits):
        phi = pcvl.P(f"phi_{prefix}_{i}")
        circ // (2 * i, BS.Ry(phi))

    return circ


# ============================================================
#  Full photonic circuit for a quantum block
# ============================================================


def quantum_block() -> pcvl.Circuit:
    """
    Full quantum block:

        ansatz("layer0") → feature("layer1") →
        ansatz("layer2") → feature("layer3") →
        ansatz("layer4")

    This matches the structure of the PennyLane version:
      A → Φ → A → Φ → A

    Note
    ----
    - The naming conventions ("layer0", "layer1", ...) must be consistent
      with how we later build the feature tensor X in QuantumBranch.
    """
    circ = pcvl.Circuit(2 * n_qubits)

    circ = circ // ansatz_layer("layer0")
    circ = circ // feature_layer("layer1")

    circ = circ // ansatz_layer("layer2")
    circ = circ // feature_layer("layer3")

    circ = circ // ansatz_layer("layer4")

    return circ


circuit = quantum_block()


# ============================================================
#  MerLin QuantumLayers (two independent branches)
# ============================================================

input_size = 2 * n_qubits

qlayer1 = QuantumLayer(
    input_size=input_size,
    circuit=circuit,
    # Dual-rail encoding of |1⟩ for each logical qubit: [1,0, 1,0, 1,0]
    input_state=[1, 0, 1, 0, 1, 0],
    trainable_parameters=["theta"],  # all symbols whose name contains "theta"
    input_parameters=["phi"],  # all symbols whose name contains "phi"
    dtype=torch.float32,
)

qlayer2 = QuantumLayer(
    input_size=input_size,
    circuit=circuit,
    input_state=[1, 0, 1, 0, 1, 0],
    trainable_parameters=["theta"],
    input_parameters=["phi"],
    dtype=torch.float32,
)


# ============================================================
#  Quantum branch: QuantumLayer + LexGrouping + Linear readout
# ============================================================


class QuantumBranch(nn.Module):
    """
    Single quantum branch based on a MerLin QuantumLayer.

    Hard points
    -----------
    - We must align the t-dependent features with the symbolic
        "phi_layer1_*" and "phi_layer3_*" parameters used in the circuit.
    - Here we mimic the PennyLane feature map φ(t) = [π t, 2π t, 3π t]
        and reuse it for the two feature layers, giving 6 inputs.
    """

    def __init__(self, qlayer: QuantumLayer, n_qubits: int) -> None:
        super().__init__()
        self.qlayer = qlayer
        self.n_qubits = n_qubits

        # Number of intermediate features extracted from the quantum output.
        self.group_dim = 2 * n_qubits

        # LexGrouping reduces the QuantumLayer output from (N, output_size)
        # to (N, group_dim) by aggregating output_size into group_dim bins.
        self.group = LexGrouping(self.qlayer.output_size, self.group_dim)

        self.readout = nn.Linear(self.group_dim, 1)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 2:
            t = t.squeeze(-1)

        # Base feature map: φ(t) = [π t, 2π t, 3π t]
        scale = np.pi
        phi0 = scale * t
        phi1 = 2.0 * scale * t
        phi2 = 3.0 * scale * t

        # Two feature layers → concatenate the same triple twice:
        # layer1: [φ0, φ1, φ2], layer3: [φ0, φ1, φ2]
        phi = torch.stack(
            [
                phi0,
                phi1,
                phi2,  # corresponds to feature_layer("layer1")
                phi0,
                phi1,
                phi2,
            ],
            dim=1,
        )

        # QuantumLayer returns probabilities over Fock states.
        q_out = self.qlayer(phi)

        # Aggregate probabilities lexicographically into group_dim features
        feat = self.group(q_out)

        # Linear readout to a single scalar per time point
        u = self.readout(feat)

        return u


# ============================================================
#  Full QQ-PINN model: sum of two quantum branches
# ============================================================


class QQ_PINN(nn.Module):
    """
    Full model with two independent MerLin branches.

    u(t) = u_q1(t) + u_q2(t)
    """

    def __init__(
        self, qlayer1: QuantumLayer, qlayer2: QuantumLayer, n_qubits: int
    ) -> None:
        super().__init__()
        self.branch1 = QuantumBranch(qlayer1, n_qubits)
        self.branch2 = QuantumBranch(qlayer2, n_qubits)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.branch1(t) + self.branch2(t)  # (N, 1)


model = QQ_PINN(qlayer1, qlayer2, n_qubits=n_qubits)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


# ============================================================
#  Remote execution (Quandela Cloud / remote simulator)
# ============================================================

# 1) Créer le RemoteProcessor Perceval
#    - "sim:slos" = simulateur distant (comme dans l'exemple MerLin)
#    - pour un vrai QPU, remplace la chaîne par l’ID fourni par Quandela Cloud
rp = pcvl.RemoteProcessor("sim:slos")

# 2) L'envelopper dans un MerlinProcessor
proc = MerlinProcessor(
    rp,
    microbatch_size=32,  # taille de chunk envoyée par appel cloud
    timeout=3600.0,  # temps max par forward (secondes)
    max_shots_per_call=None,  # cap facultatif par appel
    chunk_concurrency=1,  # jobs parallèles par feuille quantique
)


# ============================================================
#  Autograd utilities: derivatives w.r.t. time t
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
#  Physics-informed loss
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
    # Fresh differentiable copy of t
    t = t.clone().detach().requires_grad_(True)


    # Forward pass on the time grid - via MerlinProcessor
    u = proc.forward(model, t, nsample=5000)
    du = derivative(u, t)
    d2u = second_derivative(u, t)

    # PDE residual: m u'' + μ u' + k u
    f = m * d2u + mu * du + k * u

    # Initial conditions at t = 0
    t0 = torch.zeros((1, 1), dtype=t.dtype, device=t.device).requires_grad_(True)
    u0 = proc.forward(model, t0, nsample=5000)
    du0 = derivative(u0, t0)

    # IC: u(0) = 1
    loss_ic_u = (u0 - 1.0) ** 2

    # IC: u'(0) = 0
    loss_ic_du = du0**2

    # PDE residual loss
    loss_f = torch.mean(f**2)

    return loss_ic_u.squeeze(), loss_ic_du.squeeze(), loss_f.squeeze()


# ============================================================
#  Training loop and PDF logging
# ============================================================

out_dir = "HQPINN/results"
os.makedirs(out_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
pdf_path = os.path.join(out_dir, f"HQPINN-A2-mm_{timestamp}.pdf")

with PdfPages(pdf_path) as pdf:
    for epoch in range(n_epochs):

        optimizer.zero_grad()

        # Compute PINN losses
        lic_u, lic_du, lf = loss_fn(model, t_train_torch)

        # Total loss
        loss = lic_u + lambda1 * lic_du + lambda2 * lf

        # Backpropagation through MerLin + PINN graph
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

            # Diagnostic grid
            t_diag = t_train_torch.clone().detach().requires_grad_(True)

            u_diag = proc.forward(model, t_diag, nsample=5000)

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

            # Numpy conversion for plotting
            with torch.no_grad():
                t_np = t_diag.squeeze().cpu().numpy()
                u_np = u_diag.squeeze().cpu().numpy()
                du_np = du_diag.squeeze().cpu().numpy()
                d2u_np = d2u_diag.squeeze().cpu().numpy()

            # ----------------------------------------------------
            # Page 1: u, u', u''
            # ----------------------------------------------------
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

            # ----------------------------------------------------
            # Page 2: exact solution vs MerLin-based PINN
            # ----------------------------------------------------

            # Exact solution with u(0)=1, u'(0)=0
            omega = np.sqrt(k - (mu / 2.0) ** 2)

            def u_exact(t_array: np.ndarray) -> np.ndarray:
                return np.exp(-mu * t_array / 2.0) * (
                    np.cos(omega * t_array)
                    + (mu / (2.0 * omega)) * np.sin(omega * t_array)
                )

            with torch.no_grad():
                u_pred = proc.forward(model, t_train_torch, nsample=5000)
                # u_pred = model(t_train_torch).cpu().numpy().flatten()
            u_ex = u_exact(t_np)

            fig2, ax2 = plt.subplots(figsize=(10, 6))
            ax2.plot(t_np, u_pred, label="PINN (merlin–merlin)")
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
