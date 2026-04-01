# a2_dho_cp.py
# Classical–PennyLane PINN with a quantum branch and a classical MLP branch

import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from ..config import (
    DEFAULT_N_OUTPUTS,
    DHO_HIDDEN_WIDTH,
    DHO_LR,
    DHO_NUM_HIDDEN_LAYERS,
    DHO_N_EPOCHS,
    DHO_PLOT_EVERY,
    DTYPE,
    N_LAYERS,
)
from ..utils import make_time_grid, make_optimizer
from .core_a2_dho import train_oscillator_pinn, u_exact
from ..run_common import run_series_inference_mode
from ..layer_pennylane import make_quantum_block, dho_feature_map, BranchPennylane
from ..layer_classical import BranchPyTorch


class CQ_PINN(nn.Module):
    """
    Hybrid Classical–Quantum PINN with linear fusion to scalar output.
    """

    def __init__(
        self,
        *,
        num_hidden_layers: int = DHO_NUM_HIDDEN_LAYERS,
        hidden_width: int = DHO_HIDDEN_WIDTH,
        n_qubits: int = DEFAULT_N_OUTPUTS,
    ) -> None:
        super().__init__()

        qblock = make_quantum_block(n_qubits=n_qubits)

        self.branch_q = BranchPennylane(
            qblock,
            feature_map=lambda t: dho_feature_map(t, n_qubits=n_qubits),
            output_as_column=True,
            n_layers=N_LAYERS,
            n_qubits=n_qubits,
        )
        self.branch_c = BranchPyTorch(
            num_hidden_layers=num_hidden_layers,
            hidden_width=hidden_width,
        )
        self.fusion = nn.Linear(4, 1, dtype=DTYPE)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        out_q = self.branch_q(t)
        out_c = self.branch_c(t)
        return self.fusion(torch.cat([out_q, out_c], dim=1))


def plot_model_prediction(u_pred, u_ex, t, save_path="HQPINN/DHO/results/dho_cp/"):
    plt.figure(figsize=(10, 6))
    plt.plot(t.cpu().numpy(), u_pred, label="Prediction PINN", lw=2)
    plt.plot(t.cpu().numpy(), u_ex, "--", label="Exact solution", lw=2)
    plt.xlabel("t")
    plt.ylabel("u(t)")
    plt.title("DHO - Classical-PennyLane PINN")
    plt.grid(True)
    plt.legend()

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs(save_path, exist_ok=True)
    png_path = os.path.join(save_path, f"dho_cp_plot_{timestamp}.png")
    plt.savefig(png_path, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to: {png_path}")


def _case_prefix(n_layers: int, n_nodes: int, n_qubits: int) -> str:
    if (
        n_layers == DHO_NUM_HIDDEN_LAYERS
        and n_nodes == DHO_HIDDEN_WIDTH
        and n_qubits == DEFAULT_N_OUTPUTS
    ):
        return "dho_cp"
    return f"dho_cp_{n_nodes}-{n_layers}-q{n_qubits}"


def run(
    mode="train",
    backend="sim:ascella",
    *,
    n_layers: int = DHO_NUM_HIDDEN_LAYERS,
    n_nodes: int = DHO_HIDDEN_WIDTH,
    n_qubits: int = DEFAULT_N_OUTPUTS,
) -> None:
    """Run the Classical–PennyLane DHO PINN experiment."""
    torch.manual_seed(0)
    np.random.seed(0)
    ckpt_dir = "HQPINN/DHO/models"
    case_prefix = _case_prefix(n_layers, n_nodes, n_qubits)
    results_dir = f"HQPINN/DHO/results/{case_prefix}"

    if mode == "train":
        model = CQ_PINN(
            num_hidden_layers=n_layers,
            hidden_width=n_nodes,
            n_qubits=n_qubits,
        )
        train_oscillator_pinn(
            model=model,
            t_train=make_time_grid(),
            optimizer=make_optimizer(model, lr=DHO_LR),
            n_epochs=DHO_N_EPOCHS,
            plot_every=DHO_PLOT_EVERY,
            out_dir=results_dir,
            model_label="cp",
        )
        os.makedirs(ckpt_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        ckpt_path = os.path.join(ckpt_dir, f"{case_prefix}_{timestamp}.pt")
        torch.save(model.state_dict(), ckpt_path)
        print(f"Model saved to: {ckpt_path}")

    elif mode == "run":
        run_series_inference_mode(
            mode="run",
            backend="local",
            ckpt_dir=ckpt_dir,
            case_prefix=case_prefix,
            model_factory=lambda processor=None: CQ_PINN(
                num_hidden_layers=n_layers,
                hidden_width=n_nodes,
                n_qubits=n_qubits,
            ),
            make_time_grid=make_time_grid,
            exact_fn=u_exact,
            plot_fn=lambda u_pred, u_ex, t: plot_model_prediction(
                u_pred, u_ex, t, save_path=results_dir
            ),
        )

    elif mode == "remote":
        print("Remote mode is not available for DHO-CP. Falling back to local run mode.")
        run_series_inference_mode(
            mode="run",
            backend="local",
            ckpt_dir=ckpt_dir,
            case_prefix=case_prefix,
            model_factory=lambda processor=None: CQ_PINN(
                num_hidden_layers=n_layers,
                hidden_width=n_nodes,
                n_qubits=n_qubits,
            ),
            make_time_grid=make_time_grid,
            exact_fn=u_exact,
            plot_fn=lambda u_pred, u_ex, t: plot_model_prediction(
                u_pred, u_ex, t, save_path=results_dir
            ),
        )

    else:
        raise ValueError("mode must be 'train', 'run', or 'remote'")
