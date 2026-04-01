# a2-dho-pp.py
# PennyLane–PennyLane PINN with two parallel quantum branches

import os
from datetime import datetime

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from ..config import DEFAULT_N_OUTPUTS, DHO_N_EPOCHS, DHO_PLOT_EVERY, DHO_LR, N_LAYERS, DTYPE
from ..utils import make_time_grid, make_optimizer
from .core_a2_dho import train_oscillator_pinn, u_exact
from ..run_common import run_series_inference_mode
from ..layer_pennylane import make_quantum_block, dho_feature_map, BranchPennylane


class PP_PINN(nn.Module):
    """
    Physics-Informed model with two independent quantum branches
    and a linear fusion to scalar output.
    """

    def __init__(
        self,
        *,
        n_qubits: int = DEFAULT_N_OUTPUTS,
    ) -> None:
        super().__init__()
        qblock1 = make_quantum_block(n_qubits=n_qubits)
        qblock2 = make_quantum_block(n_qubits=n_qubits)

        # Two distinct branches => two independent parameter sets
        self.branch1 = BranchPennylane(
            qblock1,
            feature_map=lambda t: dho_feature_map(t, n_qubits=n_qubits),
            output_as_column=True,
            n_layers=N_LAYERS,
            n_qubits=n_qubits,
        )
        self.branch2 = BranchPennylane(
            qblock2,
            feature_map=lambda t: dho_feature_map(t, n_qubits=n_qubits),
            output_as_column=True,
            n_layers=N_LAYERS,
            n_qubits=n_qubits,
        )
        self.fusion = nn.Linear(2, 1, dtype=DTYPE)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        out1 = self.branch1(t)
        out2 = self.branch2(t)
        return self.fusion(torch.cat([out1, out2], dim=1))


def plot_model_prediction(u_pred, u_ex, t, save_path="HQPINN/DHO/results/dho_pp/"):
    plt.figure(figsize=(10, 6))
    plt.plot(t.cpu().numpy(), u_pred, label="Prediction PINN", lw=2)
    plt.plot(t.cpu().numpy(), u_ex, "--", label="Exact solution", lw=2)
    plt.xlabel("t")
    plt.ylabel("u(t)")
    plt.title("DHO - PennyLane-PennyLane PINN")
    plt.grid(True)
    plt.legend()

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs(save_path, exist_ok=True)
    png_path = os.path.join(save_path, f"dho_pp_plot_{timestamp}.png")
    plt.savefig(png_path, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to: {png_path}")


def _case_prefix(n_qubits: int) -> str:
    if n_qubits == DEFAULT_N_OUTPUTS:
        return "dho_pp"
    return f"dho_pp_q{n_qubits}"


def run(
    mode="train",
    backend="sim:ascella",
    *,
    n_qubits: int = DEFAULT_N_OUTPUTS,
):
    torch.manual_seed(0)
    ckpt_dir = "HQPINN/DHO/models"
    case_prefix = _case_prefix(n_qubits)
    results_dir = f"HQPINN/DHO/results/{case_prefix}"
    if mode == "train":
        model = PP_PINN(n_qubits=n_qubits)
        train_oscillator_pinn(
            model=model,
            t_train=make_time_grid(),
            optimizer=make_optimizer(model, lr=DHO_LR),
            n_epochs=DHO_N_EPOCHS,
            plot_every=DHO_PLOT_EVERY,
            out_dir=results_dir,
            model_label="pp",
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
            model_factory=lambda processor=None: PP_PINN(n_qubits=n_qubits),
            make_time_grid=make_time_grid,
            exact_fn=u_exact,
            plot_fn=lambda u_pred, u_ex, t: plot_model_prediction(
                u_pred, u_ex, t, save_path=results_dir
            ),
        )

    elif mode == "remote":
        print("Remote mode is not available for DHO-PP. Falling back to local run mode.")
        run_series_inference_mode(
            mode="run",
            backend="local",
            ckpt_dir=ckpt_dir,
            case_prefix=case_prefix,
            model_factory=lambda processor=None: PP_PINN(n_qubits=n_qubits),
            make_time_grid=make_time_grid,
            exact_fn=u_exact,
            plot_fn=lambda u_pred, u_ex, t: plot_model_prediction(
                u_pred, u_ex, t, save_path=results_dir
            ),
        )

    else:
        raise ValueError("mode must be 'train', 'run', or 'remote'")
