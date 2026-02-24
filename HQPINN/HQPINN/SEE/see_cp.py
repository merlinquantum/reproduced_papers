# see_cp.py
# Classical–PennyLane PINN

import csv
from datetime import datetime

import torch
import torch.nn as nn

from ..config import (
    SEE_CC_NUM_HIDDEN_LAYERS,
    SEE_CC_HIDDEN_WIDTH,
    DTYPE,
    SEE_N_EPOCHS,
    SEE_PLOT_EVERY,
    N_LAYERS,
)
from ..utils import make_time_grid, make_optimizer
from .core_see import train_see
from ..layer_pennylane import (
    make_quantum_block_multiout,
    see_feature_map,
    BranchPennylane,
)
from ..layer_classical import BranchPyTorch


class CP_PINN(nn.Module):
    """
    Classical–PennyLane PINN with one quantum branch and one classical MLP branch.
    """

    def __init__(
        self,
        size: int = N_LAYERS,
        hidden_width: int = SEE_CC_HIDDEN_WIDTH,
        num_hidden_layers: int = SEE_CC_NUM_HIDDEN_LAYERS,
    ) -> None:
        super().__init__()

        qblock_multi_1 = make_quantum_block_multiout(n_layers=size)

        # Two parallel PennyLane branches: each (x,t) -> (rho_like, u_like, p_like)
        self.branch1 = BranchPennylane(
            qblock_multi_1,
            feature_map=see_feature_map,
            output_as_column=False,
            n_layers=size,
        )
        self.branch2 = BranchPyTorch(
            in_features=2,
            out_features=3,
            num_hidden_layers=num_hidden_layers,
            hidden_width=hidden_width,
        )

        # Fusion head: combines outputs of both branches into (rho, u, p)
        self.fusion = nn.Sequential(
            nn.Linear(3, 8, dtype=DTYPE),
            nn.Tanh(),
            nn.Linear(8, 3, dtype=DTYPE),
        )

        # Human-readable size label (e.g. "2", "3", "4")
        self.size_label = f"{size}"

    def forward(self, xt: torch.Tensor) -> torch.Tensor:
        # Forward pass: sum two quantum branches then apply fusion head
        out1 = self.branch1(xt)  # [N, 3]
        out2 = self.branch2(xt)  # [N, 3]
        combined = out1 + out2  # [N, 3]
        return self.fusion(combined)  # [N, 3]


MODELS = [
    ("10-4-2", 10, 4, 2),
    ("10-7-2", 10, 7, 2),
    ("20-4-2", 20, 4, 2),
]


def run():
    """Run all SEE PennyLane–PennyLane models and write summary CSV."""
    torch.manual_seed(0)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_csv = f"HQPINN/SEE/results/pp_summary_{timestamp}.csv"

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Model",
                "Size",
                "Trainable parameters",
                "Loss",
                "Density error",
                "Pressure error",
            ]
        )

        for label, width, layers, size in MODELS:
            print(f"\nTraining SEE-CP model: {label} size={size}")

            model = CP_PINN(size=size, hidden_width=width, num_hidden_layers=layers)
            optimizer = make_optimizer(model, lr=5e-4)

            final_loss, err_rho, err_p, n_params = train_see(
                model=model,
                t_train=make_time_grid(),  # kept for API consistency
                optimizer=optimizer,
                n_epochs=SEE_N_EPOCHS,
                plot_every=SEE_PLOT_EVERY,
                out_dir=f"HQPINN/SEE/results/cp-{label}",
                model_label=f"cp-{label}",
            )

            writer.writerow(
                [
                    "cp",  # model type: Classical–PennyLane
                    label,  # size label ("2", "3", "4")
                    n_params,
                    f"{final_loss:.6e}",
                    f"{err_rho:.6e}",
                    f"{err_p:.6e}",
                ]
            )

    print(f"Summary CSV saved to: {out_csv}")
