# see_cc.py
# Classical–Classical PINN

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
)
from ..utils import make_time_grid, make_optimizer
from .core_see import train_see
from ..layer_classical import BranchPyTorch


class CC_PINN(nn.Module):
    """
    Classical-classical PINN with two parallel branches (cc-N-L)
    and a small fusion head (845 trainable parameters for cc-10-4).
    """

    def __init__(
        self,
        hidden_width: int = SEE_CC_HIDDEN_WIDTH,
        num_hidden_layers: int = SEE_CC_NUM_HIDDEN_LAYERS,
    ) -> None:
        super().__init__()

        # Two parallel classical branches: each (x,t) -> (rho,u,p)
        self.branch1 = BranchPyTorch(
            in_features=2,
            out_features=3,
            num_hidden_layers=num_hidden_layers,
            hidden_width=hidden_width,
        )
        self.branch2 = BranchPyTorch(
            in_features=2,
            out_features=3,
            num_hidden_layers=num_hidden_layers,
            hidden_width=hidden_width,
        )

        # Fusion head: combines outputs of both branches
        self.fusion = nn.Sequential(
            nn.Linear(3, 8, dtype=DTYPE),
            nn.Tanh(),
            nn.Linear(8, 3, dtype=DTYPE),
        )

        # Human-readable size label, e.g. "10-4"
        self.size_label = f"{hidden_width}-{num_hidden_layers}"

    def forward(self, xt: torch.Tensor) -> torch.Tensor:
        # Forward pass: sum two branches then apply fusion head
        out1 = self.branch1(xt)
        out2 = self.branch2(xt)
        combined = out1 + out2
        return self.fusion(combined)


MODELS = [
    ("10-4", 10, 4),
    ("10-7", 10, 7),
    ("20-4", 20, 4),
]


def run():
    """Run all SEE classical–classical models and write summary CSV."""
    torch.manual_seed(0)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_csv = f"HQPINN/SEE/results/cc_summary_{timestamp}.csv"

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

        for label, width, layers in MODELS:
            print(f"\nTraining SEE-CC model: {label} (width={width}, layers={layers})")

            model = CC_PINN(hidden_width=width, num_hidden_layers=layers)
            optimizer = make_optimizer(model, lr=5e-4)

            final_loss, err_rho, err_p, n_params = train_see(
                model=model,
                t_train=make_time_grid(),  # kept for API consistency
                optimizer=optimizer,
                n_epochs=SEE_N_EPOCHS,
                plot_every=SEE_PLOT_EVERY,
                out_dir=f"HQPINN/SEE/results/cc-{label}",
                model_label=label,
            )

            writer.writerow(
                [
                    "cc",
                    label,
                    n_params,
                    f"{final_loss:.6e}",
                    f"{err_rho:.6e}",
                    f"{err_p:.6e}",
                ]
            )

    print(f"Summary CSV saved to: {out_csv}")
