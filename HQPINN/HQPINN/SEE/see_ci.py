# see_ci.py
# Classical–Interferometer PINN

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
    SEE_LR,
)
from ..utils import make_time_grid, make_optimizer
from .core_see import train_see
from ..layer_classical import BranchPyTorch
from ..layer_merlin import make_interf_qlayer, BranchMerlin


class CI_PINN(nn.Module):
    """
    Classical-Interferometer PINN with one classical branch and one quantum branch.
    The quantum branch is a MerLin interferometer with independent parameters.
    """

    def __init__(
        self,
        n_photons: int,
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
        self.branch2 = BranchMerlin(
            make_interf_qlayer(n_photons=n_photons), n_outputs=3
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
    ("10-4-2", 10, 4, 2),
    ("10-7-2", 10, 7, 2),
    ("20-4-2", 20, 4, 2),
]


def run():
    """Run all SEE Classical–Interferometer models and write summary CSV."""
    torch.manual_seed(0)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_csv = f"HQPINN/SEE/results/ci_summary_{timestamp}.csv"
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

        for label, width, layers, n_photons in MODELS:
            print(
                f"\nTraining SEE-CI model: {label} (width={width}, layers={layers}, {n_photons} photons)"
            )

            model = CI_PINN(
                n_photons=n_photons, hidden_width=width, num_hidden_layers=layers
            )
            optimizer = make_optimizer(model, lr=SEE_LR)

            final_loss, err_rho, err_p, n_params = train_see(
                model=model,
                t_train=make_time_grid(),  # kept for API consistency
                optimizer=optimizer,
                n_epochs=SEE_N_EPOCHS,
                plot_every=SEE_PLOT_EVERY,
                out_dir=f"HQPINN/SEE/results/ci-{label}",
                model_label=label,
            )

            writer.writerow(
                [
                    "ci",
                    label,
                    n_params,
                    f"{final_loss:.6e}",
                    f"{err_rho:.6e}",
                    f"{err_p:.6e}",
                ]
            )

    print(f"Summary CSV saved to: {out_csv}")
