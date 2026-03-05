# taf_ci.py
# Classical–Interferometer PINN for TAF (Sec. 3.3)

import csv
import os
from datetime import datetime

import torch
import torch.nn as nn

from ..config import (
    DEVICE,
    DTYPE,
    TAF_ADAM_STEPS,
    TAF_CC_HIDDEN_WIDTH,
    TAF_CC_NUM_HIDDEN_LAYERS,
    TAF_EPSILON_LAMBDA,
    TAF_LBFGS_STEPS,
    TAF_LR,
    TAF_PLOT_EVERY,
)
from ..layer_classical import BranchPyTorch
from ..layer_merlin import BranchMerlin, make_interf_qlayer
from ..utils import make_optimizer
from .core_taf import load_training_sets, train_taf


class CI_PINN(nn.Module):
    """Classical-Interferometer TAF PINN with independent branch parameters."""

    def __init__(
        self,
        n_photons: int,
        hidden_width: int = TAF_CC_HIDDEN_WIDTH,
        num_hidden_layers: int = TAF_CC_NUM_HIDDEN_LAYERS,
        processor=None,
    ) -> None:
        super().__init__()

        self.branch1 = BranchPyTorch(
            in_features=2,
            out_features=4,
            num_hidden_layers=num_hidden_layers,
            hidden_width=hidden_width,
        )
        self.branch2 = BranchMerlin(
            make_interf_qlayer(n_photons=n_photons),
            n_outputs=4,
            processor=processor,
            feature_map_kind="taf",
        )

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        return self.branch1(xy) + self.branch2(xy)


MODELS = [
    ("40-4-2", 40, 4, 2),
    ("40-7-2", 40, 7, 2),
    ("80-4-2", 80, 4, 2),
]


def run() -> None:
    """Run TAF classical-interferometer models and write summary CSV."""
    torch.manual_seed(0)

    data = load_training_sets()

    # Sec. 3.3 inlet values (SI)
    U_in = torch.tensor([1.225, 272.15, 0.0, 288.15], dtype=DTYPE, device=DEVICE)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs("HQPINN/TAF/results", exist_ok=True)
    out_csv = f"HQPINN/TAF/results/ci_summary_{timestamp}.csv"

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Model",
                "Size",
                "Trainable parameters",
                "Loss",
                "Boundary loss",
                "PDE loss",
            ]
        )

        for label, width, layers, n_photons in MODELS:
            print(
                f"\nTraining TAF-CI model: {label} "
                f"(width={width}, layers={layers}, {n_photons} photons)"
            )

            case_prefix = f"taf_ci_{label}"
            model = CI_PINN(
                n_photons=n_photons,
                hidden_width=width,
                num_hidden_layers=layers,
            ).to(DEVICE)
            optimizer = make_optimizer(model, lr=TAF_LR)

            final_loss, loss_bc, loss_f, n_params = train_taf(
                model=model,
                optimizer=optimizer,
                n_epochs=TAF_ADAM_STEPS,
                plot_every=TAF_PLOT_EVERY,
                out_dir=f"HQPINN/TAF/results/{case_prefix}",
                model_label=f"ci_{label}",
                data=data,
                U_in=U_in,
                lbfgs_steps=TAF_LBFGS_STEPS,
                eps_lambda=TAF_EPSILON_LAMBDA,
            )

            writer.writerow(
                [
                    "ci",
                    label,
                    n_params,
                    f"{final_loss:.6e}",
                    f"{loss_bc:.6e}",
                    f"{loss_f:.6e}",
                ]
            )

    print(f"Summary CSV saved to: {out_csv}")
