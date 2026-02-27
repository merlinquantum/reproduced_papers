# dee_ci.py
# Classical–Interferometer PINN for DEE

import os
import csv
from datetime import datetime

import torch
import torch.nn as nn

from ..config import (
    DEE_CC_NUM_HIDDEN_LAYERS,
    DEE_CC_HIDDEN_WIDTH,
    DTYPE,
    DEE_N_EPOCHS,
    DEE_PLOT_EVERY,
    DEE_LR,
)
from ..utils import make_optimizer
from .core_dee import train_dee, save_density_plot
from ..run_common import run_density_inference_mode
from ..layer_classical import BranchPyTorch
from ..layer_merlin import make_interf_qlayer, BranchMerlin


class CI_PINN(nn.Module):
    """
    Classical-Interferometer PINN with one classical branch and one quantum branch.
    """

    def __init__(
        self,
        n_photons: int,
        hidden_width: int = DEE_CC_HIDDEN_WIDTH,
        num_hidden_layers: int = DEE_CC_NUM_HIDDEN_LAYERS,
        processor=None,
    ) -> None:
        super().__init__()

        self.branch1 = BranchPyTorch(
            in_features=2,
            out_features=3,
            num_hidden_layers=num_hidden_layers,
            hidden_width=hidden_width,
        )
        self.branch2 = BranchMerlin(
            make_interf_qlayer(n_photons=n_photons),
            n_outputs=3,
            processor=processor,
            feature_map_kind="dee",
        )

        self.fusion = nn.Sequential(
            nn.Linear(3, 8, dtype=DTYPE),
            nn.Tanh(),
            nn.Linear(8, 3, dtype=DTYPE),
        )

        self.size_label = f"{hidden_width}-{num_hidden_layers}"

    def forward(self, xt: torch.Tensor) -> torch.Tensor:
        out1 = self.branch1(xt)
        out2 = self.branch2(xt)
        return self.fusion(out1 + out2)


MODELS = [
    ("10-4-1", 10, 4, 1),
    ("10-7-1", 10, 7, 1),
    ("20-4-1", 20, 4, 1),
]


def _get_model_config(model_size: str) -> tuple[str, int, int, int]:
    for label, width, layers, n_photons in MODELS:
        if label == model_size:
            return label, width, layers, n_photons
    valid = ", ".join(label for label, *_ in MODELS)
    raise ValueError(f"Unknown model_size='{model_size}'. Valid values: {valid}")


def run(mode="train", backend="sim:ascella", model_size="10-4-1"):
    """Run DEE Classical-Interferometer models and write summary CSV."""
    torch.manual_seed(0)

    ckpt_dir = "HQPINN/DEE/"
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    if mode == "train":
        print("=== TRAINING MODE ===")
        out_csv = f"HQPINN/DEE/results/ci_summary_{timestamp}.csv"
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
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
                    f"\nTraining DEE-CI model: {label} (width={width}, layers={layers}, {n_photons} photons)"
                )

                case_prefix = f"dee_ci_{label}"
                model = CI_PINN(
                    n_photons=n_photons, hidden_width=width, num_hidden_layers=layers
                )
                optimizer = make_optimizer(model, lr=DEE_LR)

                final_loss, err_rho, err_p, n_params = train_dee(
                    model=model,
                    t_train=None,
                    optimizer=optimizer,
                    n_epochs=DEE_N_EPOCHS,
                    plot_every=DEE_PLOT_EVERY,
                    out_dir=f"HQPINN/DEE/results/{case_prefix}",
                    model_label=case_prefix,
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

                model_dir = os.path.join(ckpt_dir, "models")
                os.makedirs(model_dir, exist_ok=True)
                ckpt_path = os.path.join(model_dir, f"{case_prefix}_{timestamp}.pt")
                torch.save(model.state_dict(), ckpt_path)
                print(f"Model saved to: {ckpt_path}")

        print(f"Summary CSV saved to: {out_csv}")

    elif mode == "run":
        label, width, layers, n_photons = _get_model_config(model_size)
        case_prefix = f"dee_ci_{label}"
        run_density_inference_mode(
            mode="run",
            backend=backend,
            ckpt_dir=ckpt_dir,
            case_prefix=case_prefix,
            n_photons=n_photons,
            timestamp=timestamp,
            model_factory=lambda processor=None: CI_PINN(
                n_photons=n_photons,
                hidden_width=width,
                num_hidden_layers=layers,
                processor=processor,
            ),
            save_plot_fn=save_density_plot,
        )

    elif mode == "remote":
        label, width, layers, n_photons = _get_model_config(model_size)
        case_prefix = f"dee_ci_{label}"
        run_density_inference_mode(
            mode="remote",
            backend=backend,
            ckpt_dir=ckpt_dir,
            case_prefix=case_prefix,
            n_photons=n_photons,
            timestamp=timestamp,
            model_factory=lambda processor=None: CI_PINN(
                n_photons=n_photons,
                hidden_width=width,
                num_hidden_layers=layers,
                processor=processor,
            ),
            save_plot_fn=save_density_plot,
        )

    else:
        raise ValueError("mode must be 'train', 'run', or 'remote'")
