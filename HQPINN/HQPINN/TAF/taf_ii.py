# taf_ii.py
# Interferometer–Interferometer PINN for TAF (Sec. 3.3)

import csv
import os
from datetime import datetime

import torch
import torch.nn as nn

from ..config import (
    DEVICE,
    DTYPE,
    TAF_ADAM_STEPS,
    TAF_EPSILON_LAMBDA,
    TAF_LBFGS_STEPS,
    TAF_LR,
    TAF_PLOT_EVERY,
)
from ..layer_merlin import BranchMerlin, make_interf_qlayer
from ..run_common import run_density_inference_mode
from ..utils import make_optimizer
from .core_taf import (
    load_training_sets,
    save_density_plot,
    train_taf,
)


class II_PINN(nn.Module):
    """Interferometer-Interferometer TAF PINN with two independent quantum branches."""

    def __init__(self, n_photons: int, processor=None) -> None:
        super().__init__()

        self.branch1 = BranchMerlin(
            make_interf_qlayer(n_photons=n_photons),
            n_outputs=4,
            processor=processor,
            feature_map_kind="taf",
        )
        self.branch2 = BranchMerlin(
            make_interf_qlayer(n_photons=n_photons),
            n_outputs=4,
            processor=processor,
            feature_map_kind="taf",
        )
        self.fusion = nn.Linear(8, 4, dtype=DTYPE)

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        # Learned linear fusion of both branch outputs.
        out1 = self.branch1(xy)
        out2 = self.branch2(xy)
        return self.fusion(torch.cat([out1, out2], dim=1))


MODELS = [
    ("1", 1),
    ("2", 2),
    ("3", 3),
    ("4", 4),
    ("5", 5),
    ("6", 6),
]

def run(mode="train", backend="sim:ascella", n_photons=2) -> None:
    """Run TAF interferometer-interferometer models and write summary CSV."""
    torch.manual_seed(0)

    data = load_training_sets()

    # Sec. 3.3 inlet values (SI)
    U_in = torch.tensor([1.225, 272.15, 0.0, 288.15], dtype=DTYPE, device=DEVICE)

    ckpt_dir = "HQPINN/TAF/"
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if mode == "train":
        os.makedirs("HQPINN/TAF/results", exist_ok=True)
        out_csv = f"HQPINN/TAF/results/ii_summary_{timestamp}.csv"

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

            for label, n_photons_sel in MODELS:
                print(f"\nTraining TAF-II model: {label} photons")

                case_prefix = f"taf_ii_{label}"
                model = II_PINN(n_photons=n_photons_sel).to(DEVICE)
                optimizer = make_optimizer(model, lr=TAF_LR)

                final_loss, loss_bc, loss_f, n_params = train_taf(
                    model=model,
                    optimizer=optimizer,
                    n_epochs=TAF_ADAM_STEPS,
                    plot_every=TAF_PLOT_EVERY,
                    out_dir=f"HQPINN/TAF/results/{case_prefix}",
                    model_label=f"ii_{label}",
                    data=data,
                    U_in=U_in,
                    lbfgs_steps=TAF_LBFGS_STEPS,
                    eps_lambda=TAF_EPSILON_LAMBDA,
                )

                writer.writerow(
                    [
                        "ii",
                        label,
                        n_params,
                        f"{final_loss:.6e}",
                        f"{loss_bc:.6e}",
                        f"{loss_f:.6e}",
                    ]
                )

                model_dir = os.path.join(ckpt_dir, "models")
                os.makedirs(model_dir, exist_ok=True)
                ckpt_path = os.path.join(model_dir, f"{case_prefix}_{timestamp}.pt")
                torch.save(model.state_dict(), ckpt_path)
                print(f"Model saved to: {ckpt_path}")

        print(f"Summary CSV saved to: {out_csv}")

    elif mode == "run":
        case_prefix = f"taf_ii_{n_photons}"
        run_density_inference_mode(
            mode="run",
            backend="local",
            ckpt_dir=ckpt_dir,
            case_prefix=case_prefix,
            n_photons=n_photons,
            timestamp=timestamp,
            model_factory=lambda processor=None: II_PINN(
                n_photons=n_photons, processor=processor
            ),
            save_plot_fn=save_density_plot,
        )

    elif mode == "remote":
        case_prefix = f"taf_ii_{n_photons}"
        run_density_inference_mode(
            mode="remote",
            backend=backend,
            ckpt_dir=ckpt_dir,
            case_prefix=case_prefix,
            n_photons=n_photons,
            timestamp=timestamp,
            model_factory=lambda processor=None: II_PINN(
                n_photons=n_photons, processor=processor
            ),
            save_plot_fn=save_density_plot,
        )

    else:
        raise ValueError("mode must be 'train', 'run', or 'remote'")
