# see-ii.py
# Interferometer-Interferometer PINN for the damped oscillator using oscillator_core + merlin_quantum

import os
from datetime import datetime
import csv

import torch
import torch.nn as nn

from ..config import (
    SEE_N_EPOCHS,
    SEE_LR,
    SEE_PLOT_EVERY,
    DTYPE,
)
from ..utils import make_optimizer
from .core_see import train_see, save_density_plot
from ..run_common import run_density_inference_mode
from ..layer_merlin import make_interf_qlayer, BranchMerlin


class II_PINN(nn.Module):
    """
    Interferometer-Interferometer PINN:

        u(t) = u_q1(t) + u_q2(t)

    Each branch uses its own QuantumLayer instance → independent parameters.
    """

    def __init__(self, n_photons: int, processor=None) -> None:
        super().__init__()

        # Two distinct quantum branches with independent parameters
        self.branch1 = BranchMerlin(
            make_interf_qlayer(n_photons=n_photons),
            n_outputs=3,
            processor=processor,
            feature_map_kind="see",
        )
        self.branch2 = BranchMerlin(
            make_interf_qlayer(n_photons=n_photons),
            n_outputs=3,
            processor=processor,
            feature_map_kind="see",
        )

        # Fusion head: combines outputs of both branches into (rho, u, p)
        self.fusion = nn.Sequential(
            nn.Linear(3, 8, dtype=DTYPE),
            nn.Tanh(),
            nn.Linear(8, 3, dtype=DTYPE),
        )

        # Human-readable size label ("1", "2", ..., "6")
        self.size_label = f"{n_photons}"

    def forward(self, xt: torch.Tensor) -> torch.Tensor:
        # Forward pass: sum two quantum branches then apply fusion head
        out1 = self.branch1(xt)  # [N, 3]
        out2 = self.branch2(xt)  # [N, 3]
        combined = out1 + out2  # [N, 3]
        return self.fusion(combined)  # [N, 3]


MODELS = [
    ("1", 1),
    ("2", 2),
    ("3", 3),
    ("4", 4),
    ("5", 5),
    ("6", 6),
]


def run(mode="train", backend="sim:ascella", n_photons=2):
    """Run all SEE Interferometer-Interferometer models and write summary CSV."""
    torch.manual_seed(0)

    ckpt_dir = "HQPINN/SEE/"
    # case_prefix = f"see_ii_{n_photons}"
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    # ======================
    #  MODE TRAIN
    # ======================

    if mode == "train":
        print("=== TRAINING MODE ===")
        out_csv = f"HQPINN/SEE/results/ii_summary_{timestamp}.csv"
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

            for label, nb_photons in MODELS:
                print(f"\nTraining SEE-II {nb_photons} photons")

                case_prefix = f"see_ii_{nb_photons}"
                model = II_PINN(n_photons=nb_photons)
                optimizer = make_optimizer(model, lr=SEE_LR)

                final_loss, err_rho, err_p, n_params = train_see(
                    model=model,
                    t_train=None,  # kept for API consistency
                    optimizer=optimizer,
                    n_epochs=SEE_N_EPOCHS,
                    plot_every=SEE_PLOT_EVERY,
                    out_dir=f"HQPINN/SEE/results/{case_prefix}",
                    model_label=case_prefix,
                )

                writer.writerow(
                    [
                        "ii",  # model type: Interferometer-Interferometer
                        label,  # size label ("1", "2", ..., "6")
                        n_params,
                        f"{final_loss:.6e}",
                        f"{err_rho:.6e}",
                        f"{err_p:.6e}",
                    ]
                )

                # === Save model ===
                model_dir = os.path.join(ckpt_dir, "models")
                os.makedirs(model_dir, exist_ok=True)
                # timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                ckpt_path = os.path.join(model_dir, f"{case_prefix}_{timestamp}.pt")
                torch.save(model.state_dict(), ckpt_path)

                print(f"Model saved to: {ckpt_path}")

        print(f"Summary CSV saved to: {out_csv}")

    # ======================
    #  MODE RUN
    # ======================

    elif mode == "run":
        case_prefix = f"see_ii_{n_photons}"
        run_density_inference_mode(
            mode="run",
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

    # ======================
    #  MODE RUN REMOTE
    # ======================

    elif mode == "remote":
        case_prefix = f"see_ii_{n_photons}"
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
