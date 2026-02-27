# dee-ii.py
# Interferometer-Interferometer PINN

import os
from datetime import datetime
import csv

import torch
import torch.nn as nn

from ..config import (
    DEE_N_EPOCHS,
    DEE_LR,
    DEE_PLOT_EVERY,
    DTYPE,
)
from ..utils import make_optimizer, get_latest_checkpoint, load_model
from .core_dee import train_dee, save_density_plot
from ..layer_merlin import make_interf_qlayer, BranchMerlin, make_merlin_processor


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
            make_interf_qlayer(n_photons=n_photons), n_outputs=3, processor=processor
        )
        self.branch2 = BranchMerlin(
            make_interf_qlayer(n_photons=n_photons), n_outputs=3, processor=processor
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


def run(mode="train", backend="sim:ascella", n_photons=2, rpc_timeout_s=None):
    """Run all DEE Interferometer-Interferometer models and write summary CSV."""
    torch.manual_seed(0)

    ckpt_dir = "HQPINN/DEE/"
    # case_prefix = f"see_ii_{n_photons}"
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    # ======================
    #  MODE TRAIN
    # ======================

    if mode == "train":
        print("=== TRAINING MODE ===")
        out_csv = f"HQPINN/DEE/results/ii_summary_{timestamp}.csv"
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
                print(f"\nTraining DEE-II {nb_photons} photons")

                case_prefix = f"dee_ii_{nb_photons}"
                model = II_PINN(n_photons=nb_photons)
                optimizer = make_optimizer(model, lr=DEE_LR)

                final_loss, err_rho, err_p, n_params = train_dee(
                    model=model,
                    t_train=None,  # kept for API consistency
                    optimizer=optimizer,
                    n_epochs=DEE_N_EPOCHS,
                    plot_every=DEE_PLOT_EVERY,
                    out_dir=f"HQPINN/DEE/results/{case_prefix}",
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

        case_prefix = f"dee_ii_{n_photons}"
        model_root = os.path.join(ckpt_dir, "models")
        ckpt_path = get_latest_checkpoint(model_root, case_prefix)
        if ckpt_path is None:
            print("No trained checkpoint found!")
            return

        print(f"Latest checkpoint found: {ckpt_path}")

        def model_proc_local(processor=None):
            return II_PINN(n_photons=n_photons, processor=processor)

        # model = load_model(ckpt_path, model_proc_local)

        if backend.lower() == "local":
            model = load_model(ckpt_path, model_proc_local)
        else:
            print(
                f"Backend '{backend}' n’est pas utilisé en mode run; for remote use mode='remote'."
            )
            model = load_model(ckpt_path, model_proc_local)

        model.eval()

        png_path = save_density_plot(
            model=model,
            ckpt_dir=ckpt_dir,
            case_prefix=case_prefix,
            n_photons=n_photons,
            timestamp=timestamp,
            backend=backend,
        )

        print(f"Figure saved to: {png_path}")

    # ======================
    #  MODE RUN REMOTE
    # ======================

    elif mode == "remote":
        print("=== REMOTE MODE ===")

        case_prefix = f"dee_ii_{n_photons}"

        model_root = os.path.join(ckpt_dir, "models")
        ckpt_path = get_latest_checkpoint(model_root, case_prefix)
        if ckpt_path is None:
            print("No trained checkpoint found!")
            return

        print(f"Latest checkpoint found: {ckpt_path}")

        if backend.lower() == "local":
            backend = "sim:ascella"  # local → simulateur Perceval

        processor = make_merlin_processor(backend)

        def model_proc_remote(processor=processor):
            return II_PINN(n_photons=n_photons, processor=processor)

        model_remote = load_model(ckpt_path, model_proc_remote, processor=processor)

        model_remote.eval()

        png_path = save_density_plot(
            model=model_remote,
            ckpt_dir=ckpt_dir,
            case_prefix=case_prefix,
            n_photons=n_photons,
            timestamp=timestamp,
            backend=backend,
        )

        print(f"Figure saved to: {png_path}")

    else:
        raise ValueError("mode must be 'train', 'run', or 'remote'")
