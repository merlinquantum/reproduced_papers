# see_ci.py
# Classical–Interferometer PINN

import os
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
from ..utils import make_optimizer, get_latest_checkpoint, load_model
from .core_see import train_see, save_density_plot
from ..layer_classical import BranchPyTorch
from ..layer_merlin import make_interf_qlayer, BranchMerlin, make_merlin_processor


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
        processor=None,
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
            make_interf_qlayer(n_photons=n_photons), n_outputs=3, processor=processor
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
    ("10-4-2", 10, 4, 1),
    ("10-7-2", 10, 7, 1),
    ("20-4-2", 20, 4, 1),
]


def _get_model_config(model_size: str) -> tuple[str, int, int, int]:
    for label, width, layers, n_photons in MODELS:
        if label == model_size:
            return label, width, layers, n_photons
    valid = ", ".join(label for label, *_ in MODELS)
    raise ValueError(f"Unknown model_size='{model_size}'. Valid values: {valid}")


def run(
    mode="train", backend="sim:ascella", model_size="10-4-2", rpc_timeout_s=None
):
    """Run SEE Classical-Interferometer models and write summary CSV."""
    torch.manual_seed(0)

    ckpt_dir = "HQPINN/SEE/"
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    if mode == "train":
        print("=== TRAINING MODE ===")
        out_csv = f"HQPINN/SEE/results/ci_summary_{timestamp}.csv"
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
                    f"\nTraining SEE-CI model: {label} (width={width}, layers={layers}, {n_photons} photons)"
                )

                case_prefix = f"see_ci_{label}"
                model = CI_PINN(
                    n_photons=n_photons, hidden_width=width, num_hidden_layers=layers
                )
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
        case_prefix = f"see_ci_{label}"
        model_root = os.path.join(ckpt_dir, "models")
        ckpt_path = get_latest_checkpoint(model_root, case_prefix)
        if ckpt_path is None:
            print("No trained checkpoint found!")
            return

        print(f"Latest checkpoint found: {ckpt_path}")

        def model_proc_local(processor=None):
            return CI_PINN(
                n_photons=n_photons,
                hidden_width=width,
                num_hidden_layers=layers,
                processor=processor,
            )

        if backend.lower() != "local":
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

    elif mode == "remote":
        print("=== REMOTE MODE ===")
        label, width, layers, n_photons = _get_model_config(model_size)
        case_prefix = f"see_ci_{label}"
        model_root = os.path.join(ckpt_dir, "models")
        ckpt_path = get_latest_checkpoint(model_root, case_prefix)
        if ckpt_path is None:
            print("No trained checkpoint found!")
            return

        print(f"Latest checkpoint found: {ckpt_path}")

        if backend.lower() == "local":
            backend = "sim:ascella"

        processor = make_merlin_processor(backend, rpc_timeout_s=rpc_timeout_s)

        def model_proc_remote(processor=processor):
            return CI_PINN(
                n_photons=n_photons,
                hidden_width=width,
                num_hidden_layers=layers,
                processor=processor,
            )

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
