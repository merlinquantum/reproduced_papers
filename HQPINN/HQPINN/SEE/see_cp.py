# see_cp.py
# Classical–PennyLane PINN

import csv
import os
from datetime import datetime

import torch
import torch.nn as nn

from ..config import (
    SEE_CC_NUM_HIDDEN_LAYERS,
    SEE_CC_HIDDEN_WIDTH,
    SEE_N_EPOCHS,
    SEE_PLOT_EVERY,
    N_LAYERS,
    DTYPE,
)
from ..utils import make_optimizer
from .core_see import save_density_plot, train_see
from ..run_common import run_density_inference_mode
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
        self.fusion = nn.Linear(6, 3, dtype=DTYPE)

        # Human-readable size label (e.g. "2", "3", "4")
        self.size_label = f"{size}"

    def forward(self, xt: torch.Tensor) -> torch.Tensor:
        # Learned linear fusion of both branch outputs.
        out1 = self.branch1(xt)  # [N, 3]
        out2 = self.branch2(xt)  # [N, 3]
        return self.fusion(torch.cat([out1, out2], dim=1))  # [N, 3]


MODELS = [
    ("10-4-2", 10, 4, 2),
    ("10-7-2", 10, 7, 2),
    ("20-4-2", 20, 4, 2),
]


def _get_model_config(model_size: str) -> tuple[str, int, int, int]:
    for label, width, layers, size in MODELS:
        if label == model_size:
            return label, width, layers, size
    valid = ", ".join(label for label, *_ in MODELS)
    raise ValueError(f"Unknown model_size='{model_size}'. Valid values: {valid}")


def run(mode="train", backend="sim:ascella", model_size="10-4-2"):
    """Run all SEE Classical models and write summary CSV."""
    torch.manual_seed(0)

    ckpt_dir = "HQPINN/SEE/"
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if mode == "train":
        out_csv = f"HQPINN/SEE/results/cp_summary_{timestamp}.csv"
        os.makedirs("HQPINN/SEE/results", exist_ok=True)

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

                case_prefix = f"see_cp_{label}"
                model = CP_PINN(size=size, hidden_width=width, num_hidden_layers=layers)
                optimizer = make_optimizer(model, lr=5e-4)

                final_loss, err_rho, err_p, n_params = train_see(
                    model=model,
                    t_train=None,  # kept for API consistency
                    optimizer=optimizer,
                    n_epochs=SEE_N_EPOCHS,
                    plot_every=SEE_PLOT_EVERY,
                    out_dir=f"HQPINN/SEE/results/{case_prefix}",
                    model_label=f"cp_{label}",
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

                model_dir = os.path.join(ckpt_dir, "models")
                os.makedirs(model_dir, exist_ok=True)
                ckpt_path = os.path.join(model_dir, f"{case_prefix}_{timestamp}.pt")
                torch.save(model.state_dict(), ckpt_path)
                print(f"Model saved to: {ckpt_path}")

        print(f"Summary CSV saved to: {out_csv}")

    elif mode == "run":
        label, width, layers, size = _get_model_config(model_size)
        case_prefix = f"see_cp_{label}"
        run_density_inference_mode(
            mode="run",
            backend="local",
            ckpt_dir=ckpt_dir,
            case_prefix=case_prefix,
            n_photons=size,
            timestamp=timestamp,
            model_factory=lambda processor=None: CP_PINN(
                size=size,
                hidden_width=width,
                num_hidden_layers=layers,
            ),
            save_plot_fn=save_density_plot,
        )

    elif mode == "remote":
        print("Remote mode is not available for SEE-CP. Falling back to local run mode.")
        label, width, layers, size = _get_model_config(model_size)
        case_prefix = f"see_cp_{label}"
        run_density_inference_mode(
            mode="run",
            backend="local",
            ckpt_dir=ckpt_dir,
            case_prefix=case_prefix,
            n_photons=size,
            timestamp=timestamp,
            model_factory=lambda processor=None: CP_PINN(
                size=size,
                hidden_width=width,
                num_hidden_layers=layers,
            ),
            save_plot_fn=save_density_plot,
        )

    else:
        raise ValueError("mode must be 'train', 'run', or 'remote'")
