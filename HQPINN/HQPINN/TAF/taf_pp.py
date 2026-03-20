# taf_pp.py
# PennyLane–PennyLane PINN for TAF (Sec. 3.3)

import csv
import os
from datetime import datetime

import torch
import torch.nn as nn

from ..config import (
    DEVICE,
    DTYPE,
    N_LAYERS,
    TAF_ADAM_STEPS,
    TAF_EPSILON_LAMBDA,
    TAF_LBFGS_STEPS,
    TAF_LR,
    TAF_N_OUTPUTS,
    TAF_PLOT_EVERY,
)
from ..layer_pennylane import (
    BranchPennylane,
    make_quantum_block_multiout,
    taf_feature_map,
)
from ..run_common import run_density_inference_mode
from ..utils import make_optimizer
from .core_taf import (
    load_training_sets,
    save_density_plot,
    train_taf,
)


class PP_PINN(nn.Module):
    """PennyLane-PennyLane TAF PINN with two independent quantum branches."""

    def __init__(self, n_layers: int = N_LAYERS) -> None:
        super().__init__()

        qblock_multi_1 = make_quantum_block_multiout(
            n_layers=n_layers, n_qubits=TAF_N_OUTPUTS
        )
        qblock_multi_2 = make_quantum_block_multiout(
            n_layers=n_layers, n_qubits=TAF_N_OUTPUTS
        )

        self.branch1 = BranchPennylane(
            qblock_multi_1,
            feature_map=taf_feature_map,
            output_as_column=False,
            n_layers=n_layers,
            n_qubits=TAF_N_OUTPUTS,
        )
        self.branch2 = BranchPennylane(
            qblock_multi_2,
            feature_map=taf_feature_map,
            output_as_column=False,
            n_layers=n_layers,
            n_qubits=TAF_N_OUTPUTS,
        )

        # Two branches of TAF_N_OUTPUTS quantum outputs each.
        self.fusion = nn.Linear(2 * TAF_N_OUTPUTS, TAF_N_OUTPUTS, dtype=DTYPE)
        self.size_label = f"{n_layers}"

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        out1 = self.branch1(xy)  # [N, TAF_N_OUTPUTS]
        out2 = self.branch2(xy)  # [N, TAF_N_OUTPUTS]
        return self.fusion(torch.cat([out1, out2], dim=1))  # [N, TAF_N_OUTPUTS]


MODELS = [
    ("2", 2),
    ("4", 4),
    ("6", 6),
]


def _get_model_config(model_size: str) -> tuple[str, int]:
    for label, size in MODELS:
        if label == model_size:
            return label, size
    valid = ", ".join(label for label, _ in MODELS)
    raise ValueError(f"Unknown model_size='{model_size}'. Valid values: {valid}")


def run(mode="train", backend="sim:ascella", model_size="2") -> None:
    """Run TAF PennyLane-PennyLane models and write summary CSV."""
    torch.manual_seed(0)

    data = load_training_sets()

    # Sec. 3.3 inlet values (SI)
    U_in = torch.tensor([1.225, 272.15, 0.0, 288.15], dtype=DTYPE, device=DEVICE)

    ckpt_dir = "HQPINN/TAF/"
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if mode == "train":
        os.makedirs("HQPINN/TAF/results", exist_ok=True)
        out_csv = f"HQPINN/TAF/results/pp_summary_{timestamp}.csv"

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

            for label, size in MODELS:
                print(f"\nTraining TAF-PP model: {label} size={size}")

                case_prefix = f"taf_pp_{label}"
                model = PP_PINN(n_layers=size).to(DEVICE)
                optimizer = make_optimizer(model, lr=TAF_LR)

                final_loss, loss_bc, loss_f, n_params = train_taf(
                    model=model,
                    optimizer=optimizer,
                    n_epochs=TAF_ADAM_STEPS,
                    plot_every=TAF_PLOT_EVERY,
                    out_dir=f"HQPINN/TAF/results/{case_prefix}",
                    model_label=f"pp_{label}",
                    data=data,
                    U_in=U_in,
                    lbfgs_steps=TAF_LBFGS_STEPS,
                    eps_lambda=TAF_EPSILON_LAMBDA,
                )

                writer.writerow(
                    [
                        "pp",
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
        label, size = _get_model_config(model_size)
        case_prefix = f"taf_pp_{label}"
        run_density_inference_mode(
            mode="run",
            backend="local",
            ckpt_dir=ckpt_dir,
            case_prefix=case_prefix,
            n_photons=size,
            timestamp=timestamp,
            model_factory=lambda processor=None: PP_PINN(n_layers=size),
            save_plot_fn=save_density_plot,
        )

    elif mode == "remote":
        print("Remote mode is not available for TAF-PP. Falling back to local run mode.")
        label, size = _get_model_config(model_size)
        case_prefix = f"taf_pp_{label}"
        run_density_inference_mode(
            mode="run",
            backend="local",
            ckpt_dir=ckpt_dir,
            case_prefix=case_prefix,
            n_photons=size,
            timestamp=timestamp,
            model_factory=lambda processor=None: PP_PINN(n_layers=size),
            save_plot_fn=save_density_plot,
        )

    else:
        raise ValueError("mode must be 'train', 'run', or 'remote'")
