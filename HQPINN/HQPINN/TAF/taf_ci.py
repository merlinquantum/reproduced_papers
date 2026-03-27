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
    TAF_N_OUTPUTS,
    TAF_PLOT_EVERY,
)
from ..layer_classical import BranchPyTorch
from ..layer_merlin import BranchMerlin, make_interf_qlayer
from ..run_common import run_density_inference_mode
from ..utils import count_trainable_params, get_latest_checkpoint, make_optimizer
from .core_taf import (
    load_training_sets,
    load_latest_training_metrics,
    save_density_plot,
    train_taf,
)


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
            out_features=TAF_N_OUTPUTS,
            num_hidden_layers=num_hidden_layers,
            hidden_width=hidden_width,
        )
        self.branch2 = BranchMerlin(
            make_interf_qlayer(n_photons=n_photons),
            n_outputs=TAF_N_OUTPUTS,
            processor=processor,
            feature_map_kind="taf",
        )
        self.fusion = nn.Linear(2 * TAF_N_OUTPUTS, TAF_N_OUTPUTS, dtype=DTYPE)

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        # Learned linear fusion of both branch outputs.
        out1 = self.branch1(xy)
        out2 = self.branch2(xy)
        return self.fusion(torch.cat([out1, out2], dim=1))


MODELS = [
    ("40-4-2", 40, 4, 2),
    ("40-7-2", 40, 7, 2),
    ("80-4-2", 80, 4, 2),
]


def _get_model_config(model_size: str) -> tuple[str, int, int, int]:
    for label, width, layers, n_photons in MODELS:
        if label == model_size:
            return label, width, layers, n_photons
    valid = ", ".join(label for label, *_ in MODELS)
    raise ValueError(f"Unknown model_size='{model_size}'. Valid values: {valid}")


def run(mode="train", backend="sim:ascella", model_size="40-4-2") -> None:
    """Run TAF classical-interferometer models and write summary CSV."""
    torch.manual_seed(0)

    data = load_training_sets()

    # Sec. 3.3 inlet values (SI)
    U_in = torch.tensor([1.225, 272.15, 0.0, 288.15], dtype=DTYPE, device=DEVICE)

    ckpt_dir = "HQPINN/TAF/"
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if mode == "train":
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
                model_dir = os.path.join(ckpt_dir, "models")
                existing_ckpt = get_latest_checkpoint(model_dir, case_prefix)
                if existing_ckpt is not None:
                    metrics = load_latest_training_metrics(
                        out_dir=f"HQPINN/TAF/results/{case_prefix}",
                        model_label=f"ci_{label}",
                    )
                    print(
                        f"Skipping {case_prefix}: existing checkpoint found at {existing_ckpt}"
                    )
                    if metrics is None:
                        print(
                            f"No existing metrics CSV found for {case_prefix}; summary row omitted."
                        )
                        continue

                    n_params = count_trainable_params(
                        CI_PINN(
                            n_photons=n_photons,
                            hidden_width=width,
                            num_hidden_layers=layers,
                        )
                    )
                    final_loss, loss_bc, loss_f = metrics
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
                    print(f"Reused latest metrics for {case_prefix} in summary CSV.")
                    continue

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

                os.makedirs(model_dir, exist_ok=True)
                ckpt_path = os.path.join(model_dir, f"{case_prefix}_{timestamp}.pt")
                torch.save(model.state_dict(), ckpt_path)
                print(f"Model saved to: {ckpt_path}")

        print(f"Summary CSV saved to: {out_csv}")

    elif mode == "run":
        label, width, layers, n_photons = _get_model_config(model_size)
        case_prefix = f"taf_ci_{label}"
        run_density_inference_mode(
            mode="run",
            backend="local",
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
        case_prefix = f"taf_ci_{label}"
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
