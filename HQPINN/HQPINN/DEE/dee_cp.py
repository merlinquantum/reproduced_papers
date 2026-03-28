# dee_cp.py
# Classical–PennyLane PINN

import csv
import os
from datetime import datetime

import torch
import torch.nn as nn

from ..config import (
    DEE_CC_NUM_HIDDEN_LAYERS,
    DEE_CC_HIDDEN_WIDTH,
    DEE_N_EPOCHS,
    DEE_PLOT_EVERY,
    N_LAYERS,
    DEE_LR,
    DTYPE,
)
from ..utils import count_trainable_params, get_latest_checkpoint, load_model, make_optimizer
from .core_dee import (
    evaluate_dee_errors,
    load_training_loss_for_checkpoint,
    save_density_plot,
    train_dee,
)
from ..run_common import run_density_inference_mode
from ..layer_pennylane import (
    make_quantum_block_multiout,
    dee_feature_map,
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
        hidden_width: int = DEE_CC_HIDDEN_WIDTH,
        num_hidden_layers: int = DEE_CC_NUM_HIDDEN_LAYERS,
    ) -> None:
        super().__init__()

        qblock_multi_1 = make_quantum_block_multiout(n_layers=size)

        self.branch1 = BranchPennylane(
            qblock_multi_1,
            feature_map=dee_feature_map,
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
    """Run all DEE Classical–PennyLane models and write summary CSV."""
    torch.manual_seed(0)

    ckpt_dir = "HQPINN/DEE/"
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if mode == "train":
        out_csv = f"HQPINN/DEE/results/cp_summary_{timestamp}.csv"
        os.makedirs("HQPINN/DEE/results", exist_ok=True)

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
                print(f"\nTraining DEE-CP model: {label} size={size}")

                case_prefix = f"dee_cp_{label}"
                model_dir = os.path.join(ckpt_dir, "models")
                existing_ckpt = get_latest_checkpoint(model_dir, case_prefix)
                if existing_ckpt is not None:
                    final_loss = load_training_loss_for_checkpoint(
                        out_dir=f"HQPINN/DEE/results/{case_prefix}",
                        model_label=f"cp_{label}",
                        ckpt_path=existing_ckpt,
                        case_prefix=case_prefix,
                    )
                    if final_loss is not None:
                        print(
                            f"Skipping {case_prefix}: existing checkpoint found at "
                            f"{existing_ckpt}"
                        )
                        try:
                            model = load_model(
                                existing_ckpt,
                                lambda processor=None: CP_PINN(
                                    size=size,
                                    hidden_width=width,
                                    num_hidden_layers=layers,
                                ),
                            )
                            err_rho, err_p = evaluate_dee_errors(model)
                        except Exception as exc:
                            print(
                                f"Checkpoint validation failed for {case_prefix} at "
                                f"{existing_ckpt}: {exc}; retraining model."
                            )
                        else:
                            n_params = count_trainable_params(model)
                            writer.writerow(
                                [
                                    "cp",
                                    label,
                                    n_params,
                                    f"{final_loss:.6e}",
                                    f"{err_rho:.6e}",
                                    f"{err_p:.6e}",
                                ]
                            )
                            print(
                                f"Reused latest metrics for {case_prefix} in summary CSV."
                            )
                            continue
                    print(
                        f"Existing checkpoint found for {case_prefix} at "
                        f"{existing_ckpt}, but no matching loss CSV was found; "
                        f"retraining model."
                    )

                model = CP_PINN(size=size, hidden_width=width, num_hidden_layers=layers)
                optimizer = make_optimizer(model, lr=DEE_LR)

                final_loss, err_rho, err_p, n_params = train_dee(
                    model=model,
                    t_train=None,  # kept for API consistency
                    optimizer=optimizer,
                    n_epochs=DEE_N_EPOCHS,
                    plot_every=DEE_PLOT_EVERY,
                    out_dir=f"HQPINN/DEE/results/{case_prefix}",
                    model_label=f"cp_{label}",
                    timestamp=timestamp,
                )

                writer.writerow(
                    [
                        "cp",  # model type: Classical–PennyLane
                        label,
                        n_params,
                        f"{final_loss:.6e}",
                        f"{err_rho:.6e}",
                        f"{err_p:.6e}",
                    ]
                )

                os.makedirs(model_dir, exist_ok=True)
                ckpt_path = os.path.join(model_dir, f"{case_prefix}_{timestamp}.pt")
                torch.save(model.state_dict(), ckpt_path)
                print(f"Model saved to: {ckpt_path}")

        print(f"Summary CSV saved to: {out_csv}")

    elif mode == "run":
        label, width, layers, size = _get_model_config(model_size)
        case_prefix = f"dee_cp_{label}"
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
        print("Remote mode is not available for DEE-CP. Falling back to local run mode.")
        label, width, layers, size = _get_model_config(model_size)
        case_prefix = f"dee_cp_{label}"
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
