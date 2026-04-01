# dee_pp.py
# PennyLane–PennyLane PINN

import os
import csv
from datetime import datetime

import torch
import torch.nn as nn

from ..config import DEE_N_EPOCHS, DEE_PLOT_EVERY, N_LAYERS, DEE_LR, DTYPE
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


class PP_PINN(nn.Module):
    """
    PennyLane–PennyLane PINN with two parallel quantum branches.

    Each quantum branch maps (x, t) -> R^3 via a multi-output PQC and
    a DEE-compatible feature map.
    """

    def __init__(self, n_layers: int = N_LAYERS) -> None:
        super().__init__()

        qblock_multi_1 = make_quantum_block_multiout(n_layers=n_layers)
        qblock_multi_2 = make_quantum_block_multiout(n_layers=n_layers)

        self.branch1 = BranchPennylane(
            qblock_multi_1,
            feature_map=dee_feature_map,
            output_as_column=False,
            n_layers=n_layers,
        )
        self.branch2 = BranchPennylane(
            qblock_multi_2,
            feature_map=dee_feature_map,
            output_as_column=False,
            n_layers=n_layers,
        )
        self.fusion = nn.Linear(6, 3, dtype=DTYPE)

        # Human-readable size label (e.g. "2", "3", "4")
        self.size_label = f"{n_layers}"

    def forward(self, xt: torch.Tensor) -> torch.Tensor:
        # Learned linear fusion of both branch outputs.
        out1 = self.branch1(xt)  # [N, 3]
        out2 = self.branch2(xt)  # [N, 3]
        return self.fusion(torch.cat([out1, out2], dim=1))  # [N, 3]


MODELS = [
    ("2", 2),
    ("3", 3),
    ("4", 4),
]


def _get_model_config(model_size: str) -> tuple[str, int]:
    for label, size in MODELS:
        if label == model_size:
            return label, size
    valid = ", ".join(label for label, _ in MODELS)
    raise ValueError(f"Unknown model_size='{model_size}'. Valid values: {valid}")


def run(mode="train", backend="sim:ascella", model_size="2"):
    """Run all DEE PennyLane–PennyLane models and write summary CSV."""
    torch.manual_seed(0)

    ckpt_dir = "HQPINN/DEE/"
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if mode == "train":
        out_csv = f"HQPINN/DEE/results/pp_summary_{timestamp}.csv"
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

            for label, size in MODELS:
                print(f"\nTraining DEE-PP model: {label} size={size}")

                case_prefix = f"dee_pp_{label}"
                model_dir = os.path.join(ckpt_dir, "models")
                existing_ckpt = get_latest_checkpoint(model_dir, case_prefix)
                if existing_ckpt is not None:
                    final_loss = load_training_loss_for_checkpoint(
                        out_dir=f"HQPINN/DEE/results/{case_prefix}",
                        model_label=f"pp_{label}",
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
                                lambda processor=None: PP_PINN(n_layers=size),
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
                                    "pp",
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

                model = PP_PINN(n_layers=size)
                optimizer = make_optimizer(model, lr=DEE_LR)

                final_loss, err_rho, err_p, n_params = train_dee(
                    model=model,
                    t_train=None,  # kept for API consistency
                    optimizer=optimizer,
                    n_epochs=DEE_N_EPOCHS,
                    plot_every=DEE_PLOT_EVERY,
                    out_dir=f"HQPINN/DEE/results/{case_prefix}",
                    model_label=f"pp_{label}",
                    timestamp=timestamp,
                )

                writer.writerow(
                    [
                        "pp",  # model type: PennyLane–PennyLane
                        label,  # size label ("2", "3", "4")
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
        label, size = _get_model_config(model_size)
        case_prefix = f"dee_pp_{label}"
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
        print("Remote mode is not available for DEE-PP. Falling back to local run mode.")
        label, size = _get_model_config(model_size)
        case_prefix = f"dee_pp_{label}"
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
