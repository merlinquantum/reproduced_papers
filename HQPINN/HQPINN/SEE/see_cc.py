# see_cc.py
# Classical–Classical PINN

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
    DTYPE,
)
from ..utils import count_trainable_params, get_latest_checkpoint, load_model, make_optimizer
from .core_see import (
    evaluate_see_errors,
    load_training_loss_for_checkpoint,
    save_density_plot,
    train_see,
)
from ..run_common import run_density_inference_mode
from ..layer_classical import BranchPyTorch


class CC_PINN(nn.Module):
    """
    Classical-classical PINN with two parallel branches (cc-N-L).
    """

    def __init__(
        self,
        hidden_width: int = SEE_CC_HIDDEN_WIDTH,
        num_hidden_layers: int = SEE_CC_NUM_HIDDEN_LAYERS,
    ) -> None:
        super().__init__()

        # Two parallel classical branches: each (x,t) -> (rho,u,p)
        self.branch1 = BranchPyTorch(
            in_features=2,
            out_features=3,
            num_hidden_layers=num_hidden_layers,
            hidden_width=hidden_width,
        )
        self.branch2 = BranchPyTorch(
            in_features=2,
            out_features=3,
            num_hidden_layers=num_hidden_layers,
            hidden_width=hidden_width,
        )

        self.fusion = nn.Linear(6, 3, dtype=DTYPE)

        # Human-readable size label, e.g. "10-4"
        self.size_label = f"{hidden_width}-{num_hidden_layers}"

    def forward(self, xt: torch.Tensor) -> torch.Tensor:
        # Learned linear fusion of both branch outputs.
        out1 = self.branch1(xt)
        out2 = self.branch2(xt)
        return self.fusion(torch.cat([out1, out2], dim=1))


MODELS = [
    ("10-4", 10, 4),
    ("10-7", 10, 7),
    ("20-4", 20, 4),
]


def _get_model_config(model_size: str) -> tuple[str, int, int]:
    for label, width, layers in MODELS:
        if label == model_size:
            return label, width, layers
    valid = ", ".join(label for label, *_ in MODELS)
    raise ValueError(f"Unknown model_size='{model_size}'. Valid values: {valid}")


def run(mode="train", backend="sim:ascella", model_size="10-4"):
    """Run all SEE classical–classical models and write summary CSV."""
    torch.manual_seed(0)

    ckpt_dir = "HQPINN/SEE/"
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if mode == "train":
        out_csv = f"HQPINN/SEE/results/cc_summary_{timestamp}.csv"
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

            for label, width, layers in MODELS:
                print(
                    f"\nTraining SEE-CC model: {label} (width={width}, layers={layers})"
                )

                case_prefix = f"see_cc_{label}"
                model_dir = os.path.join(ckpt_dir, "models")
                existing_ckpt = get_latest_checkpoint(model_dir, case_prefix)
                if existing_ckpt is not None:
                    final_loss = load_training_loss_for_checkpoint(
                        out_dir=f"HQPINN/SEE/results/{case_prefix}",
                        model_label=f"cc_{label}",
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
                                lambda processor=None: CC_PINN(
                                    hidden_width=width, num_hidden_layers=layers
                                ),
                            )
                            err_rho, err_p = evaluate_see_errors(model)
                        except Exception as exc:
                            print(
                                f"Checkpoint validation failed for {case_prefix} at "
                                f"{existing_ckpt}: {exc}; retraining model."
                            )
                        else:
                            n_params = count_trainable_params(model)
                            writer.writerow(
                                [
                                    "cc",
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

                model = CC_PINN(hidden_width=width, num_hidden_layers=layers)
                optimizer = make_optimizer(model, lr=5e-4)

                final_loss, err_rho, err_p, n_params = train_see(
                    model=model,
                    t_train=None,  # kept for API consistency
                    optimizer=optimizer,
                    n_epochs=SEE_N_EPOCHS,
                    plot_every=SEE_PLOT_EVERY,
                    out_dir=f"HQPINN/SEE/results/{case_prefix}",
                    model_label=f"cc_{label}",
                    timestamp=timestamp,
                )

                writer.writerow(
                    [
                        "cc",
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
        label, width, layers = _get_model_config(model_size)
        case_prefix = f"see_cc_{label}"
        run_density_inference_mode(
            mode="run",
            backend="local",
            ckpt_dir=ckpt_dir,
            case_prefix=case_prefix,
            n_photons=0,
            timestamp=timestamp,
            model_factory=lambda processor=None: CC_PINN(
                hidden_width=width, num_hidden_layers=layers
            ),
            save_plot_fn=save_density_plot,
        )

    elif mode == "remote":
        print(
            "Remote mode is not available for SEE-CC. Falling back to local run mode."
        )
        label, width, layers = _get_model_config(model_size)
        case_prefix = f"see_cc_{label}"
        run_density_inference_mode(
            mode="run",
            backend="local",
            ckpt_dir=ckpt_dir,
            case_prefix=case_prefix,
            n_photons=0,
            timestamp=timestamp,
            model_factory=lambda processor=None: CC_PINN(
                hidden_width=width, num_hidden_layers=layers
            ),
            save_plot_fn=save_density_plot,
        )

    else:
        raise ValueError("mode must be 'train', 'run', or 'remote'")
