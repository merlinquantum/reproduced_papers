# dee_ci.py
# Classical–Interferometer PINN for DEE

import os
from datetime import datetime

import torch
import torch.nn as nn

from ..config import (
    DEE_CC_NUM_HIDDEN_LAYERS,
    DEE_CC_HIDDEN_WIDTH,
    DEE_N_EPOCHS,
    DEE_PLOT_EVERY,
    DEE_LR,
    DTYPE,
)
from ..utils import (
    count_trainable_params,
    get_latest_checkpoint,
    load_model,
    make_optimizer,
    set_global_seed,
)
from .core_dee import (
    append_summary_row,
    evaluate_dee_errors,
    get_run_id_from_checkpoint,
    load_training_loss_for_checkpoint,
    load_training_row_for_run_id,
    save_density_plot,
    train_dee,
)
from ..run_common import run_density_inference_mode
from ..layer_classical import BranchPyTorch
from ..layer_merlin import make_interf_qlayer, BranchMerlin


class CI_PINN(nn.Module):
    """
    Classical-Interferometer PINN with one classical branch and one quantum branch.
    """

    def __init__(
        self,
        n_photons: int,
        hidden_width: int = DEE_CC_HIDDEN_WIDTH,
        num_hidden_layers: int = DEE_CC_NUM_HIDDEN_LAYERS,
        processor=None,
    ) -> None:
        super().__init__()

        self.branch1 = BranchPyTorch(
            in_features=2,
            out_features=3,
            num_hidden_layers=num_hidden_layers,
            hidden_width=hidden_width,
        )
        self.branch2 = BranchMerlin(
            make_interf_qlayer(n_photons=n_photons),
            n_outputs=3,
            processor=processor,
            feature_map_kind="dee",
        )
        self.fusion = nn.Linear(6, 3, dtype=DTYPE)

        self.size_label = f"{hidden_width}-{num_hidden_layers}"

    def forward(self, xt: torch.Tensor) -> torch.Tensor:
        out1 = self.branch1(xt)
        out2 = self.branch2(xt)
        # Learned linear fusion of both branch outputs.
        return self.fusion(torch.cat([out1, out2], dim=1))


MODELS = [
    ("10-4-1", 10, 4, 1),
    ("10-7-1", 10, 7, 1),
    ("20-4-1", 20, 4, 1),
]


def _get_model_config(model_size: str) -> tuple[str, int, int, int]:
    for label, width, layers, n_photons in MODELS:
        if label == model_size:
            return label, width, layers, n_photons
    valid = ", ".join(label for label, *_ in MODELS)
    raise ValueError(f"Unknown model_size='{model_size}'. Valid values: {valid}")


def run(mode="train", backend="sim:ascella", model_size="10-4-1"):
    """Run DEE Classical-Interferometer models and write summary CSV."""
    set_global_seed(0)

    ckpt_dir = "HQPINN/DEE/"
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")

    if mode == "train":
        print("=== TRAINING MODE ===")
        summary_csv = "HQPINN/DEE/results/dee_summary.csv"
        for label, width, layers, n_photons in MODELS:
            set_global_seed(0)
            print(
                f"\nTraining DEE-CI model: {label} (width={width}, layers={layers}, {n_photons} photons)"
            )

            case_prefix = f"dee_ci_{label}"
            model_dir = os.path.join(ckpt_dir, "models")
            existing_ckpt = get_latest_checkpoint(model_dir, case_prefix)
            if existing_ckpt is not None:
                final_loss = load_training_loss_for_checkpoint(
                    out_dir=f"HQPINN/DEE/results/{case_prefix}",
                    model_label=f"ci_{label}",
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
                            lambda processor=None: CI_PINN(
                                n_photons=n_photons,
                                hidden_width=width,
                                num_hidden_layers=layers,
                                processor=processor,
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
                        case_run_id = get_run_id_from_checkpoint(existing_ckpt, case_prefix)
                        row = (
                            load_training_row_for_run_id(
                                out_dir=f"HQPINN/DEE/results/{case_prefix}",
                                model_label=f"ci_{label}",
                                run_id=case_run_id,
                            )
                            if case_run_id is not None
                            else None
                        )
                        append_summary_row(
                            summary_csv,
                            {
                                "run_id": case_run_id or "",
                                "Model": "ci",
                                "Size": label,
                                "epoch": row["epoch"] if row is not None else "",
                                "elapsed (s)": row["elapsed (s)"] if row is not None else "",
                                "Trainable parameters": n_params,
                                "Loss": row["Loss"] if row is not None else f"{final_loss:.6e}",
                                "IC": row["IC"] if row is not None else "",
                                "BC": row["BC"] if row is not None else "",
                                "F": row["F"] if row is not None else "",
                                "Density error": f"{err_rho:.6e}",
                                "Pressure error": f"{err_p:.6e}",
                            },
                        )
                        print(f"Reused latest metrics for {case_prefix} in summary CSV.")
                        continue
                print(
                    f"Existing checkpoint found for {case_prefix} at "
                    f"{existing_ckpt}, but no matching loss CSV was found; "
                    f"retraining model."
                )

            model = CI_PINN(
                n_photons=n_photons, hidden_width=width, num_hidden_layers=layers
            )
            optimizer = make_optimizer(model, lr=DEE_LR)

            final_loss, err_rho, err_p, n_params = train_dee(
                model=model,
                t_train=None,
                optimizer=optimizer,
                n_epochs=DEE_N_EPOCHS,
                plot_every=DEE_PLOT_EVERY,
                out_dir=f"HQPINN/DEE/results/{case_prefix}",
                model_label=f"ci_{label}",
                run_id=run_id,
            )
            row = load_training_row_for_run_id(
                out_dir=f"HQPINN/DEE/results/{case_prefix}",
                model_label=f"ci_{label}",
                run_id=run_id,
            )

            append_summary_row(
                summary_csv,
                {
                    "run_id": run_id,
                    "Model": "ci",
                    "Size": label,
                    "epoch": row["epoch"] if row is not None else "",
                    "elapsed (s)": row["elapsed (s)"] if row is not None else "",
                    "Trainable parameters": n_params,
                    "Loss": row["Loss"] if row is not None else f"{final_loss:.6e}",
                    "IC": row["IC"] if row is not None else "",
                    "BC": row["BC"] if row is not None else "",
                    "F": row["F"] if row is not None else "",
                    "Density error": f"{err_rho:.6e}",
                    "Pressure error": f"{err_p:.6e}",
                },
            )

            os.makedirs(model_dir, exist_ok=True)
            ckpt_path = os.path.join(model_dir, f"{case_prefix}_{run_id}.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Model saved to: {ckpt_path}")

        print(f"Summary CSV appended to: {summary_csv}")

    elif mode == "run":
        label, width, layers, n_photons = _get_model_config(model_size)
        case_prefix = f"dee_ci_{label}"
        run_density_inference_mode(
            mode="run",
            backend=backend,
            ckpt_dir=ckpt_dir,
            case_prefix=case_prefix,
            plot_label=f"{n_photons} photons",
            run_id=run_id,
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
        case_prefix = f"dee_ci_{label}"
        run_density_inference_mode(
            mode="remote",
            backend=backend,
            ckpt_dir=ckpt_dir,
            case_prefix=case_prefix,
            plot_label=f"{n_photons} photons",
            run_id=run_id,
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
