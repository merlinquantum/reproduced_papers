# dee_hy_pl.py
# Classical–PennyLane PINN

import os
from datetime import datetime

import torch
import torch.nn as nn

from ...config import (
    DEE_CC_NUM_HIDDEN_LAYERS,
    DEE_CC_HIDDEN_WIDTH,
    DEE_N_EPOCHS,
    DEE_PLOT_EVERY,
    N_LAYERS,
    DEE_LR,
    DTYPE,
)
from ...utils import (
    count_trainable_params,
    get_latest_checkpoint,
    load_model,
    make_optimizer,
)
from ...runtime import seed_everything
from .core_dee import (
    append_summary_row,
    evaluate_dee_errors,
    get_run_id_from_checkpoint,
    load_training_loss_for_checkpoint,
    load_training_row_for_run_id,
    save_density_plot,
    train_dee,
)
from ...run_common import run_density_inference_mode
from ...layer_pennylane import (
    make_quantum_block_multiout,
    dee_feature_map,
    BranchPennylane,
)
from ...layer_classical import BranchPyTorch


class CP_PINN(nn.Module):
    """
    Classical–PennyLane PINN with one quantum branch and one classical MLP branch.
    """

    def __init__(
        self,
        q_layers: int = N_LAYERS,
        hidden_width: int = DEE_CC_HIDDEN_WIDTH,
        num_hidden_layers: int = DEE_CC_NUM_HIDDEN_LAYERS,
        *,
        size: int | None = None,
    ) -> None:
        super().__init__()
        if size is not None:
            # Backward-compatible alias while the rest of the repo catches up.
            q_layers = size

        qblock_multi_1 = make_quantum_block_multiout(n_layers=q_layers)

        self.branch1 = BranchPennylane(
            qblock_multi_1,
            feature_map=dee_feature_map,
            output_as_column=False,
            n_layers=q_layers,
        )
        self.branch2 = BranchPyTorch(
            in_features=2,
            out_features=3,
            num_hidden_layers=num_hidden_layers,
            hidden_width=hidden_width,
        )
        self.fusion = nn.Linear(6, 3, dtype=DTYPE)

        self.size_label = f"{q_layers}"

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
    for label, width, layers, q_layers in MODELS:
        if label == model_size:
            return label, width, layers, q_layers
    valid = ", ".join(label for label, *_ in MODELS)
    raise ValueError(f"Unknown model_size='{model_size}'. Valid values: {valid}")


def run(mode="train", backend="sim:ascella", model_size="10-4-2"):
    """Run all DEE Classical–PennyLane models and write summary CSV."""
    seed_everything(0)

    ckpt_dir = "HQPINN/lib/DEE/"
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    if mode == "train":
        summary_csv = "HQPINN/lib/DEE/results/dee_summary.csv"

        for label, width, layers, q_layers in MODELS:
            seed_everything(0)
            print(f"\nTraining DEE-HY-PL model: {label} q_layers={q_layers}")

            case_prefix = f"dee_hy_pl_{label}"
            model_dir = os.path.join(ckpt_dir, "models")
            existing_ckpt = get_latest_checkpoint(model_dir, case_prefix)
            if existing_ckpt is not None:
                final_loss = load_training_loss_for_checkpoint(
                    out_dir=f"HQPINN/lib/DEE/results/{case_prefix}",
                    model_label=f"hy-pl_{label}",
                    ckpt_path=existing_ckpt,
                    case_prefix=case_prefix,
                )
                if final_loss is not None:
                    try:
                        model = load_model(
                            existing_ckpt,
                            lambda processor=None: CP_PINN(
                                q_layers=q_layers,
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
                        case_run_id = get_run_id_from_checkpoint(
                            existing_ckpt, case_prefix
                        )
                        row = (
                            load_training_row_for_run_id(
                                out_dir=f"HQPINN/lib/DEE/results/{case_prefix}",
                                model_label=f"hy-pl_{label}",
                                run_id=case_run_id,
                            )
                            if case_run_id is not None
                            else None
                        )
                        print(
                            f"Skipping training for {case_prefix}: existing checkpoint found at {existing_ckpt}."
                        )
                        is_duplicate = append_summary_row(
                            summary_csv,
                            {
                                "run_id": case_run_id or "",
                                "Model": "hy-pl",
                                "Size": label,
                                "epoch": row["epoch"] if row is not None else "",
                                "elapsed (s)": row["elapsed (s)"]
                                if row is not None
                                else "",
                                "Trainable parameters": n_params,
                                "Loss": row["Loss"]
                                if row is not None
                                else f"{final_loss:.6e}",
                                "IC": row["IC"] if row is not None else "",
                                "BC": row["BC"] if row is not None else "",
                                "F": row["F"] if row is not None else "",
                                "Density error": f"{err_rho:.6e}",
                                "Pressure error": f"{err_p:.6e}",
                            },
                        )
                        if is_duplicate:
                            print(
                                f"Duplicate summary row appended for run_id={case_run_id} to: {summary_csv}"
                            )
                        else:
                            print(f"Summary CSV appended to: {summary_csv}")
                        print(f"Reused checkpoint metrics for {case_prefix}.")
                        print()
                        continue
                print(
                    f"Existing checkpoint found for {case_prefix} at "
                    f"{existing_ckpt}, but no matching training CSV was found; "
                    f"retraining model."
                )

            model = CP_PINN(
                q_layers=q_layers,
                hidden_width=width,
                num_hidden_layers=layers,
            )
            optimizer = make_optimizer(model, lr=DEE_LR)

            final_loss, err_rho, err_p, n_params = train_dee(
                model=model,
                t_train=None,  # kept for API consistency
                optimizer=optimizer,
                n_epochs=DEE_N_EPOCHS,
                plot_every=DEE_PLOT_EVERY,
                out_dir=f"HQPINN/lib/DEE/results/{case_prefix}",
                model_label=f"hy-pl_{label}",
                run_id=run_id,
            )
            row = load_training_row_for_run_id(
                out_dir=f"HQPINN/lib/DEE/results/{case_prefix}",
                model_label=f"hy-pl_{label}",
                run_id=run_id,
            )

            is_duplicate = append_summary_row(
                summary_csv,
                {
                    "run_id": run_id,
                    "Model": "hy-pl",
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
            if is_duplicate:
                print(
                    f"Duplicate summary row appended for run_id={run_id} to: {summary_csv}"
                )
            else:
                print(f"Summary CSV appended to: {summary_csv}")
            print()

    elif mode == "run":
        label, width, layers, q_layers = _get_model_config(model_size)
        case_prefix = f"dee_hy_pl_{label}"
        run_density_inference_mode(
            mode="run",
            backend="local",
            ckpt_dir=ckpt_dir,
            case_prefix=case_prefix,
            plot_label=f"q_layers={q_layers}",
            run_id=run_id,
            model_factory=lambda processor=None: CP_PINN(
                q_layers=q_layers,
                hidden_width=width,
                num_hidden_layers=layers,
            ),
            save_plot_fn=save_density_plot,
        )

    elif mode == "remote":
        print(
            "Remote mode is not available for DEE-HY-PL. Falling back to local run mode."
        )
        label, width, layers, q_layers = _get_model_config(model_size)
        case_prefix = f"dee_hy_pl_{label}"
        run_density_inference_mode(
            mode="run",
            backend="local",
            ckpt_dir=ckpt_dir,
            case_prefix=case_prefix,
            plot_label=f"q_layers={q_layers}",
            run_id=run_id,
            model_factory=lambda processor=None: CP_PINN(
                q_layers=q_layers,
                hidden_width=width,
                num_hidden_layers=layers,
            ),
            save_plot_fn=save_density_plot,
        )

    else:
        raise ValueError("mode must be 'train', 'run', or 'remote'")
