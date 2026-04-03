# see_qq_pl.py
# PennyLane–PennyLane PINN

import os
from datetime import datetime

import torch
import torch.nn as nn

from ...config import SEE_N_EPOCHS, SEE_PLOT_EVERY, N_LAYERS, SEE_LR, DTYPE
from ...utils import (
    count_trainable_params,
    get_latest_checkpoint,
    load_model,
    make_optimizer,
)
from ...runtime import seed_everything
from .core_see import (
    append_summary_row,
    evaluate_see_errors,
    get_run_id_from_checkpoint,
    load_training_loss_for_checkpoint,
    load_training_row_for_run_id,
    save_density_plot,
    train_see,
)
from ...run_common import run_density_inference_mode
from ..layer_pennylane import (
    make_quantum_block_multiout,
    see_feature_map,
    BranchPennylane,
)


class PP_PINN(nn.Module):
    """
    PennyLane–PennyLane PINN with two parallel quantum branches.

    Each quantum branch maps (x, t) -> R^3 via a multi-output PQC and
    a SEE-specific feature map.
    """

    def __init__(self, q_layers: int = N_LAYERS) -> None:
        super().__init__()

        qblock_multi_1 = make_quantum_block_multiout(n_layers=q_layers)
        qblock_multi_2 = make_quantum_block_multiout(n_layers=q_layers)

        # Two parallel PennyLane branches: each (x,t) -> (rho_like, u_like, p_like)
        self.branch1 = BranchPennylane(
            qblock_multi_1,
            feature_map=see_feature_map,
            output_as_column=False,
            n_layers=q_layers,
        )
        self.branch2 = BranchPennylane(
            qblock_multi_2,
            feature_map=see_feature_map,
            output_as_column=False,
            n_layers=q_layers,
        )
        self.fusion = nn.Linear(6, 3, dtype=DTYPE)

        # Human-readable quantum-depth label (e.g. "2", "3", "4")
        self.size_label = f"{q_layers}"

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
    for label, q_layers in MODELS:
        if label == model_size:
            return label, q_layers
    valid = ", ".join(label for label, _ in MODELS)
    raise ValueError(f"Unknown model_size='{model_size}'. Valid values: {valid}")


def _resolve_model_config(
    *,
    model_size: str | None = None,
    q_layers: int | None = None,
) -> tuple[str, int]:
    if model_size is not None:
        return _get_model_config(model_size)
    if q_layers is None:
        raise ValueError("SEE-QQ-PL requires either model_size or q_layers")
    return str(q_layers), q_layers


def run(
    mode="train",
    backend="sim:ascella",
    model_size="2",
    *,
    q_layers: int | None = None,
):
    """Run all SEE PennyLane–PennyLane models and write summary CSV."""
    seed_everything(0)

    ckpt_dir = "HQPINN/lib/SEE/"
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    if mode == "train":
        summary_csv = "HQPINN/lib/SEE/results/see_summary.csv"
        if q_layers is not None:
            models = [_resolve_model_config(q_layers=q_layers)]
        else:
            models = MODELS
        for label, q_layers in models:
            seed_everything(0)
            print(f"\nTraining SEE-QQ-PL model: {label} q_layers={q_layers}")

            case_prefix = f"see_qq_pl_{label}"
            model_dir = os.path.join(ckpt_dir, "models")
            existing_ckpt = get_latest_checkpoint(model_dir, case_prefix)
            if existing_ckpt is not None:
                final_loss = load_training_loss_for_checkpoint(
                    out_dir=f"HQPINN/lib/SEE/results/{case_prefix}",
                    model_label=f"qq-pl_{label}",
                    ckpt_path=existing_ckpt,
                    case_prefix=case_prefix,
                )
                if final_loss is not None:
                    try:
                        model = load_model(
                            existing_ckpt,
                            lambda processor=None: PP_PINN(q_layers=q_layers),
                        )
                        err_rho, err_p = evaluate_see_errors(model)
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
                                out_dir=f"HQPINN/lib/SEE/results/{case_prefix}",
                                model_label=f"qq-pl_{label}",
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
                                "Model": "qq-pl",
                                "Size": label,
                                "run_id": case_run_id or "",
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

            model = PP_PINN(q_layers=q_layers)
            optimizer = make_optimizer(model, lr=SEE_LR)

            final_loss, err_rho, err_p, n_params = train_see(
                model=model,
                t_train=None,  # kept for API consistency
                optimizer=optimizer,
                n_epochs=SEE_N_EPOCHS,
                plot_every=SEE_PLOT_EVERY,
                out_dir=f"HQPINN/lib/SEE/results/{case_prefix}",
                model_label=f"qq-pl_{label}",
                run_id=run_id,
            )
            row = load_training_row_for_run_id(
                out_dir=f"HQPINN/lib/SEE/results/{case_prefix}",
                model_label=f"qq-pl_{label}",
                run_id=run_id,
            )

            is_duplicate = append_summary_row(
                summary_csv,
                {
                    "Model": "qq-pl",
                    "Size": label,
                    "run_id": run_id,
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
        label, q_layers = _resolve_model_config(
            model_size=model_size if q_layers is None else None,
            q_layers=q_layers,
        )
        case_prefix = f"see_qq_pl_{label}"
        run_density_inference_mode(
            mode="run",
            backend="local",
            ckpt_dir=ckpt_dir,
            case_prefix=case_prefix,
            plot_label=f"q_layers={q_layers}",
            run_id=run_id,
            model_factory=lambda processor=None: PP_PINN(q_layers=q_layers),
            save_plot_fn=save_density_plot,
        )

    elif mode == "remote":
        print(
            "Remote mode is not available for SEE-QQ-PL. Falling back to local run mode."
        )
        label, q_layers = _resolve_model_config(
            model_size=model_size if q_layers is None else None,
            q_layers=q_layers,
        )
        case_prefix = f"see_qq_pl_{label}"
        run_density_inference_mode(
            mode="run",
            backend="local",
            ckpt_dir=ckpt_dir,
            case_prefix=case_prefix,
            plot_label=f"q_layers={q_layers}",
            run_id=run_id,
            model_factory=lambda processor=None: PP_PINN(q_layers=q_layers),
            save_plot_fn=save_density_plot,
        )

    else:
        raise ValueError("mode must be 'train', 'run', or 'remote'")
