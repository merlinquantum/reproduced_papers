# dee_cc.py
# Classical–Classical PINN

import os
from datetime import datetime

import torch
import torch.nn as nn

from ...config import (
    DEE_CC_NUM_HIDDEN_LAYERS,
    DEE_CC_HIDDEN_WIDTH,
    DEE_N_EPOCHS,
    DEE_PLOT_EVERY,
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
from ..layer_classical import BranchPyTorch


class CC_PINN(nn.Module):
    """
    Classical-classical PINN with two parallel branches (cc-N-L).
    """

    def __init__(
        self,
        hidden_width: int = DEE_CC_HIDDEN_WIDTH,
        num_hidden_layers: int = DEE_CC_NUM_HIDDEN_LAYERS,
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


def _resolve_model_config(
    *,
    model_size: str | None = None,
    n_nodes: int | None = None,
    n_layers: int | None = None,
) -> tuple[str, int, int]:
    if model_size is not None:
        return _get_model_config(model_size)
    if n_nodes is None or n_layers is None:
        raise ValueError(
            "DEE-CC requires either model_size or both n_nodes and n_layers"
        )
    return f"{n_nodes}-{n_layers}", n_nodes, n_layers


def run(
    mode="train",
    backend="sim:ascella",
    model_size="10-4",
    *,
    n_nodes: int | None = None,
    n_layers: int | None = None,
):
    """Run all DEE classical–classical models and write summary CSV."""
    seed_everything(0)

    ckpt_dir = "HQPINN/lib/DEE/"
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    if mode == "train":
        summary_csv = "HQPINN/lib/DEE/results/dee_summary.csv"
        if n_nodes is not None or n_layers is not None:
            models = [_resolve_model_config(n_nodes=n_nodes, n_layers=n_layers)]
        else:
            models = MODELS

        for label, width, layers in models:
            seed_everything(0)
            print(f"\nTraining DEE-CC model: {label} (width={width}, layers={layers})")

            case_prefix = f"dee_cc_{label}"
            model_dir = os.path.join(ckpt_dir, "models")
            existing_ckpt = get_latest_checkpoint(model_dir, case_prefix)
            if existing_ckpt is not None:
                final_loss = load_training_loss_for_checkpoint(
                    out_dir=f"HQPINN/lib/DEE/results/{case_prefix}",
                    model_label=f"cc_{label}",
                    ckpt_path=existing_ckpt,
                    case_prefix=case_prefix,
                )
                if final_loss is not None:
                    try:
                        model = load_model(
                            existing_ckpt,
                            lambda processor=None: CC_PINN(
                                hidden_width=width, num_hidden_layers=layers
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
                                model_label=f"cc_{label}",
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
                                "Model": "cc",
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

            model = CC_PINN(hidden_width=width, num_hidden_layers=layers)
            optimizer = make_optimizer(model, lr=5e-4)

            final_loss, err_rho, err_p, n_params = train_dee(
                model=model,
                t_train=None,  # kept for API consistency
                optimizer=optimizer,
                n_epochs=DEE_N_EPOCHS,
                plot_every=DEE_PLOT_EVERY,
                out_dir=f"HQPINN/lib/DEE/results/{case_prefix}",
                model_label=f"cc_{label}",
                run_id=run_id,
            )
            row = load_training_row_for_run_id(
                out_dir=f"HQPINN/lib/DEE/results/{case_prefix}",
                model_label=f"cc_{label}",
                run_id=run_id,
            )

            is_duplicate = append_summary_row(
                summary_csv,
                {
                    "run_id": run_id,
                    "Model": "cc",
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
        label, width, layers = _resolve_model_config(
            model_size=model_size,
            n_nodes=n_nodes,
            n_layers=n_layers,
        )
        case_prefix = f"dee_cc_{label}"
        run_density_inference_mode(
            mode="run",
            backend="local",
            ckpt_dir=ckpt_dir,
            case_prefix=case_prefix,
            plot_label=None,
            run_id=run_id,
            model_factory=lambda processor=None: CC_PINN(
                hidden_width=width, num_hidden_layers=layers
            ),
            save_plot_fn=save_density_plot,
        )

    elif mode == "remote":
        print(
            "Remote mode is not available for DEE-CC. Falling back to local run mode."
        )
        label, width, layers = _resolve_model_config(
            model_size=model_size,
            n_nodes=n_nodes,
            n_layers=n_layers,
        )
        case_prefix = f"dee_cc_{label}"
        run_density_inference_mode(
            mode="run",
            backend="local",
            ckpt_dir=ckpt_dir,
            case_prefix=case_prefix,
            plot_label=None,
            run_id=run_id,
            model_factory=lambda processor=None: CC_PINN(
                hidden_width=width, num_hidden_layers=layers
            ),
            save_plot_fn=save_density_plot,
        )

    else:
        raise ValueError("mode must be 'train', 'run', or 'remote'")
