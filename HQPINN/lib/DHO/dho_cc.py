# dho_cc.py
# Classical–Classical PINN with two parallel MLP branches

import os
from datetime import datetime

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# Relative imports because this file is inside HQPINN/HQPINN/lib/DHO/
from ...config import (
    DHO_HIDDEN_WIDTH,
    DHO_LR,
    DHO_NUM_HIDDEN_LAYERS,
    DHO_N_EPOCHS,
    DHO_PLOT_EVERY,
    DTYPE,
)
from ...utils import (
    count_trainable_params,
    get_latest_checkpoint,
    load_model,
    make_time_grid,
    make_optimizer,
)
from ...runtime import seed_everything
from .core_dho import (
    append_summary_row,
    evaluate_dho_error,
    get_run_id_from_checkpoint,
    load_training_row_for_run_id,
    train_oscillator_pinn,
    u_exact,
)
from ...run_common import run_series_inference_mode
from ..layer_classical import DHOBranchPyTorch


class CC_PINN(nn.Module):
    """
    Classical–Classical PINN with two parallel MLP branches
    and a linear fusion readout to a scalar u(t).
    """

    def __init__(
        self,
        *,
        num_hidden_layers: int = DHO_NUM_HIDDEN_LAYERS,
        hidden_width: int = DHO_HIDDEN_WIDTH,
    ) -> None:
        super().__init__()
        self.branch1 = DHOBranchPyTorch(
            out_features=3,
            num_hidden_layers=num_hidden_layers,
            hidden_width=hidden_width,
        )
        self.branch2 = DHOBranchPyTorch(
            out_features=3,
            num_hidden_layers=num_hidden_layers,
            hidden_width=hidden_width,
        )
        self.fusion = nn.Linear(6, 1, dtype=DTYPE)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        out1 = self.branch1(t)
        out2 = self.branch2(t)
        return self.fusion(torch.cat([out1, out2], dim=1))


def plot_model_prediction(u_pred, u_ex, t, save_path="HQPINN/lib/DHO/results/dho_cc/"):
    plt.figure(figsize=(10, 6))
    plt.plot(t.cpu().numpy(), u_pred, label="Prediction PINN", lw=2)
    plt.plot(t.cpu().numpy(), u_ex, "--", label="Exact solution", lw=2)
    plt.xlabel("t")
    plt.ylabel("u(t)")
    plt.title("DHO - Classical-Classical PINN")
    plt.grid(True)
    plt.legend()

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs(save_path, exist_ok=True)
    png_path = os.path.join(save_path, f"dho_cc_plot_{timestamp}.png")
    plt.savefig(png_path, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to: {png_path}")


def _case_prefix(n_layers: int, n_nodes: int) -> str:
    if n_layers == DHO_NUM_HIDDEN_LAYERS and n_nodes == DHO_HIDDEN_WIDTH:
        return "dho_cc"
    return f"dho_cc_{n_nodes}-{n_layers}"


def run(
    mode="train",
    backend="sim:ascella",
    *,
    n_layers: int = DHO_NUM_HIDDEN_LAYERS,
    n_nodes: int = DHO_HIDDEN_WIDTH,
    force_retrain: bool = False,
):
    """Run the Classical–Classical DHO PINN experiment."""
    seed_everything(0)
    ckpt_dir = "HQPINN/lib/DHO/models"
    case_prefix = _case_prefix(n_layers, n_nodes)
    results_dir = f"HQPINN/lib/DHO/results/{case_prefix}"
    summary_csv = "HQPINN/lib/DHO/results/dho_summary.csv"
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")

    if mode == "train":
        existing_ckpt = None if force_retrain else get_latest_checkpoint(ckpt_dir, case_prefix)
        if force_retrain:
            print(f"Forcing retraining for {case_prefix}; existing checkpoints will be ignored.")
        if existing_ckpt is not None:
            try:
                model = load_model(
                    existing_ckpt,
                    lambda processor=None: CC_PINN(
                        num_hidden_layers=n_layers,
                        hidden_width=n_nodes,
                    ),
                )
            except Exception as exc:
                print(
                    f"Checkpoint validation failed for {case_prefix} at "
                    f"{existing_ckpt}: {exc}; retraining model."
                )
            else:
                t_train = make_time_grid()
                case_run_id = get_run_id_from_checkpoint(existing_ckpt, case_prefix)
                row = (
                    load_training_row_for_run_id(results_dir, "cc", case_run_id)
                    if case_run_id is not None
                    else None
                )
                is_duplicate = append_summary_row(
                    summary_csv,
                    {
                        "run_id": case_run_id or "",
                        "Model": "cc",
                        "Size": f"{n_nodes}-{n_layers}",
                        "epoch": row["epoch"] if row is not None else "",
                        "elapsed time (s)": row["elapsed time (s)"]
                        if row is not None
                        else "",
                        "Trainable parameters": count_trainable_params(model),
                        "Loss": row["Loss"] if row is not None else "",
                        "IC_u": row["IC_u"] if row is not None else "",
                        "IC_du": row["IC_du"] if row is not None else "",
                        "PDE": row["PDE"] if row is not None else "",
                        "Relative L2 error": f"{evaluate_dho_error(model, t_train):.6e}",
                    },
                )
                print(
                    f"Skipping training for {case_prefix}: existing checkpoint found at {existing_ckpt}."
                )
                if is_duplicate:
                    print(
                        f"Duplicate summary row appended for run_id={case_run_id} to: {summary_csv}"
                    )
                else:
                    print(f"Summary CSV appended to: {summary_csv}")
                print(f"Reused checkpoint metrics for {case_prefix}.")
                print()
                return

        model = CC_PINN(num_hidden_layers=n_layers, hidden_width=n_nodes)
        t_train = make_time_grid()
        train_oscillator_pinn(
            model=model,
            t_train=t_train,
            optimizer=make_optimizer(model, DHO_LR),
            n_epochs=DHO_N_EPOCHS,
            plot_every=DHO_PLOT_EVERY,
            out_dir=results_dir,
            model_label="cc",
            run_id=run_id,
        )
        row = load_training_row_for_run_id(results_dir, "cc", run_id)
        is_duplicate = append_summary_row(
            summary_csv,
            {
                "run_id": run_id,
                "Model": "cc",
                "Size": f"{n_nodes}-{n_layers}",
                "epoch": row["epoch"] if row is not None else "",
                "elapsed time (s)": row["elapsed time (s)"] if row is not None else "",
                "Trainable parameters": count_trainable_params(model),
                "Loss": row["Loss"] if row is not None else "",
                "IC_u": row["IC_u"] if row is not None else "",
                "IC_du": row["IC_du"] if row is not None else "",
                "PDE": row["PDE"] if row is not None else "",
                "Relative L2 error": f"{evaluate_dho_error(model, t_train):.6e}",
            },
        )
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, f"{case_prefix}_{run_id}.pt")
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
        run_series_inference_mode(
            mode="run",
            backend="local",
            ckpt_dir=ckpt_dir,
            case_prefix=case_prefix,
            model_factory=lambda processor=None: CC_PINN(
                num_hidden_layers=n_layers,
                hidden_width=n_nodes,
            ),
            make_time_grid=make_time_grid,
            exact_fn=u_exact,
            plot_fn=lambda u_pred, u_ex, t: plot_model_prediction(
                u_pred, u_ex, t, save_path=results_dir
            ),
        )

    elif mode == "remote":
        print(
            "Remote mode is not available for DHO-CC. Falling back to local run mode."
        )
        run_series_inference_mode(
            mode="run",
            backend="local",
            ckpt_dir=ckpt_dir,
            case_prefix=case_prefix,
            model_factory=lambda processor=None: CC_PINN(
                num_hidden_layers=n_layers,
                hidden_width=n_nodes,
            ),
            make_time_grid=make_time_grid,
            exact_fn=u_exact,
            plot_fn=lambda u_pred, u_ex, t: plot_model_prediction(
                u_pred, u_ex, t, save_path=results_dir
            ),
        )

    else:
        raise ValueError("mode must be 'train', 'run', or 'remote'")
