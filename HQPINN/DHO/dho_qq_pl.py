# dho_qq_pl.py
# PennyLane–PennyLane PINN with two parallel quantum branches

import os
from datetime import datetime

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from ..config import (
    DEFAULT_N_OUTPUTS,
    DHO_N_EPOCHS,
    DHO_PLOT_EVERY,
    DHO_LR,
    N_LAYERS,
    DTYPE,
)
from ..utils import (
    count_trainable_params,
    get_latest_checkpoint,
    load_model,
    make_time_grid,
    make_optimizer,
)
from ..runtime import seed_everything
from .core_dho import (
    append_summary_row,
    evaluate_dho_error,
    get_run_id_from_checkpoint,
    load_training_row_for_run_id,
    train_oscillator_pinn,
    u_exact,
)
from ..run_common import run_series_inference_mode
from ..layer_pennylane import make_quantum_block, dho_feature_map, BranchPennylane
from ..layer_classical import LearnedScalarFusion


class PP_PINN(nn.Module):
    """
    Physics-Informed model with two independent quantum branches
    and a linear fusion to scalar output.
    """

    def __init__(
        self,
        *,
        n_qubits: int = DEFAULT_N_OUTPUTS,
    ) -> None:
        super().__init__()
        qblock1 = make_quantum_block(n_qubits=n_qubits)
        qblock2 = make_quantum_block(n_qubits=n_qubits)

        # Two distinct branches => two independent parameter sets
        self.branch1 = BranchPennylane(
            qblock1,
            feature_map=lambda t: dho_feature_map(t, n_qubits=n_qubits),
            output_as_column=True,
            n_layers=N_LAYERS,
            n_qubits=n_qubits,
        )
        self.branch2 = BranchPennylane(
            qblock2,
            feature_map=lambda t: dho_feature_map(t, n_qubits=n_qubits),
            output_as_column=True,
            n_layers=N_LAYERS,
            n_qubits=n_qubits,
        )
        self.fusion = LearnedScalarFusion()

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        out1 = self.branch1(t)
        out2 = self.branch2(t)
        return self.fusion(out1, out2)


def plot_model_prediction(u_pred, u_ex, t, save_path="HQPINN/DHO/results/dho_pp/"):
    plt.figure(figsize=(10, 6))
    plt.plot(t.cpu().numpy(), u_pred, label="Prediction PINN", lw=2)
    plt.plot(t.cpu().numpy(), u_ex, "--", label="Exact solution", lw=2)
    plt.xlabel("t")
    plt.ylabel("u(t)")
    plt.title("DHO - PennyLane-PennyLane PINN")
    plt.grid(True)
    plt.legend()

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs(save_path, exist_ok=True)
    png_path = os.path.join(save_path, f"dho_pp_plot_{timestamp}.png")
    plt.savefig(png_path, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to: {png_path}")


def _case_prefix(n_qubits: int) -> str:
    if n_qubits == DEFAULT_N_OUTPUTS:
        return "dho_pp"
    return f"dho_pp_q{n_qubits}"


def run(
    mode="train",
    backend="sim:ascella",
    *,
    n_qubits: int = DEFAULT_N_OUTPUTS,
):
    seed_everything(0)
    ckpt_dir = "HQPINN/DHO/models"
    case_prefix = _case_prefix(n_qubits)
    results_dir = f"HQPINN/DHO/results/{case_prefix}"
    summary_csv = "HQPINN/DHO/results/dho_summary.csv"
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    if mode == "train":
        existing_ckpt = get_latest_checkpoint(ckpt_dir, case_prefix)
        if existing_ckpt is not None:
            try:
                model = load_model(
                    existing_ckpt,
                    lambda processor=None: PP_PINN(n_qubits=n_qubits),
                )
            except Exception as exc:
                print(
                    f"Existing checkpoint found for {case_prefix} at "
                    f"{existing_ckpt}, but loading failed: {exc}; retraining model."
                )
            else:
                t_train = make_time_grid()
                case_run_id = get_run_id_from_checkpoint(existing_ckpt, case_prefix)
                row = (
                    load_training_row_for_run_id(results_dir, "pp", case_run_id)
                    if case_run_id is not None
                    else None
                )
                append_summary_row(
                    summary_csv,
                    {
                        "run_id": case_run_id or "",
                        "Model": "pp",
                        "Size": str(n_qubits),
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
                    f"Skipping training for {case_prefix}: existing checkpoint found."
                )
                print(f"Summary CSV appended to: {summary_csv}")
                return

        model = PP_PINN(n_qubits=n_qubits)
        t_train = make_time_grid()
        train_oscillator_pinn(
            model=model,
            t_train=t_train,
            optimizer=make_optimizer(model, lr=DHO_LR),
            n_epochs=DHO_N_EPOCHS,
            plot_every=DHO_PLOT_EVERY,
            out_dir=results_dir,
            model_label="pp",
            run_id=run_id,
        )
        row = load_training_row_for_run_id(results_dir, "pp", run_id)
        append_summary_row(
            summary_csv,
            {
                "run_id": run_id,
                "Model": "pp",
                "Size": str(n_qubits),
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
        print(f"Summary CSV appended to: {summary_csv}")

    elif mode == "run":
        run_series_inference_mode(
            mode="run",
            backend="local",
            ckpt_dir=ckpt_dir,
            case_prefix=case_prefix,
            model_factory=lambda processor=None: PP_PINN(n_qubits=n_qubits),
            make_time_grid=make_time_grid,
            exact_fn=u_exact,
            plot_fn=lambda u_pred, u_ex, t: plot_model_prediction(
                u_pred, u_ex, t, save_path=results_dir
            ),
        )

    elif mode == "remote":
        print(
            "Remote mode is not available for DHO-PP. Falling back to local run mode."
        )
        run_series_inference_mode(
            mode="run",
            backend="local",
            ckpt_dir=ckpt_dir,
            case_prefix=case_prefix,
            model_factory=lambda processor=None: PP_PINN(n_qubits=n_qubits),
            make_time_grid=make_time_grid,
            exact_fn=u_exact,
            plot_fn=lambda u_pred, u_ex, t: plot_model_prediction(
                u_pred, u_ex, t, save_path=results_dir
            ),
        )

    else:
        raise ValueError("mode must be 'train', 'run', or 'remote'")
