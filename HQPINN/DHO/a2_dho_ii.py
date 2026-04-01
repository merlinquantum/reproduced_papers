# a2_dho_ii.py
# Interferometer-Interferometer PINN for the damped oscillator using oscillator_core + merlin_quantum

import os
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt

# Use non-interactive backend for batch image export
matplotlib.use("Agg")

import torch
import torch.nn as nn

from ..config import DHO_N_EPOCHS, DHO_PLOT_EVERY, DHO_LR, DTYPE
from ..utils import (
    count_trainable_params,
    get_latest_checkpoint,
    load_model,
    make_time_grid,
    make_optimizer,
    set_global_seed,
)
from .core_a2_dho import (
    append_summary_row,
    evaluate_dho_error,
    get_run_id_from_checkpoint,
    load_training_row_for_run_id,
    train_oscillator_pinn,
    u_exact,
)
from ..run_common import run_series_inference_mode
from ..layer_merlin import make_interf_qlayer, BranchMerlin


# ============================================================
#  MM_PINN model: two MerLin quantum branches
# ============================================================


class MM_PINN(nn.Module):
    """
    Interferometer-Interferometer PINN with linear fusion to scalar output.
    """

    def __init__(
        self,
        processor=None,
        *,
        n_photons: int = 1,
    ) -> None:
        super().__init__()

        # Two distinct quantum branches with independent parameters
        self.branch1 = BranchMerlin(
            make_interf_qlayer(n_photons=n_photons),
            processor=processor,
            feature_map_kind="dho",
        )
        self.branch2 = BranchMerlin(
            make_interf_qlayer(n_photons=n_photons),
            processor=processor,
            feature_map_kind="dho",
        )
        self.fusion = nn.Linear(2, 1, dtype=DTYPE)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        out1 = self.branch1(t)
        out2 = self.branch2(t)
        return self.fusion(torch.cat([out1, out2], dim=1))


def plot_model_prediction(u_pred, u_ex, t, save_path="HQPINN/DHO/results/dho_ii/"):
    plt.figure(figsize=(10, 6))
    plt.plot(t.cpu().numpy(), u_pred, label="Prediction PINN", lw=2)
    plt.plot(t.cpu().numpy(), u_ex, "--", label="Exact solution", lw=2)
    plt.xlabel("t")
    plt.ylabel("u(t)")
    plt.title("DHO – Interferometer–Interferometer PINN")
    plt.grid(True)
    plt.legend()

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    os.makedirs(save_path, exist_ok=True)
    save_path = os.path.join(save_path, f"dho_ii_plot_{timestamp}.png")

    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

    print(f"Plot saved to: {save_path}")


def _case_prefix(n_photons: int) -> str:
    if n_photons == 1:
        return "dho_ii"
    return f"dho_ii_p{n_photons}"


def run(
    mode="train",
    backend="sim:ascella",
    *,
    n_photons: int = 1,
) -> None:
    """
    mode = "train" : train the model from scratch and save the checkpoint
    mode = "run"   : load the latest checkpoint and run inference (not implemented here, but can be added)
    mode = "remote" : load and run in remote
    """
    set_global_seed(0)

    ckpt_dir = "HQPINN/DHO/models/"
    case_prefix = _case_prefix(n_photons)
    results_dir = f"HQPINN/DHO/results/{case_prefix}"
    summary_csv = "HQPINN/DHO/results/dho_summary.csv"
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")

    # ======================
    #  MODE TRAIN
    # ======================
    if mode == "train":
        print("=== TRAINING MODE ===")
        existing_ckpt = get_latest_checkpoint(ckpt_dir, case_prefix)
        if existing_ckpt is not None:
            try:
                model = load_model(
                    existing_ckpt,
                    lambda processor=None: MM_PINN(
                        processor=processor,
                        n_photons=n_photons,
                    ),
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
                    load_training_row_for_run_id(results_dir, "ii", case_run_id)
                    if case_run_id is not None
                    else None
                )
                append_summary_row(
                    summary_csv,
                    {
                        "run_id": case_run_id or "",
                        "Model": "ii",
                        "Size": str(n_photons),
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
                print(f"Skipping training for {case_prefix}: existing checkpoint found.")
                print(f"Summary CSV appended to: {summary_csv}")
                return

        model = MM_PINN(n_photons=n_photons)
        t_train = make_time_grid()

        train_oscillator_pinn(
            model=model,
            t_train=t_train,
            optimizer=make_optimizer(model, lr=DHO_LR),
            n_epochs=DHO_N_EPOCHS,
            plot_every=DHO_PLOT_EVERY,
            out_dir=results_dir,
            model_label="ii",
            run_id=run_id,
        )
        row = load_training_row_for_run_id(results_dir, "ii", run_id)
        append_summary_row(
            summary_csv,
            {
                "run_id": run_id,
                "Model": "ii",
                "Size": str(n_photons),
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

        # === Save model ===
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, f"{case_prefix}_{run_id}.pt")
        torch.save(model.state_dict(), ckpt_path)

        print(f"Model saved to: {ckpt_path}")
        print(f"Summary CSV appended to: {summary_csv}")

    # ======================
    #  MODE RUN
    # ======================
    elif mode == "run":
        print("=== RUN MODE ===")
        run_series_inference_mode(
            mode="run",
            backend=backend,
            ckpt_dir=ckpt_dir,
            case_prefix=case_prefix,
            model_factory=lambda processor=None: MM_PINN(
                processor=processor,
                n_photons=n_photons,
            ),
            make_time_grid=make_time_grid,
            exact_fn=u_exact,
            plot_fn=lambda u_pred, u_ex, t: plot_model_prediction(
                u_pred, u_ex, t, save_path=results_dir
            ),
        )

    # ======================
    #  MODE RUN REMOTE
    # ======================
    elif mode == "remote":
        run_series_inference_mode(
            mode="remote",
            backend=backend,
            ckpt_dir=ckpt_dir,
            case_prefix=case_prefix,
            model_factory=lambda processor=None: MM_PINN(
                processor=processor,
                n_photons=n_photons,
            ),
            make_time_grid=make_time_grid,
            exact_fn=u_exact,
            plot_fn=lambda u_pred, u_ex, t: plot_model_prediction(
                u_pred, u_ex, t, save_path=results_dir
            ),
        )

    else:
        raise ValueError("mode must be 'train', 'run', or 'remote'")
