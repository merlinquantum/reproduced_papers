# dee-ii.py
# Interferometer-Interferometer PINN

import os
from datetime import datetime

import torch
import torch.nn as nn

from ..config import (
    DEE_N_EPOCHS,
    DEE_LR,
    DEE_PLOT_EVERY,
    DTYPE,
)
from ..utils import (
    count_trainable_params,
    get_latest_checkpoint,
    load_model,
    make_optimizer,
)
from ..runtime import seed_everything
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
from ..layer_merlin import make_interf_qlayer, BranchMerlin


class II_PINN(nn.Module):
    """
    Interferometer-Interferometer PINN:

        u(t) = u_q1(t) + u_q2(t)

    Each branch uses its own QuantumLayer instance → independent parameters.
    """

    def __init__(self, n_photons: int, processor=None) -> None:
        super().__init__()

        # Two distinct quantum branches with independent parameters
        self.branch1 = BranchMerlin(
            make_interf_qlayer(n_photons=n_photons),
            n_outputs=3,
            processor=processor,
            feature_map_kind="dee",
        )
        self.branch2 = BranchMerlin(
            make_interf_qlayer(n_photons=n_photons),
            n_outputs=3,
            processor=processor,
            feature_map_kind="dee",
        )
        self.fusion = nn.Linear(6, 3, dtype=DTYPE)

        # Human-readable size label ("1", "2", ..., "6")
        self.size_label = f"{n_photons}"

    def forward(self, xt: torch.Tensor) -> torch.Tensor:
        # Learned linear fusion of both branch outputs.
        out1 = self.branch1(xt)  # [N, 3]
        out2 = self.branch2(xt)  # [N, 3]
        return self.fusion(torch.cat([out1, out2], dim=1))  # [N, 3]


MODELS = [
    ("1", 1),
    ("2", 2),
    ("3", 3),
    ("4", 4),
    ("5", 5),
    ("6", 6),
]

def run(mode="train", backend="sim:ascella", n_photons=2):
    """Run all DEE Interferometer-Interferometer models and write summary CSV."""
    seed_everything(0)

    ckpt_dir = "HQPINN/DEE/"
    # case_prefix = f"see_ii_{n_photons}"
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")

    # ======================
    #  MODE TRAIN
    # ======================

    if mode == "train":
        print("=== TRAINING MODE ===")
        summary_csv = "HQPINN/DEE/results/dee_summary.csv"
        for label, nb_photons in MODELS:
            seed_everything(0)
            print(f"\nTraining DEE-II {nb_photons} photons")

            case_prefix = f"dee_ii_{nb_photons}"
            model_dir = os.path.join(ckpt_dir, "models")
            existing_ckpt = get_latest_checkpoint(model_dir, case_prefix)
            if existing_ckpt is not None:
                final_loss = load_training_loss_for_checkpoint(
                    out_dir=f"HQPINN/DEE/results/{case_prefix}",
                    model_label=f"ii_{nb_photons}",
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
                            lambda processor=None: II_PINN(
                                n_photons=nb_photons, processor=processor
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
                                model_label=f"ii_{nb_photons}",
                                run_id=case_run_id,
                            )
                            if case_run_id is not None
                            else None
                        )
                        append_summary_row(
                            summary_csv,
                            {
                                "run_id": case_run_id or "",
                                "Model": "ii",
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

            model = II_PINN(n_photons=nb_photons)
            optimizer = make_optimizer(model, lr=DEE_LR)

            final_loss, err_rho, err_p, n_params = train_dee(
                model=model,
                t_train=None,  # kept for API consistency
                optimizer=optimizer,
                n_epochs=DEE_N_EPOCHS,
                plot_every=DEE_PLOT_EVERY,
                out_dir=f"HQPINN/DEE/results/{case_prefix}",
                model_label=f"ii_{nb_photons}",
                run_id=run_id,
            )
            row = load_training_row_for_run_id(
                out_dir=f"HQPINN/DEE/results/{case_prefix}",
                model_label=f"ii_{nb_photons}",
                run_id=run_id,
            )

            append_summary_row(
                summary_csv,
                {
                    "run_id": run_id,
                    "Model": "ii",
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

    # ======================
    #  MODE RUN
    # ======================

    elif mode == "run":
        case_prefix = f"dee_ii_{n_photons}"
        run_density_inference_mode(
            mode="run",
            backend=backend,
            ckpt_dir=ckpt_dir,
            case_prefix=case_prefix,
            plot_label=f"{n_photons} photons",
            run_id=run_id,
            model_factory=lambda processor=None: II_PINN(
                n_photons=n_photons, processor=processor
            ),
            save_plot_fn=save_density_plot,
        )

    # ======================
    #  MODE RUN REMOTE
    # ======================

    elif mode == "remote":
        case_prefix = f"dee_ii_{n_photons}"
        run_density_inference_mode(
            mode="remote",
            backend=backend,
            ckpt_dir=ckpt_dir,
            case_prefix=case_prefix,
            plot_label=f"{n_photons} photons",
            run_id=run_id,
            model_factory=lambda processor=None: II_PINN(
                n_photons=n_photons, processor=processor
            ),
            save_plot_fn=save_density_plot,
        )

    else:
        raise ValueError("mode must be 'train', 'run', or 'remote'")
