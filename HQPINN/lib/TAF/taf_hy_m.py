# taf_hy_m.py
# Classical–Interferometer PINN for TAF (Sec. 3.3)

import os
from datetime import datetime

import torch
import torch.nn as nn

from ...config import (
    DEVICE,
    DTYPE,
    TAF_ADAM_STEPS,
    TAF_CC_HIDDEN_WIDTH,
    TAF_CC_NUM_HIDDEN_LAYERS,
    TAF_EPSILON_LAMBDA,
    TAF_LBFGS_STEPS,
    TAF_LR,
    TAF_N_OUTPUTS,
    TAF_PLOT_EVERY,
)
from ...layer_classical import BranchPyTorch
from ...layer_merlin import BranchMerlin, make_interf_qlayer
from ...run_common import run_density_inference_mode
from ...utils import (
    count_trainable_params,
    get_latest_checkpoint,
    make_optimizer,
)
from ...runtime import seed_everything
from .core_taf import (
    append_summary_row,
    get_run_id_from_checkpoint,
    load_training_row_for_run_id,
    load_training_sets,
    load_training_metrics_for_checkpoint,
    save_density_plot,
    train_taf,
)


class CI_PINN(nn.Module):
    """Classical-Interferometer TAF PINN with independent branch parameters."""

    def __init__(
        self,
        n_photons: int,
        hidden_width: int = TAF_CC_HIDDEN_WIDTH,
        num_hidden_layers: int = TAF_CC_NUM_HIDDEN_LAYERS,
        processor=None,
    ) -> None:
        super().__init__()

        self.branch1 = BranchPyTorch(
            in_features=2,
            out_features=TAF_N_OUTPUTS,
            num_hidden_layers=num_hidden_layers,
            hidden_width=hidden_width,
        )
        self.branch2 = BranchMerlin(
            make_interf_qlayer(n_photons=n_photons),
            n_outputs=TAF_N_OUTPUTS,
            processor=processor,
            feature_map_kind="taf",
        )
        self.fusion = nn.Linear(2 * TAF_N_OUTPUTS, TAF_N_OUTPUTS, dtype=DTYPE)

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        # Learned linear fusion of both branch outputs.
        out1 = self.branch1(xy)
        out2 = self.branch2(xy)
        return self.fusion(torch.cat([out1, out2], dim=1))


MODELS = [
    ("40-4-2", 40, 4, 2),
    ("40-7-2", 40, 7, 2),
    ("80-4-2", 80, 4, 2),
]


def _get_model_config(model_size: str) -> tuple[str, int, int, int]:
    for label, width, layers, n_photons in MODELS:
        if label == model_size:
            return label, width, layers, n_photons
    valid = ", ".join(label for label, *_ in MODELS)
    raise ValueError(f"Unknown model_size='{model_size}'. Valid values: {valid}")


def run(mode="train", backend="sim:ascella", model_size="40-4-2") -> None:
    """Run TAF classical-interferometer models and write summary CSV."""
    seed_everything(0)

    data = load_training_sets()

    # Sec. 3.3 inlet values (SI)
    U_in = torch.tensor([1.225, 272.15, 0.0, 288.15], dtype=DTYPE, device=DEVICE)

    ckpt_dir = "HQPINN/lib/TAF/"
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    if mode == "train":
        summary_csv = "HQPINN/lib/TAF/results/taf_summary.csv"

        for label, width, layers, n_photons in MODELS:
            seed_everything(0)
            print(
                f"\nTraining TAF-HY-M model: {label} "
                f"(width={width}, layers={layers}, {n_photons} photons)"
            )

            case_prefix = f"taf_hy_m_{label}"
            model_dir = os.path.join(ckpt_dir, "models")
            existing_ckpt = get_latest_checkpoint(model_dir, case_prefix)
            if existing_ckpt is not None:
                try:
                    torch.load(existing_ckpt, map_location="cpu")
                except Exception as exc:
                    print(
                        f"Checkpoint validation failed for {case_prefix} at "
                        f"{existing_ckpt}: {exc}; retraining model."
                    )
                else:
                    metrics = load_training_metrics_for_checkpoint(
                        out_dir=f"HQPINN/lib/TAF/results/{case_prefix}",
                        model_label=f"hy-m_{label}",
                        ckpt_path=existing_ckpt,
                        case_prefix=case_prefix,
                    )
                    if metrics is not None:
                        print(
                            f"Skipping training for {case_prefix}: existing checkpoint found at {existing_ckpt}."
                        )
                        n_params = count_trainable_params(
                            CI_PINN(
                                n_photons=n_photons,
                                hidden_width=width,
                                num_hidden_layers=layers,
                            )
                        )
                        case_run_id = get_run_id_from_checkpoint(
                            existing_ckpt, case_prefix
                        )
                        row = (
                            load_training_row_for_run_id(
                                out_dir=f"HQPINN/lib/TAF/results/{case_prefix}",
                                model_label=f"hy-m_{label}",
                                run_id=case_run_id,
                            )
                            if case_run_id is not None
                            else None
                        )
                        final_loss, _, _ = metrics
                        is_duplicate = append_summary_row(
                            summary_csv,
                            {
                                "run_id": case_run_id or "",
                                "Model": "hy-m",
                                "Size": label,
                                "step": row["step"] if row is not None else "",
                                "elapsed (s)": row["elapsed (s)"]
                                if row is not None
                                else "",
                                "Trainable parameters": n_params,
                                "Loss": row["Loss"]
                                if row is not None
                                else f"{final_loss:.6e}",
                                "BC": row["BC"] if row is not None else "",
                                "F": row["F"] if row is not None else "",
                                "L_in": row["L_in"] if row is not None else "",
                                "L_out": row["L_out"] if row is not None else "",
                                "L_wall": row["L_wall"] if row is not None else "",
                                "L_per": row["L_per"] if row is not None else "",
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

            model = CI_PINN(
                n_photons=n_photons,
                hidden_width=width,
                num_hidden_layers=layers,
            ).to(DEVICE)
            optimizer = make_optimizer(model, lr=TAF_LR)

            final_loss, loss_bc, loss_f, n_params = train_taf(
                model=model,
                optimizer=optimizer,
                n_epochs=TAF_ADAM_STEPS,
                plot_every=TAF_PLOT_EVERY,
                out_dir=f"HQPINN/lib/TAF/results/{case_prefix}",
                model_label=f"hy-m_{label}",
                run_id=run_id,
                data=data,
                U_in=U_in,
                lbfgs_steps=TAF_LBFGS_STEPS,
                eps_lambda=TAF_EPSILON_LAMBDA,
            )
            row = load_training_row_for_run_id(
                out_dir=f"HQPINN/lib/TAF/results/{case_prefix}",
                model_label=f"hy-m_{label}",
                run_id=run_id,
            )

            is_duplicate = append_summary_row(
                summary_csv,
                {
                    "run_id": run_id,
                    "Model": "hy-m",
                    "Size": label,
                    "step": row["step"] if row is not None else "",
                    "elapsed (s)": row["elapsed (s)"] if row is not None else "",
                    "Trainable parameters": n_params,
                    "Loss": row["Loss"] if row is not None else f"{final_loss:.6e}",
                    "BC": row["BC"] if row is not None else f"{loss_bc:.6e}",
                    "F": row["F"] if row is not None else f"{loss_f:.6e}",
                    "L_in": row["L_in"] if row is not None else "",
                    "L_out": row["L_out"] if row is not None else "",
                    "L_wall": row["L_wall"] if row is not None else "",
                    "L_per": row["L_per"] if row is not None else "",
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
        label, width, layers, n_photons = _get_model_config(model_size)
        case_prefix = f"taf_hy_m_{label}"
        run_density_inference_mode(
            mode="run",
            backend="local",
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
        case_prefix = f"taf_hy_m_{label}"
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
