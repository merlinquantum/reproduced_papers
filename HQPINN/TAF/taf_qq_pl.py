# taf_qq_pl.py
# PennyLane–PennyLane PINN for TAF (Sec. 3.3)

import os
from datetime import datetime

import torch
import torch.nn as nn

from ..config import (
    DEVICE,
    DTYPE,
    N_LAYERS,
    TAF_ADAM_STEPS,
    TAF_EPSILON_LAMBDA,
    TAF_LBFGS_STEPS,
    TAF_LR,
    TAF_N_OUTPUTS,
    TAF_PLOT_EVERY,
)
from ..layer_pennylane import (
    BranchPennylane,
    make_quantum_block_multiout,
    taf_feature_map,
)
from ..run_common import run_density_inference_mode
from ..utils import (
    count_trainable_params,
    get_latest_checkpoint,
    make_optimizer,
)
from ..runtime import seed_everything
from .core_taf import (
    append_summary_row,
    get_run_id_from_checkpoint,
    load_training_row_for_run_id,
    load_training_sets,
    load_training_metrics_for_checkpoint,
    save_density_plot,
    train_taf,
)


class PP_PINN(nn.Module):
    """PennyLane-PennyLane TAF PINN with two independent quantum branches."""

    def __init__(
        self,
        q_layers: int = N_LAYERS,
        *,
        n_layers: int | None = None,
    ) -> None:
        super().__init__()
        if n_layers is not None:
            # Backward-compatible alias while the rest of the repo catches up.
            q_layers = n_layers

        qblock_multi_1 = make_quantum_block_multiout(
            n_layers=q_layers, n_qubits=TAF_N_OUTPUTS
        )
        qblock_multi_2 = make_quantum_block_multiout(
            n_layers=q_layers, n_qubits=TAF_N_OUTPUTS
        )

        self.branch1 = BranchPennylane(
            qblock_multi_1,
            feature_map=taf_feature_map,
            output_as_column=False,
            n_layers=q_layers,
            n_qubits=TAF_N_OUTPUTS,
        )
        self.branch2 = BranchPennylane(
            qblock_multi_2,
            feature_map=taf_feature_map,
            output_as_column=False,
            n_layers=q_layers,
            n_qubits=TAF_N_OUTPUTS,
        )

        # Two branches of TAF_N_OUTPUTS quantum outputs each.
        self.fusion = nn.Linear(2 * TAF_N_OUTPUTS, TAF_N_OUTPUTS, dtype=DTYPE)
        self.size_label = f"{q_layers}"

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        out1 = self.branch1(xy)  # [N, TAF_N_OUTPUTS]
        out2 = self.branch2(xy)  # [N, TAF_N_OUTPUTS]
        return self.fusion(torch.cat([out1, out2], dim=1))  # [N, TAF_N_OUTPUTS]


MODELS = [
    ("2", 2),
    ("4", 4),
    ("6", 6),
]


def _get_model_config(model_size: str) -> tuple[str, int]:
    for label, q_layers in MODELS:
        if label == model_size:
            return label, q_layers
    valid = ", ".join(label for label, _ in MODELS)
    raise ValueError(f"Unknown model_size='{model_size}'. Valid values: {valid}")


def run(mode="train", backend="sim:ascella", model_size="2") -> None:
    """Run TAF PennyLane-PennyLane models and write summary CSV."""
    seed_everything(0)

    data = load_training_sets()

    # Sec. 3.3 inlet values (SI)
    U_in = torch.tensor([1.225, 272.15, 0.0, 288.15], dtype=DTYPE, device=DEVICE)

    ckpt_dir = "HQPINN/TAF/"
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    if mode == "train":
        summary_csv = "HQPINN/TAF/results/taf_summary.csv"

        for label, q_layers in MODELS:
            seed_everything(0)
            print(f"\nTraining TAF-PP model: {label} q_layers={q_layers}")

            case_prefix = f"taf_pp_{label}"
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
                        out_dir=f"HQPINN/TAF/results/{case_prefix}",
                        model_label=f"pp_{label}",
                        ckpt_path=existing_ckpt,
                        case_prefix=case_prefix,
                    )
                    if metrics is not None:
                        print(
                            f"Skipping {case_prefix}: existing checkpoint found at "
                            f"{existing_ckpt}"
                        )
                        n_params = count_trainable_params(PP_PINN(q_layers=q_layers))
                        case_run_id = get_run_id_from_checkpoint(
                            existing_ckpt, case_prefix
                        )
                        row = (
                            load_training_row_for_run_id(
                                out_dir=f"HQPINN/TAF/results/{case_prefix}",
                                model_label=f"pp_{label}",
                                run_id=case_run_id,
                            )
                            if case_run_id is not None
                            else None
                        )
                        final_loss, _, _ = metrics
                        append_summary_row(
                            summary_csv,
                            {
                                "run_id": case_run_id or "",
                                "Model": "pp",
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
                        print(
                            f"Reused latest metrics for {case_prefix} in summary CSV."
                        )
                        continue
                    print(
                        f"Existing checkpoint found for {case_prefix} at "
                        f"{existing_ckpt}, but no matching metrics CSV was found; "
                        f"retraining model."
                    )

            model = PP_PINN(q_layers=q_layers).to(DEVICE)
            optimizer = make_optimizer(model, lr=TAF_LR)

            final_loss, loss_bc, loss_f, n_params = train_taf(
                model=model,
                optimizer=optimizer,
                n_epochs=TAF_ADAM_STEPS,
                plot_every=TAF_PLOT_EVERY,
                out_dir=f"HQPINN/TAF/results/{case_prefix}",
                model_label=f"pp_{label}",
                run_id=run_id,
                data=data,
                U_in=U_in,
                lbfgs_steps=TAF_LBFGS_STEPS,
                eps_lambda=TAF_EPSILON_LAMBDA,
            )
            row = load_training_row_for_run_id(
                out_dir=f"HQPINN/TAF/results/{case_prefix}",
                model_label=f"pp_{label}",
                run_id=run_id,
            )

            append_summary_row(
                summary_csv,
                {
                    "run_id": run_id,
                    "Model": "pp",
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

        print(f"Summary CSV appended to: {summary_csv}")

    elif mode == "run":
        label, q_layers = _get_model_config(model_size)
        case_prefix = f"taf_pp_{label}"
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
            "Remote mode is not available for TAF-PP. Falling back to local run mode."
        )
        label, q_layers = _get_model_config(model_size)
        case_prefix = f"taf_pp_{label}"
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
