"""Runtime entrypoints for the Quantum Self-Supervised Learning reproduction."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch.nn as nn
from torch.utils.data import DataLoader
from torchsummary import summary

from lib.data_utils import load_finetuning_data, load_transformed_data
from lib.model import QSSL
from lib.training_utils import (
    disable_qiskit_statevector_capture,
    linear_evaluation,
    save_results_to_json,
    train,
)
from runtime_lib.data_paths import paper_data_dir

LOGGER = logging.getLogger(__name__)


def _as_str(value: Any) -> str:
    return str(value) if isinstance(value, Path) else str(value)


def _build_args(cfg: dict[str, Any]) -> SimpleNamespace:
    dataset_cfg = cfg.get("dataset", {})
    training_cfg = cfg.get("training", {})
    model_cfg = cfg.get("model", {})

    backend = str(model_cfg.get("backend", "classical")).lower()

    data_root = dataset_cfg.get("root")
    resolved_data_dir = paper_data_dir("qSSL", data_root)

    args_dict: dict[str, Any] = {
        "datadir": resolved_data_dir,
        "classes": int(dataset_cfg.get("classes", 2)),
        "batch_size": int(dataset_cfg.get("batch_size", 128)),
        "epochs": int(training_cfg.get("epochs", 2)),
        "le_epochs": int(training_cfg.get("le_epochs", 100)),
        "ckpt_step": int(training_cfg.get("ckpt_step", 1)),
        "max_steps": (
            None
            if training_cfg.get("max_steps") is None
            else int(training_cfg.get("max_steps"))
        ),
        "total_steps": (
            None
            if training_cfg.get("total_steps") is None
            else int(training_cfg.get("total_steps"))
        ),
        "le_max_steps": (
            None
            if training_cfg.get("le_max_steps") is None
            else int(training_cfg.get("le_max_steps"))
        ),
        "width": int(model_cfg.get("width", 8)),
        "loss_dim": int(model_cfg.get("loss_dim", 128)),
        "projection_head": bool(model_cfg.get("projection_head", True)),
        "batch_norm": bool(model_cfg.get("batch_norm", False)),
        "temperature": float(model_cfg.get("temperature", 0.07)),
        "modes": int(model_cfg.get("modes", 10)),
        "no_bunching": bool(model_cfg.get("no_bunching", False)),
        "layers": int(model_cfg.get("layers", 2)),
        "encoding": str(model_cfg.get("encoding", "vector")),
        "q_ansatz": str(model_cfg.get("q_ansatz", "sim_circ_14_half")),
        "q_sweeps": int(model_cfg.get("q_sweeps", 1)),
        "activation": str(model_cfg.get("activation", "null")),
        "shots": int(model_cfg.get("shots", 100)),
        "q_backend": str(model_cfg.get("q_backend", "qasm_simulator")),
        "save_dhs": bool(model_cfg.get("save_dhs", False)),
        "dhs_freq": int(model_cfg.get("dhs_freq", 1)),
        "device": str(cfg.get("device", "cpu")),
        "merlin": backend == "merlin",
        "qiskit": backend == "qiskit",
    }
    return SimpleNamespace(**args_dict)


def _serialize_args(args: SimpleNamespace) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            payload[key] = str(value)
        else:
            payload[key] = value
    return payload


def _make_dataloader(dataset: Any, batch_size: int) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def run_qssl_experiment(cfg: dict[str, Any], run_dir: Path) -> None:
    """Orchestrate the SSL + linear evaluation pipeline under the runtime."""

    run_dir = Path(run_dir)
    args = _build_args(cfg)

    args_path = run_dir / "args.json"
    args_path.write_text(json.dumps(_serialize_args(args), indent=2), encoding="utf-8")
    LOGGER.info("Saved resolved arguments to %s", args_path)

    LOGGER.info(
        "Preparing SSL dataset (classes=%s, batch_size=%s)",
        args.classes,
        args.batch_size,
    )
    ssl_dataset = load_transformed_data(args)
    ssl_loader = _make_dataloader(ssl_dataset, args.batch_size)

    model = QSSL(args)
    summary(model, [(3, 32, 32), (3, 32, 32)])

    LOGGER.info("Starting SSL training for %s epochs", args.epochs)
    model, ssl_losses = train(model, ssl_loader, str(run_dir), args)

    if args.qiskit:
        disable_qiskit_statevector_capture(model)
        LOGGER.info("Disabled Qiskit statevector capture after SSL training")

    LOGGER.info("Building frozen model for linear evaluation")
    frozen_representation_layers = [
        model.backbone,
        model.comp,
    ]
    if model.batch_norm:
        frozen_representation_layers.append(model.bn)
    frozen_representation_layers.extend(
        [
            model.quantum_input_preprocessor,
            model.representation_network,
        ]
    )
    frozen_model = nn.Sequential(
        *frozen_representation_layers,
        nn.Linear(model.rep_net_output_size, args.classes),
    )
    frozen_model.requires_grad_(False)
    frozen_model[-1].requires_grad_(True)

    LOGGER.info("Loading linear evaluation datasets")
    ft_train_dataset, ft_val_dataset = load_finetuning_data(args)
    ft_train_loader = _make_dataloader(ft_train_dataset, args.batch_size)
    ft_val_loader = _make_dataloader(ft_val_dataset, args.batch_size)

    (
        _,
        ft_train_losses,
        ft_val_losses,
        ft_train_accs,
        ft_val_accs,
    ) = linear_evaluation(
        frozen_model, ft_train_loader, ft_val_loader, args, str(run_dir)
    )

    save_results_to_json(
        args,
        ssl_losses,
        ft_train_losses,
        ft_val_losses,
        ft_train_accs,
        ft_val_accs,
        str(run_dir),
    )

    LOGGER.info("Finished qSSL experiment. Artifacts available under %s", run_dir)


def train_and_evaluate(cfg: dict[str, Any], run_dir: Path) -> None:
    """Adapter required by the shared runtime."""

    run_qssl_experiment(cfg, run_dir)


__all__ = ["run_qssl_experiment", "train_and_evaluate"]
