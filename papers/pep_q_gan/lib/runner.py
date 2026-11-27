"""Shared entry point for the selected MerLin fixed-Fock experiment."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
from pathlib import Path

import numpy as np
import torch

from .artifacts import write_json
from .circuits import (
    SCHEMA,
    circuit_from_spec,
    circuit_id,
    generate_spec,
    input_state,
    save_circuit,
)


def train_and_evaluate(cfg: dict, run_dir: Path) -> dict:
    """Inspect the release experiment or explicitly request a later-stage action.

    Parameters
    ----------
    cfg : dict
        Resolved shared-runtime configuration.
    run_dir : Path
        Timestamped directory supplied by the common runner.

    Returns
    -------
    dict
        Scope inspection or an explicitly requested sampling/training result.
    """
    if cfg["stage"] == "prepare-data":
        from runtime_lib.data_paths import paper_data_dir

        from .data_acquisition import prepare_reference

        root = (
            Path(cfg["dataset"]["root"])
            if cfg["dataset"]["root"]
            else paper_data_dir("pep_q_gan", ensure_exists=False)
        )
        result = prepare_reference(root, Path(__file__).resolve().parents[1])
        write_json(run_dir / "data_summary.json", result)
        return result
    if cfg["stage"] == "evaluate":
        from .evaluation import evaluate_existing

        return evaluate_existing(
            cfg["evaluation"],
            run_dir,
            device=str(cfg["device"]),
            threads=cfg["sampling"]["threads"],
        )
    torch.set_num_threads(cfg["sampling"]["threads"])
    import exqalibur as xq

    xq.Config.compute_max_thread_count = cfg["sampling"]["threads"]
    xq.Config.use_single_precision = False
    circuit_cfg = cfg["circuit"]
    if (
        circuit_cfg["modes"] != 32
        or circuit_cfg["layers"] != 1
        or cfg["sampling"]["photons"] != [16]
    ):
        raise ValueError(
            "The public experiment requires 32 modes, one layer, and 16 photons"
        )
    spec = generate_spec(
        circuit_cfg["modes"], circuit_cfg["layers"], circuit_cfg["seed"]
    )
    source = cfg["training"]["prior_source"]
    if source not in ("gaussian", "dist", "indist"):
        raise ValueError("Release priors are gaussian, dist, and indist")
    stage = cfg["stage"]
    if stage == "inspect":
        _, unitary = save_circuit(spec, run_dir / "circuit")
        result = {
            "status": "complete",
            "stage": "inspect",
            "scope": "MerLin one-layer fixed-Fock Dist/Indist with a Gaussian reference",
            "circuit": spec,
            "circuit_id": circuit_id(spec),
            "unitary_sha256": hashlib.sha256(unitary.tobytes()).hexdigest(),
            "unitarity_max_error": float(
                np.max(abs(unitary.conj().T @ unitary - np.eye(32)))
            ),
            "input_state": input_state(16, 32),
            "priors": ["dist", "indist", "gaussian"],
            "campaign_seeds": cfg["experiment"]["seeds"],
            "events_sampled": 0,
            "training_steps": 0,
            "binding_evaluation_run": False,
            "netmhcpan_used": False,
            "versions": {
                p: importlib.metadata.version(p)
                for p in ("merlinquantum", "perceval-quandela", "torch", "numpy")
            },
        }
        write_json(run_dir / "scope_summary.json", result)
        return result
    if stage == "sample":
        from .classical_pools import generate_classical_pool
        from .devices import checked_device
        from .sampling import generate_pool

        pools_root = (
            Path(cfg["sampling"]["pool_root"])
            if cfg["sampling"]["pool_root"]
            else run_dir / "pools"
        )
        if source == "gaussian":
            return generate_classical_pool(source, cfg["sampling"], pools_root / source)
        backend = cfg["sampling"]["backend"]
        if backend != "perceval":
            raise ValueError("The selected 16-photon event path uses Perceval on CPU")
        checked_device(cfg["sampling"]["device"], backend)
        circuit, unitary = save_circuit(spec, run_dir / "circuit")
        return generate_pool(
            circuit,
            unitary,
            spec,
            16,
            backend,
            cfg["sampling"],
            pools_root / source,
            condition=source,
        )
    if stage == "train":
        return _train(cfg, run_dir, spec)
    raise ValueError("Stage must be inspect, sample, train, prepare-data, or evaluate")


def _train(cfg, run_dir, spec):
    from runtime_lib.data_paths import paper_data_dir
    from runtime_lib.dtypes import dtype_label

    from .devices import checked_device
    from .priors import FockPoolPrior
    from .training import TrainingConfig, run_training

    if dtype_label(cfg["dtype"]) != "float32":
        raise ValueError("GAN training requires float32")
    device = checked_device(cfg["device"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if device.type == "cpu":
        logging.warning(
            "Full training can take a very long time without a GPU; use CPU for small checks"
        )
    training = cfg["training"]
    source = training["prior_source"]
    prior = None
    if source in {"dist", "indist"}:
        from .sampling import SAMPLER_SCHEMA

        if not training["pool"]:
            raise ValueError("Set --pool to a completed matching photonic pool")
        metadata = json.loads(
            (Path(training["pool"]) / "complete.json").read_text(encoding="utf-8")
        )
        identity = metadata["identity"]
        if (
            identity.get("circuit_schema") != SCHEMA
            or identity.get("sampler_schema") != SAMPLER_SCHEMA
            or identity.get("condition") != source
            or identity["circuit_id"] != circuit_id(spec)
            or identity["photons"] != 16
        ):
            raise ValueError(
                "Pool does not belong to the configured selected MerLin circuit"
            )
        _, unitary = circuit_from_spec(spec)
        if (
            identity["unitary_sha256"]
            != hashlib.sha256(np.ascontiguousarray(unitary).tobytes()).hexdigest()
        ):
            raise ValueError(
                "Pool unitary does not match the configured MerLin circuit"
            )
        prior = FockPoolPrior(training["pool"])
    elif training["pool"]:
        from .classical_pools import ClassicalPoolPrior

        prior = ClassicalPoolPrior(training["pool"], source)
    data_dir = (
        Path(cfg["dataset"]["root"])
        if cfg["dataset"]["root"]
        else paper_data_dir("pep_q_gan", cfg["data_root"], ensure_exists=False)
    )
    config = TrainingConfig(
        prior=source,
        seed=cfg["seed"],
        device=str(device),
        batch_size=training["batch_size"],
        max_steps=training["steps"],
        learning_rate=training["learning_rate"],
        weight_decay=training["weight_decay"],
        discriminator_iterations=training["discriminator_iterations"],
        gradient_penalty=training["gradient_penalty"],
        warmup_steps=training["warmup_steps"],
        rng_mode=training["rng_mode"],
    )
    target = (
        Path(training["resume_dir"]) if training["resume_dir"] else run_dir / "training"
    )
    return run_training(
        config,
        data_dir,
        target,
        prior=prior,
        resume=bool(training["resume_dir"]),
        save_every=training["save_every"],
    )
