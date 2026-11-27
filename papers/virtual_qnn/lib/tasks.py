"""Per-task orchestration: build data/model, train once, evaluate, save artifacts.

Every task function has the signature ``(cfg, run_dir, generator) -> dict`` and
returns the JSON-serializable metrics of exactly one training (or solve). The
lifecycle, status file and metric writing live in ``lib/runner.py``.

Tasks
-----
- ``min_overhead_sdp`` (E0): App. D SDP for each configured noise channel.
- ``vqem`` (E1-E3): Alg. 1 (``algorithm = "penalty"``) or Alg. 2
  (``algorithm = "budget"``).
- ``vqc`` (E4): virtual quantum classifier and its quantum baselines.
- ``classical_classifier`` (E4 baselines): logistic regression / small MLP.
- ``vvqe`` (E5): virtual VQE on the HeH+ Hamiltonian; ``budget = 1`` is VQE.
"""

from __future__ import annotations

import csv
import json
import logging
import math
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from torch import nn

from . import classical, data, sdp, settings, shot_budget, vqc, vqc_merlin, vqem, vvqe
from .training import optimize

logger = logging.getLogger(__name__)


def describe_dataset(cfg: dict) -> dict:
    """Dataset identity recorded in ``run_status.json`` before data construction."""
    task = cfg["experiment"]["task"]
    if task in ("vqc", "vqc_merlin", "classical_classifier"):
        data_cfg = cfg["data"]
        return {
            "name": f"circle_r0.7_{data_cfg['source']}",
            "splits": {"train": data_cfg["n_train"], "test": data_cfg["n_test"]},
            "preprocessing": "none (features in [-1, 1]); authors' rejection-sampling rule",
            "subset": f"seed_data={data_cfg['seed_data']}, boundary_gap={data_cfg['boundary_gap']}",
        }
    if task == "vvqe":
        return {
            "name": "HeH+_STO-3G_JW_tapered_2q (PennyLane qchem)",
            "splits": "n/a (variational ground-state problem)",
            "preprocessing": "Z2 qubit tapering, 2-electron sector",
            "subset": f"half_distance_bohr={cfg['vvqe']['half_distance_bohr']}",
        }
    if task == "vqem":
        vqem_cfg = cfg["vqem"]
        return {
            "name": f"{vqem_cfg['noise_model']}_channel",
            "splits": "n/a (Choi-state fitting)",
            "preprocessing": "none",
            "subset": f"noise_level={vqem_cfg['noise_level']}",
        }
    if task == "equal_shot_eval":
        eq = cfg["equal_shot"]
        return {
            "name": f"trained models from Phase 3/4 sweeps ({eq['experiment']})",
            "splits": "test split of each trained model (E4) / n/a (E5)",
            "preprocessing": "none; trained parameters reloaded from model_state.pt / virtual_state.npz",
            "subset": json.dumps(eq["sweeps"], sort_keys=True),
        }
    if task == "min_overhead_sdp":
        return {
            "name": "single_qubit_noise_channels",
            "splits": "n/a (convex optimization)",
            "preprocessing": "none",
            "subset": ";".join(
                f"{c['noise_model']}={c['noise_level']}" for c in cfg["sdp"]["channels"]
            ),
        }
    raise ValueError(f"Unsupported experiment.task={task!r}")


def describe_selection(cfg: dict) -> dict:
    """Checkpoint/selection rule recorded in ``run_status.json``."""
    task = cfg["experiment"]["task"]
    if task == "equal_shot_eval":
        return {
            "name": "equal-shot RMSE (E5) / expected test accuracy (E4)",
            "direction": "minimize"
            if cfg["equal_shot"]["experiment"] == "e5"
            else "maximize",
            "split": "test" if cfg["equal_shot"]["experiment"] == "e4" else "n/a",
            "checkpoint": "final parameters of each trained model (no re-selection)",
        }
    if task == "min_overhead_sdp":
        return {
            "name": "gamma_min",
            "direction": "minimize",
            "split": "n/a",
            "checkpoint": "solver optimum",
        }
    if task == "classical_classifier":
        return {
            "name": "validation_accuracy (learning-rate grid)",
            "direction": "maximize",
            "split": "validation (held out from train)",
            "checkpoint": "final epoch, refit on full train",
        }
    if task in ("vqc", "vqc_merlin"):
        return {
            "name": "train_loss",
            "direction": "minimize",
            "split": "train",
            "checkpoint": "final iteration (no early stopping)",
        }
    return {
        "name": "loss",
        "direction": "minimize",
        "split": "n/a",
        "checkpoint": "final iteration (no early stopping)",
    }


def _write_history(path: Path, history: list[dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    logger.info("ARTIFACT_WRITTEN | path=%s | rows=%d", path.name, len(history))


def _save_state(path: Path, model: nn.Module) -> None:
    torch.save(model.state_dict(), path)
    logger.info("ARTIFACT_WRITTEN | path=%s", path.name)


def run_min_overhead_sdp(cfg: dict, run_dir: Path, generator: torch.Generator) -> dict:
    """E0: solve the App. D SDP for every configured channel."""
    channels = cfg["sdp"]["channels"]
    logger.info("DATASET_READY | channels=%d", len(channels))
    logger.info("EVALUATION_STARTED | split=n/a | solver=%s", cfg["sdp"]["solver"])
    results = []
    for channel in channels:
        model, level = channel["noise_model"], float(channel["noise_level"])
        solution = sdp.min_sampling_overhead(model, level, solver=cfg["sdp"]["solver"])
        entry = {"noise_model": model, "noise_level": level, **solution}
        if model == "depolarizing":
            entry["gamma_min_analytic"] = sdp.depolarizing_min_overhead_analytic(level)
        results.append(entry)
        logger.info(
            "EVALUATION_COMPLETED | channel=%s(%.3g) | gamma_min=%.8f | status=%s",
            model,
            level,
            solution["gamma_min"],
            solution["status"],
        )
    return {"task": "min_overhead_sdp", "channels": results}


def run_vqem(cfg: dict, run_dir: Path, generator: torch.Generator) -> dict:
    """E1-E3: train one VQEM model (Alg. 1 or Alg. 2) and evaluate it."""
    vqem_cfg = cfg["vqem"]
    device = settings.device(cfg)
    algorithm = vqem_cfg["algorithm"]
    coefficient_mode = {"penalty": "trainable", "budget": "budget"}[algorithm]
    model = vqem.VQEMModel(
        noise_model=vqem_cfg["noise_model"],
        noise_level=float(vqem_cfg["noise_level"]),
        depth=int(vqem_cfg["depth"]),
        coefficient_mode=coefficient_mode,
        initial_t=float(vqem_cfg["initial_t"]),
        budget=float(vqem_cfg["budget"]),
        gate_noise=tuple(vqem_cfg["gate_noise"]),
        generator=generator,
        complex_dtype=settings.complex_dtype(cfg),
    ).to(device)
    unmitigated = model.unmitigated_hs_squared(device)
    logger.info(
        "DATASET_READY | channel=%s(%.3g) | unmitigated_hs_squared=%.6g",
        vqem_cfg["noise_model"],
        vqem_cfg["noise_level"],
        unmitigated,
    )
    penalty = float(vqem_cfg["penalty"])

    def loss_fn():
        outputs = model(device)
        loss = vqem.vqem_loss(outputs, algorithm, penalty)
        return loss, {
            "hs_squared": float(outputs["hs_squared"].detach()),
            "overhead": float(outputs["overhead"].detach()),
        }

    history = optimize(model.parameters(), loss_fn, cfg["training"])
    _write_history(run_dir / "losses.csv", history)

    logger.info("EVALUATION_STARTED | split=n/a | state=final parameters")
    with torch.no_grad():
        outputs = model(device)
        final_loss = float(vqem.vqem_loss(outputs, algorithm, penalty))
    coefficients = [float(c) for c in outputs["coefficients"]]
    sigma = outputs["sigma"].detach().cpu().numpy()
    np.savez(
        run_dir / "virtual_state.npz", sigma=sigma, coefficients=np.array(coefficients)
    )
    logger.info("ARTIFACT_WRITTEN | path=virtual_state.npz")
    _save_state(run_dir / "model_state.pt", model)
    metrics = {
        "task": "vqem",
        "algorithm": algorithm,
        "noise_model": vqem_cfg["noise_model"],
        "noise_level": float(vqem_cfg["noise_level"]),
        "depth": int(vqem_cfg["depth"]),
        "gate_noise": list(vqem_cfg["gate_noise"]),
        "penalty": penalty if algorithm == "penalty" else None,
        "budget": float(vqem_cfg["budget"]) if algorithm == "budget" else None,
        "hs_squared": float(outputs["hs_squared"]),
        "overhead": float(outputs["overhead"]),
        "coefficients": coefficients,
        "final_loss": final_loss,
        "unmitigated_hs_squared": unmitigated,
        "trace_sigma": float(np.real(np.trace(sigma))),
        "n_parameters": sum(p.numel() for p in model.parameters()),
        "iterations": len(history),
    }
    logger.info(
        "EVALUATION_COMPLETED | hs_squared=%.6g | overhead=%.6f | c=(%.6f, %.6f)",
        metrics["hs_squared"],
        metrics["overhead"],
        *coefficients,
    )
    return metrics


def _classification_tensors(cfg: dict) -> dict[str, torch.Tensor]:
    train = next(iter(data.build_dataloader(cfg, "train")))
    test = next(iter(data.build_dataloader(cfg, "test")))
    return {
        "train_x": train[0],
        "train_y": train[1],
        "test_x": test[0],
        "test_y": test[1],
    }


def _save_dataset(run_dir: Path, tensors: dict[str, torch.Tensor]) -> None:
    np.savez(
        run_dir / "dataset.npz", **{k: v.cpu().numpy() for k, v in tensors.items()}
    )
    logger.info("ARTIFACT_WRITTEN | path=dataset.npz")


def _fit_and_evaluate_classifier(
    cfg: dict, run_dir: Path, model: nn.Module, tensors: dict, metrics: dict
) -> dict:
    """Train a (virtual) classifier full-batch with MSE, then evaluate both splits.

    Shared by the gate-model VQC (E4) and its photonic MerLin adaptation, so
    loss, accuracy rule, optimizer, logging and artifacts are identical.
    """

    def loss_fn():
        predictions = model(tensors["train_x"])
        loss = ((predictions - tensors["train_y"]) ** 2).mean()
        return loss, {
            "train_accuracy": vqc.reference_accuracy(
                predictions.detach(), tensors["train_y"]
            )
        }

    history = optimize(model.parameters(), loss_fn, cfg["training"])
    _write_history(run_dir / "losses.csv", history)
    metrics.update(
        {
            "budget": model.budget,
            "coefficients": [model.c1, model.c2],
            "shot_multiplier": model.budget**2,
            "n_parameters": model.n_parameters(),
            "seed_data": int(cfg["data"]["seed_data"]),
            "iterations": len(history),
        }
    )
    with torch.no_grad():
        for split in ("train", "test"):
            logger.info("EVALUATION_STARTED | split=%s", split)
            predictions = model(tensors[f"{split}_x"])
            labels = tensors[f"{split}_y"]
            metrics[f"{split}_accuracy"] = vqc.reference_accuracy(predictions, labels)
            metrics[f"{split}_mse"] = float(((predictions - labels) ** 2).mean())
            logger.info(
                "EVALUATION_COMPLETED | split=%s | accuracy=%.4f | mse=%.6f",
                split,
                metrics[f"{split}_accuracy"],
                metrics[f"{split}_mse"],
            )
    _save_state(run_dir / "model_state.pt", model)
    return metrics


def _classification_setup(cfg: dict, run_dir: Path) -> dict:
    tensors = {
        k: v.to(settings.device(cfg)) for k, v in _classification_tensors(cfg).items()
    }
    _save_dataset(run_dir, tensors)
    logger.info(
        "DATASET_READY | train=%d | test=%d | train_label1_fraction=%.3f",
        len(tensors["train_x"]),
        len(tensors["test_x"]),
        float(tensors["train_y"].mean()),
    )
    return tensors


def run_vqc(cfg: dict, run_dir: Path, generator: torch.Generator) -> dict:
    """E4: train one (virtual) quantum classifier full-batch and evaluate it."""
    vqc_cfg = cfg["vqc"]
    tensors = _classification_setup(cfg, run_dir)
    model = vqc.VirtualQuantumClassifier(
        n_qubits=int(vqc_cfg["n_qubits"]),
        depth=int(vqc_cfg["depth"]),
        budget=float(vqc_cfg["budget"]),
        gate_noise=tuple(vqc_cfg["gate_noise"]),
        generator=generator,
        complex_dtype=settings.complex_dtype(cfg),
    ).to(settings.device(cfg))
    metrics = {
        "task": "vqc",
        "depth": int(vqc_cfg["depth"]),
        "n_qubits": int(vqc_cfg["n_qubits"]),
        "gate_noise": list(vqc_cfg["gate_noise"]),
    }
    return _fit_and_evaluate_classifier(cfg, run_dir, model, tensors, metrics)


def run_vqc_merlin(cfg: dict, run_dir: Path, generator: torch.Generator) -> dict:
    """Phase 4: photonic (MerLin) adaptation of the virtual classifier (not the original)."""
    merlin_cfg = cfg["vqc_merlin"]
    tensors = _classification_setup(cfg, run_dir)
    # MerLin initializes its phases from the global torch RNG; seed it explicitly
    # so every repetition is reproducible from cfg["seed"].
    torch.manual_seed(int(cfg["seed"]))
    model = vqc_merlin.PhotonicVirtualClassifier(
        budget=float(merlin_cfg["budget"]),
        n_modes=int(merlin_cfg["n_modes"]),
        input_state=list(merlin_cfg["input_state"]),
        encoding_modes=list(merlin_cfg["encoding_modes"]),
        encoding_scale=float(merlin_cfg["encoding_scale"]),
        computation_space=merlin_cfg["computation_space"],
        indistinguishability=merlin_cfg["indistinguishability"],
        dtype=settings.real_dtype(cfg),
        device=settings.device(cfg),
    )
    hardware = {
        "adaptation": "MerLin photonic adaptation (not the original method)",
        "computation_space": merlin_cfg["computation_space"],
        "detector_model": "photon-number-resolving"
        if merlin_cfg["computation_space"] == "fock"
        else "threshold",
        "n_photons": int(sum(merlin_cfg["input_state"])),
        "n_modes": int(merlin_cfg["n_modes"]),
        "input_state": list(merlin_cfg["input_state"]),
        "encoding": f"phase (angle) encoding of (x1, x2) on modes {merlin_cfg['encoding_modes']}, scale={merlin_cfg['encoding_scale']}",
        "measurement_strategy": "MeasurementStrategy.probs -> first-half photon-number parity (-1)^(n0+n1)",
        "postselection": "none",
        "noise_model": "none"
        if merlin_cfg["indistinguishability"] is None
        else f"pcvl.NoiseModel(indistinguishability={merlin_cfg['indistinguishability']})",
        "simulator": "MerLin exact CPU simulation (shots=0)",
        "gradients": "exact (autograd through MerLin simulation)",
    }
    logger.info(
        "PHOTONIC_SETUP | %s", " | ".join(f"{k}={v}" for k, v in hardware.items())
    )
    metrics = {"task": "vqc_merlin", "hardware": hardware}
    return _fit_and_evaluate_classifier(cfg, run_dir, model, tensors, metrics)


def _fit_classical(
    kind: str,
    x: torch.Tensor,
    y: torch.Tensor,
    lr: float,
    iterations: int,
    generator: torch.Generator,
    log_progress: bool,
) -> nn.Module:
    model = classical.build_classical_model(kind, generator, x.dtype)
    training_cfg = {
        "iterations": iterations,
        "lr": lr,
        "scheduler_factor": None,
        "scheduler_patience": None,
    }
    loss_function = nn.BCEWithLogitsLoss()

    def loss_fn():
        logits = model(x).squeeze(-1)
        loss = loss_function(logits, y)
        return loss, {
            "train_accuracy": classical.accuracy_from_logits(logits.detach(), y)
        }

    if log_progress:
        optimize(model.parameters(), loss_fn, training_cfg)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        for _ in range(iterations):
            optimizer.zero_grad()
            loss, _ = loss_fn()
            loss.backward()
            optimizer.step()
    return model


def run_classical_classifier(
    cfg: dict, run_dir: Path, generator: torch.Generator
) -> dict:
    """E4 classical baseline: select the learning rate on validation, refit, evaluate."""
    classical_cfg = cfg["classical"]
    tensors = _classification_tensors(cfg)
    _save_dataset(run_dir, tensors)
    logger.info(
        "DATASET_READY | train=%d | test=%d",
        len(tensors["train_x"]),
        len(tensors["test_x"]),
    )
    kind = classical_cfg["kind"]
    iterations = int(cfg["training"]["iterations"])
    fit_x, fit_y, val_x, val_y = classical.split_train_validation(
        tensors["train_x"],
        tensors["train_y"],
        float(classical_cfg["validation_fraction"]),
        generator,
    )
    selection_scores = {}
    for lr in classical_cfg["lr_grid"]:
        model = _fit_classical(
            kind, fit_x, fit_y, float(lr), iterations, generator, log_progress=False
        )
        with torch.no_grad():
            score = classical.accuracy_from_logits(model(val_x).squeeze(-1), val_y)
        selection_scores[str(lr)] = score
        logger.info(
            "SELECTION_CANDIDATE_COMPLETED | lr=%s | validation_accuracy=%.4f",
            lr,
            score,
        )
    # Ties resolve to the first learning rate in the configured grid.
    selected_lr = float(
        max(classical_cfg["lr_grid"], key=lambda lr: selection_scores[str(lr)])
    )
    logger.info("SELECTION_COMPLETED | selected_lr=%s", selected_lr)
    model = _fit_classical(
        kind,
        tensors["train_x"],
        tensors["train_y"],
        selected_lr,
        iterations,
        generator,
        log_progress=True,
    )
    metrics = {
        "task": "classical_classifier",
        "kind": kind,
        "selected_lr": selected_lr,
        "validation_scores": selection_scores,
        "n_parameters": sum(p.numel() for p in model.parameters()),
        "seed_data": int(cfg["data"]["seed_data"]),
        "iterations": iterations,
    }
    with torch.no_grad():
        for split in ("train", "test"):
            logger.info("EVALUATION_STARTED | split=%s", split)
            logits = model(tensors[f"{split}_x"]).squeeze(-1)
            metrics[f"{split}_accuracy"] = classical.accuracy_from_logits(
                logits, tensors[f"{split}_y"]
            )
            logger.info(
                "EVALUATION_COMPLETED | split=%s | accuracy=%.4f",
                split,
                metrics[f"{split}_accuracy"],
            )
    return metrics


def run_vvqe(cfg: dict, run_dir: Path, generator: torch.Generator) -> dict:
    """E5: train one VVQE (or VQE at budget 1) at one bond length and evaluate it."""
    vvqe_cfg = cfg["vvqe"]
    device = settings.device(cfg)
    complex_dtype = settings.complex_dtype(cfg)
    molecule = data.heh_tapered_hamiltonian(float(vvqe_cfg["half_distance_bohr"]))
    hamiltonian = torch.as_tensor(
        molecule["matrix"], dtype=complex_dtype, device=device
    )
    logger.info(
        "DATASET_READY | half_distance_bohr=%.4f | separation_angstrom=%.4f | fci=%.8f | hf=%.8f",
        vvqe_cfg["half_distance_bohr"],
        molecule["separation_angstrom"],
        molecule["fci_energy"],
        molecule["hf_energy"],
    )
    model = vvqe.VirtualVQE(
        n_qubits=int(vvqe_cfg["n_qubits"]),
        depth=int(vvqe_cfg["depth"]),
        budget=float(vvqe_cfg["budget"]),
        gate_noise=tuple(vvqe_cfg["gate_noise"]),
        generator=generator,
        complex_dtype=complex_dtype,
    ).to(device)
    coefficients = vvqe.negativity_chebyshev_coefficients(
        int(vvqe_cfg["chebyshev_order"])
    )
    penalty = float(vvqe_cfg["penalty"])

    def loss_fn():
        terms = vvqe.vvqe_terms(
            model.virtual_state(device),
            hamiltonian,
            model.budget,
            penalty,
            coefficients,
        )
        return terms["loss"], {
            "energy": float(terms["energy"].detach()),
            "negativity_proxy": float(terms["negativity_proxy"].detach()),
        }

    history = optimize(model.parameters(), loss_fn, cfg["training"])
    _write_history(run_dir / "losses.csv", history)

    logger.info("EVALUATION_STARTED | split=n/a | state=final parameters")
    with torch.no_grad():
        sigma = model.virtual_state(device)
        terms = vvqe.vvqe_terms(sigma, hamiltonian, model.budget, penalty, coefficients)
    negativity, min_eigenvalue = vvqe.exact_negativity(sigma)
    energy = float(terms["energy"])
    np.savez(run_dir / "virtual_state.npz", sigma=sigma.cpu().numpy())
    logger.info("ARTIFACT_WRITTEN | path=virtual_state.npz")
    _save_state(run_dir / "model_state.pt", model)
    metrics = {
        "task": "vvqe",
        "budget": model.budget,
        "coefficients": [model.c1, model.c2],
        "shot_multiplier": model.budget**2,
        "half_distance_bohr": float(vvqe_cfg["half_distance_bohr"]),
        "separation_bohr": molecule["separation_bohr"],
        "separation_angstrom": molecule["separation_angstrom"],
        "energy": energy,
        "fci_energy": molecule["fci_energy"],
        "hf_energy": molecule["hf_energy"],
        "abs_error_fci": abs(energy - molecule["fci_energy"]),
        "abs_error_hf": abs(energy - molecule["hf_energy"]),
        "below_fci": energy < molecule["fci_energy"],
        "negativity_proxy": float(terms["negativity_proxy"]),
        "exact_negativity": negativity,
        "min_eigenvalue_sigma": min_eigenvalue,
        "trace_sigma": float(torch.real(torch.trace(sigma))),
        "final_loss": float(terms["loss"]),
        "penalty": penalty,
        "chebyshev_order": int(vvqe_cfg["chebyshev_order"]),
        "depth": int(vvqe_cfg["depth"]),
        "gate_noise": list(vvqe_cfg["gate_noise"]),
        "n_parameters": model.n_parameters(),
        "iterations": len(history),
    }
    logger.info(
        "EVALUATION_COMPLETED | energy=%.8f | fci=%.8f | abs_error_fci=%.6f | min_eig_sigma=%.3g",
        energy,
        molecule["fci_energy"],
        metrics["abs_error_fci"],
        min_eigenvalue,
    )
    return metrics


def _completed_runs(sweep_dir: Path) -> list[tuple[dict, Path]]:
    """``(run_entry, run_dir)`` for every COMPLETED run of a validated sweep.

    Raises
    ------
    ValueError
        If the sweep is not COMPLETED (partial evidence is not used here).
    """
    status = json.loads((sweep_dir / "sweep_status.json").read_text())
    if status["status"] != "COMPLETED":
        raise ValueError(
            f"{sweep_dir} is {status['status']}; equal-shot evaluation needs a complete sweep"
        )
    return [
        (run, sweep_dir / run["run_dir"])
        for run in status["runs"]
        if run["status"] == "COMPLETED"
    ]


def _equal_shot_e5(cfg: dict, run_dir: Path) -> dict:
    eq = cfg["equal_shot"]
    grid = [float(s) if s != "inf" else math.inf for s in eq["shot_grid"]]
    runs = _completed_runs(Path(eq["sweeps"]["vvqe"]))
    logger.info(
        "DATASET_READY | source=%s | runs=%d | shot_grid=%s",
        eq["sweeps"]["vvqe"],
        len(runs),
        eq["shot_grid"],
    )
    rows = []
    logger.info(
        "EVALUATION_STARTED | split=n/a | experiment=e5 | estimator=quasi-probability Eq. 6"
    )
    for run, path in runs:
        metrics = json.loads((path / "metrics.json").read_text())
        sigma = np.load(path / "virtual_state.npz")["sigma"]
        molecule = data.heh_tapered_hamiltonian(float(metrics["half_distance_bohr"]))
        for shots in grid:
            estimate = shot_budget.energy_estimator_rmse(
                molecule["matrix"],
                sigma,
                metrics["budget"],
                shots,
                molecule["fci_energy"],
            )
            rows.append(
                {
                    "run_id": run["run_id"],
                    "seed": run["seed"],
                    "half_distance": metrics["half_distance_bohr"],
                    "budget": metrics["budget"],
                    "shots": "inf" if math.isinf(shots) else shots,
                    **estimate,
                }
            )
    with (run_dir / "equal_shot_e5_rows.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    logger.info("ARTIFACT_WRITTEN | path=equal_shot_e5_rows.csv | rows=%d", len(rows))
    summary = {}
    for row in rows:
        key = f"d={row['half_distance']}|gamma={row['budget']}|S={row['shots']}"
        summary.setdefault(key, []).append(row["rmse"])
    aggregated = {
        k: {"rmse_mean": float(np.mean(v)), "rmse_std": float(np.std(v)), "n": len(v)}
        for k, v in summary.items()
    }
    distances = sorted({r["half_distance"] for r in rows})
    decisions = {}
    for budget in sorted({r["budget"] for r in rows if r["budget"] > 1.0}):
        for shots in eq["shot_grid"]:
            s_key = shots if shots == "inf" else float(shots)
            wins = 0
            for d in distances:
                virtual = aggregated[f"d={d}|gamma={budget}|S={s_key}"]["rmse_mean"]
                conventional = aggregated[f"d={d}|gamma=1.0|S={s_key}"]["rmse_mean"]
                wins += virtual < conventional - float(eq["tie_tolerance_ha"])
            decisions[f"gamma={budget}|S={shots}"] = {
                "distances_virtual_better": wins,
                "n_distances": len(distances),
            }
            logger.info(
                "EVALUATION_COMPLETED | gamma=%s | shots=%s | virtual_better_at=%d/%d distances",
                budget,
                shots,
                wins,
                len(distances),
            )
    return {
        "task": "equal_shot_eval",
        "experiment": "e5",
        "n_models": len(runs),
        "shot_grid": eq["shot_grid"],
        "aggregated_rmse": aggregated,
        "decisions": decisions,
        "rows_path": "equal_shot_e5_rows.csv",
    }


def _branch_expectations(
    kind: str, path: Path
) -> tuple[np.ndarray, np.ndarray, list[float], int]:
    """Per-test-point branch expectations ``z`` (n_branches, n_test) of a trained classifier."""
    snapshot = json.loads((path / "config_snapshot.json").read_text())
    dataset = np.load(path / "dataset.npz")
    real = torch.float64
    if kind == "gate":
        section = snapshot["vqc"]
        model = vqc.VirtualQuantumClassifier(
            n_qubits=int(section["n_qubits"]),
            depth=int(section["depth"]),
            budget=float(section["budget"]),
            gate_noise=tuple(section["gate_noise"]),
            generator=torch.Generator().manual_seed(0),
            complex_dtype=torch.complex128,
        )
        model.load_state_dict(torch.load(path / "model_state.pt"))
        x = torch.as_tensor(dataset["test_x"], dtype=real)
        with torch.no_grad():
            rho = vqc.encode_points(x, model.n_qubits, model.complex_dtype)
            z = [
                torch.real(
                    torch.einsum("bij,ji->b", branch(rho), model.observable)
                ).numpy()
                for branch in model.branches
            ]
    else:
        section = snapshot["vqc_merlin"]
        model = vqc_merlin.PhotonicVirtualClassifier(
            budget=float(section["budget"]),
            n_modes=int(section["n_modes"]),
            input_state=list(section["input_state"]),
            encoding_modes=list(section["encoding_modes"]),
            encoding_scale=float(section["encoding_scale"]),
            computation_space=section["computation_space"],
            indistinguishability=section["indistinguishability"],
            dtype=real,
            device=torch.device("cpu"),
        )
        model.load_state_dict(torch.load(path / "model_state.pt"))
        with torch.no_grad():
            z = [
                t.numpy()
                for t in model.branch_readouts(
                    torch.as_tensor(dataset["test_x"], dtype=real)
                )
            ]
    coefficients = [model.c1, model.c2][: len(z)]
    return np.stack(z), dataset["test_y"], coefficients, model.n_parameters()


def _equal_shot_e4(cfg: dict, run_dir: Path) -> dict:
    eq = cfg["equal_shot"]
    grid = [s if s == "inf" else int(s) for s in eq["shot_grid"]]
    sources = eq["sweeps"]
    models = []
    for label, spec in sources.items():
        for run, path in _completed_runs(Path(spec["path"])):
            if spec.get("filter") and not all(
                run["grid_point"].get(k) == v for k, v in spec["filter"].items()
            ):
                continue
            budget_value = (
                run["grid_point"].get(spec["budget_axis"])
                if spec.get("budget_axis")
                else 1.0
            )
            models.append((f"{label}|gamma={budget_value}", run, path, spec["kind"]))
    logger.info(
        "DATASET_READY | trained classifiers=%d | test points per model=100 | shot_grid=%s",
        len(models),
        eq["shot_grid"],
    )
    logger.info(
        "EVALUATION_STARTED | split=test | experiment=e4 | estimator=quasi-probability Eq. 6, exact binomial"
    )
    rows = []
    for name, run, path, kind in models:
        z, labels, coefficients, n_params = _branch_expectations(kind, path)
        budget = sum(abs(c) for c in coefficients)
        mu = sum(c * zi for c, zi in zip(coefficients, z))
        for shots in grid:
            accuracy = shot_budget.expected_accuracy(
                mu, labels, budget, math.inf if shots == "inf" else shots
            )
            rows.append(
                {
                    "model": name,
                    "run_id": run["run_id"],
                    "repetition": run["repetition"],
                    "budget": budget,
                    "n_parameters": n_params,
                    "shots": shots,
                    "expected_test_accuracy": accuracy,
                }
            )
    with (run_dir / "equal_shot_e4_rows.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    logger.info("ARTIFACT_WRITTEN | path=equal_shot_e4_rows.csv | rows=%d", len(rows))
    table = {}
    for row in rows:
        table.setdefault(f"{row['model']}|S={row['shots']}", {})[row["repetition"]] = (
            row["expected_test_accuracy"]
        )
    aggregated = {
        k: {
            "mean": float(np.mean(list(v.values()))),
            "std": float(np.std(list(v.values()))),
            "n": len(v),
        }
        for k, v in table.items()
    }
    tests = {}
    for comparison in eq["paired_comparisons"]:
        for shots in eq["shot_grid"]:
            a, b = (
                table[f"{comparison[0]}|S={shots}"],
                table[f"{comparison[1]}|S={shots}"],
            )
            keys = sorted(set(a) & set(b))
            diff = np.array([a[k] - b[k] for k in keys])
            result = (
                stats.ttest_rel([a[k] for k in keys], [b[k] for k in keys])
                if np.any(diff != 0)
                else None
            )
            entry = {
                "mean_difference": float(diff.mean()),
                "std_difference": float(diff.std(ddof=1)),
                "n_pairs": len(keys),
                "t_statistic": float(result.statistic) if result else 0.0,
                "p_value_two_sided": float(result.pvalue) if result else 1.0,
            }
            entry["advantage_at_equal_shots"] = (
                entry["mean_difference"] > 0 and entry["p_value_two_sided"] < 0.05
            )
            tests[f"{comparison[0]} - {comparison[1]}|S={shots}"] = entry
            logger.info(
                "EVALUATION_COMPLETED | %s - %s | S=%s | diff=%+.4f | p=%.3g",
                comparison[0],
                comparison[1],
                shots,
                entry["mean_difference"],
                entry["p_value_two_sided"],
            )
    return {
        "task": "equal_shot_eval",
        "experiment": "e4",
        "n_models": len(models),
        "shot_grid": eq["shot_grid"],
        "aggregated_accuracy": aggregated,
        "paired_tests": tests,
        "rows_path": "equal_shot_e4_rows.csv",
    }


def run_equal_shot_eval(cfg: dict, run_dir: Path, generator: torch.Generator) -> dict:
    """Phase 5: evaluate trained Phase 3/4 models at equal measurement-shot budgets."""
    experiment = cfg["equal_shot"]["experiment"]
    if experiment == "e5":
        return _equal_shot_e5(cfg, run_dir)
    if experiment == "e4":
        return _equal_shot_e4(cfg, run_dir)
    raise ValueError(
        f"Unknown equal_shot.experiment {experiment!r}; expected 'e4' or 'e5'"
    )


TASKS = {
    "min_overhead_sdp": run_min_overhead_sdp,
    "vqem": run_vqem,
    "vqc": run_vqc,
    "vqc_merlin": run_vqc_merlin,
    "classical_classifier": run_classical_classifier,
    "vvqe": run_vvqe,
    "equal_shot_eval": run_equal_shot_eval,
}
