from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from lib.approx_kernel import (
    build_quantum_model,
    classical_features,
    sample_random_features,
    transform_inputs,
)

from .data import target_function


@dataclass
class QuantumResult:
    accuracy: float
    kernel_train: np.ndarray
    kernel_test: np.ndarray
    decision_data: dict
    losses: dict[str, list[float]] | None


def _resolve_scaling(value, r: int) -> float:
    if isinstance(value, str):
        if value == "1/sqrt(R)":
            return 1.0 / math.sqrt(r)
        if value == "sqrt(R)":
            return math.sqrt(r)
        if value == "sqrt(R)+3":
            return math.sqrt(r) + 3
        raise ValueError(f"Unknown z_q_matrix_scaling: {value}")
    return float(value)


def _hybrid_training_data(train_proj: np.ndarray, test_proj: np.ndarray, cfg: dict):
    mode = cfg.get("hybrid_model_data", "default").lower()
    if mode == "default":
        return train_proj, test_proj
    if mode == "generated":
        train_min = train_proj.min(axis=0)
        train_max = train_proj.max(axis=0)
        test_min = test_proj.min(axis=0)
        test_max = test_proj.max(axis=0)
        synthetic_train = np.linspace(train_min, train_max, 540, axis=0)
        synthetic_test = np.linspace(test_min, test_max, 100, axis=0)
        return synthetic_train, synthetic_test
    raise ValueError(f"Unknown hybrid_model_data: {mode}")


def _train_quantum_model(
    projections_train: np.ndarray,
    projections_test: np.ndarray,
    cfg: dict,
) -> tuple[torch.nn.Module, dict[str, list[float]]]:
    model = build_quantum_model(cfg)
    dataset = TensorDataset(
        torch.tensor(projections_train, dtype=torch.float32),
        torch.tensor(target_function(projections_train), dtype=torch.float32),
    )
    dataloader = DataLoader(dataset, batch_size=cfg.get("batch_size", 32), shuffle=True)
    criterion = torch.nn.MSELoss()
    opt_name = cfg.get("optimizer", "adam").lower()
    lr = cfg.get("learning_rate", 0.01)
    weight_decay = cfg.get("weight_decay", 0.0)

    if opt_name == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            betas=cfg.get("betas", (0.9, 0.999)),
            weight_decay=weight_decay,
        )
    elif opt_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
    elif opt_name == "adagrad":
        optimizer = torch.optim.Adagrad(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")

    epochs = int(cfg.get("num_epochs", 200))
    losses = {"train": [], "test": []}
    best_state = None
    best_test = float("inf")

    for _ in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch_x, batch_y in dataloader:
            if len(batch_x) < cfg.get("batch_size", 32):
                continue
            optimizer.zero_grad()
            input_tensor = batch_x.view(-1, 1) * float(
                cfg.get("pre_encoding_scaling", 1.0 / math.pi)
            )
            preds = model(input_tensor)
            preds = preds.view(batch_x.size(0), -1)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= max(1, len(dataloader))
        losses["train"].append(epoch_loss)

        model.eval()
        with torch.no_grad():
            eval_input = torch.tensor(projections_test, dtype=torch.float32).view(-1, 1)
            eval_input *= float(cfg.get("pre_encoding_scaling", 1.0 / math.pi))
            preds = model(eval_input).view(len(projections_test), -1)
            test_loss = criterion(
                preds,
                torch.tensor(target_function(projections_test), dtype=torch.float32),
            )
            losses["test"].append(float(test_loss))
            if test_loss < best_test:
                best_test = float(test_loss)
                best_state = {
                    k: v.detach().clone() for k, v in model.state_dict().items()
                }

    if best_state:
        model.load_state_dict(best_state)
    return model, losses


def _quantum_features(
    x_train: np.ndarray,
    x_test: np.ndarray,
    params: dict,
    cfg: dict,
) -> QuantumResult:
    w, b = params["w"], params["b"]
    r = params["r"]
    gamma = params["gamma"]
    projections_train = transform_inputs(x_train, w, b, r, gamma)
    projections_test = transform_inputs(x_test, w, b, r, gamma)

    if cfg.get("train_hybrid_model", False):
        synth_train, synth_test = _hybrid_training_data(
            projections_train, projections_test, cfg
        )
        model, losses = _train_quantum_model(
            synth_train,
            synth_test,
            cfg,
        )
    else:
        model = build_quantum_model(cfg)
        losses = None

    model.eval()
    with torch.no_grad():
        train_input = torch.tensor(projections_train, dtype=torch.float32).view(-1, 1)
        train_input *= float(cfg.get("pre_encoding_scaling", 1.0 / math.pi))
        z_train = model(train_input).view(len(projections_train), -1)

        test_input = torch.tensor(projections_test, dtype=torch.float32).view(-1, 1)
        test_input *= float(cfg.get("pre_encoding_scaling", 1.0 / math.pi))
        z_test = model(test_input).view(len(projections_test), -1)

    scale = _resolve_scaling(cfg.get("z_q_matrix_scaling", "1/sqrt(R)"), r)
    z_train = z_train.detach().numpy() * scale
    z_test = z_test.detach().numpy() * scale

    kernel_train = z_train @ z_train.T
    kernel_test = z_test @ z_train.T

    return QuantumResult(
        accuracy=0.0,
        kernel_train=kernel_train,
        kernel_test=kernel_test,
        decision_data={
            "model": model,
            "w": w,
            "b": b,
            "r": r,
            "gamma": gamma,
            "pre_scale": float(cfg.get("pre_encoding_scaling", 1.0 / math.pi)),
            "z_scale": scale,
            "x_train": x_train,
        },
        losses=losses,
    )


def _classical_features(
    x_train: np.ndarray, x_test: np.ndarray, params: dict
) -> tuple[np.ndarray, np.ndarray]:
    w, b = params["w"], params["b"]
    r = params["r"]
    gamma = params["gamma"]
    train_proj = transform_inputs(x_train, w, b, r, gamma)
    test_proj = transform_inputs(x_test, w, b, r, gamma)
    z_train = classical_features(train_proj)
    z_test = classical_features(test_proj)
    return z_train @ z_train.T, z_test @ z_train.T


def _train_svm(k_train, k_test, y_train, y_test, C) -> float:
    svc = SVC(C=C, kernel="precomputed")
    svc.fit(k_train, y_train)
    preds = svc.predict(k_test)
    return accuracy_score(y_test, preds)


def run_rks_experiments(
    dataset: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    sweep_cfg: dict,
    model_cfg: dict,
    training_cfg: dict,
    classifier_cfg: dict,
    base_seed: int,
) -> list[dict]:
    x_train, x_test, y_train, y_test = dataset
    results: list[dict] = []
    repeat = int(sweep_cfg.get("repeats", 1))
    seed = base_seed

    default_gammas = list(range(1, 11))
    r_values = sweep_cfg.get("r_values", [1])
    gamma_values = sweep_cfg.get("gamma_values", default_gammas)

    train_hybrid = bool(training_cfg.get("train_hybrid_model", False))
    if train_hybrid:
        print(
            "Hybrid quantum model training ENABLED — optimizing merlin layer for each (R, γ)."
        )
    else:
        print(
            "Hybrid quantum model training DISABLED — using frozen merlin layer for each (R, γ)."
        )

    total_combos = repeat * len(r_values) * len(gamma_values)
    progress = tqdm(total=total_combos, desc="Sweeping (R, γ)", unit="combo")

    for r in r_values:
        for rep in range(repeat):
            seed_value = seed + rep + r * 9973
            w, b = sample_random_features(r, seed_value)
            for gamma in gamma_values:
                params = {"w": w, "b": b, "r": r, "gamma": gamma}

                q_cfg = {**model_cfg, **training_cfg, "r": r, "gamma": gamma}
                quantum_result = _quantum_features(x_train, x_test, params, q_cfg)
                q_acc = _train_svm(
                    quantum_result.kernel_train,
                    quantum_result.kernel_test,
                    y_train,
                    y_test,
                    classifier_cfg.get("C", 5.0),
                )
                quantum_result.accuracy = q_acc

                classical_train, classical_test = _classical_features(
                    x_train, x_test, params
                )
                c_acc = _train_svm(
                    classical_train,
                    classical_test,
                    y_train,
                    y_test,
                    classifier_cfg.get("C", 5.0),
                )

                progress.set_postfix(
                    {
                        "R": r,
                        "γ": gamma,
                        "q_acc": f"{q_acc:.3f}",
                        "c_acc": f"{c_acc:.3f}",
                    }
                )
                progress.update(1)

                results.append(
                    {
                        "method": "quantum",
                        "r": r,
                        "gamma": gamma,
                        "repeat": rep,
                        "accuracy": q_acc,
                        "kernel_train": quantum_result.kernel_train,
                        "kernel_test": quantum_result.kernel_test,
                        "decision_data": quantum_result.decision_data,
                        "losses": quantum_result.losses,
                        "C": classifier_cfg.get("C", 5.0),
                    }
                )
                results.append(
                    {
                        "method": "classical",
                        "r": r,
                        "gamma": gamma,
                        "repeat": rep,
                        "accuracy": c_acc,
                        "kernel_train": classical_train,
                        "kernel_test": classical_test,
                        "decision_data": {
                            "w": w,
                            "b": b,
                            "r": r,
                            "gamma": gamma,
                            "x_train": x_train,
                        },
                        "losses": None,
                        "C": classifier_cfg.get("C", 5.0),
                    }
                )
    progress.close()
    return results
