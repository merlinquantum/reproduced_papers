# ruff: noqa: N999
"""Experiment runner for the BAS dataset using the MerLin implementation."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from photonic_QCNN.lib.data import (
    convert_dataset_to_tensor,
    convert_tensor_to_loader,
    get_dataset,
)
from photonic_QCNN.lib.src.merlin_pqcnn import HybridModel
from photonic_QCNN.lib.training.train_model import train_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"

DEFAULT_BAS_CONFIG: dict[str, Any] = {
    "n_runs": 5,
    "data_source": "paper",
    "conv_circuit": "BS",
    "dense_circuit": "BS",
    "measure_subset": 2,
    "dense_added_modes": 2,
    "output_proba_type": "mode",
    "output_formatting": "Mod_grouping",
    "random_states": [42, 123, 456, 789, 999],
    "batch_size": 6,
    "results_subdir": "BAS",
    "outdir": str(DEFAULT_RESULTS_DIR),
    "seed": 42,
    "device": "cpu",
}


def _resolve_base_outdir(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _prepare_config(overrides: dict[str, Any] | None) -> dict[str, Any]:
    cfg = copy.deepcopy(DEFAULT_BAS_CONFIG)
    if overrides:
        for key, value in overrides.items():
            if value is not None:
                cfg[key] = value
    return cfg


def _prepare_random_states(cfg: dict[str, Any]) -> np.ndarray:
    n_runs = int(cfg.get("n_runs", 1))
    random_states = list(cfg.get("random_states", []))
    if len(random_states) < n_runs:
        rng = np.random.default_rng(cfg.get("seed", 42))
        needed = n_runs - len(random_states)
        random_states.extend(rng.integers(0, 10_000, size=needed).tolist())
    return np.array(random_states[:n_runs])


def _prepare_output_dir(base_outdir: Path, dataset_name: str) -> Path:
    run_dir = base_outdir / dataset_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def run_experiment(
    random_state: int,
    run_id: int,
    source: str,
    conv_circuit: str,
    dense_circuit: str,
    measure_subset: int,
    dense_added_modes: int,
    output_proba_type: str,
    output_formatting: str,
    batch_size: int,
    training_params: dict[str, float] | None,
    device: str | None,
):
    """Run the complete experiment for one random state on BAS."""
    print(
        f"Running experiment {run_id} with random state {random_state} on BAS "
        f"(device={device or 'cpu'})"
    )

    # Set random seeds
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    # Load dataset
    print("Loading datasets")
    if source == "paper":
        x_train, x_test, y_train, y_test = get_dataset("BAS", "paper", random_state)
    elif source == "scratch":
        x_train, x_test, y_train, y_test = get_dataset("BAS", "scratch", random_state)
    else:
        raise ValueError(f"Unknown dataset source: {source}")

    x_train, x_test, y_train, y_test = convert_dataset_to_tensor(
        x_train, x_test, y_train, y_test
    )
    train_loader = convert_tensor_to_loader(x_train, y_train, batch_size=batch_size)

    # Train each dataset
    print("Training BAS...")

    # Create model
    dims = (4, 4)

    device_obj = torch.device(device or "cpu")

    model = HybridModel(
        dims,
        conv_circuit,
        dense_circuit,
        measure_subset,
        dense_added_modes,
        output_proba_type,
        output_formatting,
    ).to(device_obj)
    # Count number of parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {num_params} trainable parameters:")
    for i, layer in enumerate(model.qcnn):
        num_params = sum(p.numel() for p in layer.parameters())
        print(f"Layer {i} ({layer.__class__.__name__}): {num_params} parameters")
    print(f"Output of circuit has size {model.qcnn_output_dim}")

    # Train model
    training_results = train_model(
        model,
        train_loader,
        x_train,
        x_test,
        y_train,
        y_test,
        training_params=training_params,
    )

    print(
        "BAS - Final train: "
        f"{training_results['final_train_acc']:.4f}, "
        f"test: {training_results['final_test_acc']:.4f}"
    )
    return training_results


def save_results(
    all_results: dict[str, dict[str, float]],
    output_dir: Path,
    source: str,
    conv_circuit: str,
    dense_circuit: str,
    measure_subset: int,
    dense_added_modes: int,
    output_proba_type: str,
    output_formatting: str,
    dataset_name: str,
) -> dict[str, float]:
    """Save results to JSON file and return summary statistics."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save detailed results
    results_file = output_dir / "detailed_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    # Save summary statistics
    num_runs = len(all_results)
    train_accs = [all_results[f"run_{i}"]["final_train_acc"] for i in range(num_runs)]
    test_accs = [all_results[f"run_{i}"]["final_test_acc"] for i in range(num_runs)]

    summary = {
        "train_acc_mean": float(np.mean(train_accs)),
        "train_acc_std": float(np.std(train_accs)),
        "test_acc_mean": float(np.mean(test_accs)),
        "test_acc_std": float(np.std(test_accs)),
        "train_accs": train_accs,
        "test_accs": test_accs,
    }

    summary_file = output_dir / "summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Create training plots for each dataset
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    colors = ["blue", "red", "green", "orange", "purple"]

    # Plot loss history for this dataset
    ax_loss = axes[0]
    for run_idx in range(num_runs):
        loss_history = all_results[f"run_{run_idx}"]["loss_history"]
        color = colors[run_idx % len(colors)]
        ax_loss.plot(
            loss_history,
            color=color,
            alpha=0.7,
            linewidth=0.8,
            label=f"Run {run_idx + 1}",
        )
    ax_loss.set_title(f"{dataset_name} - Training Loss")
    ax_loss.set_xlabel("Training Steps")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)

    # Plot train accuracy for this dataset
    ax_train = axes[1]
    for run_idx in range(num_runs):
        train_acc_history = all_results[f"run_{run_idx}"]["train_acc_history"]
        epochs = range(len(train_acc_history))
        color = colors[run_idx % len(colors)]
        ax_train.plot(
            epochs,
            train_acc_history,
            color=color,
            alpha=0.7,
            linewidth=0.8,
            label=f"Run {run_idx + 1}",
        )
    ax_train.set_title(f"{dataset_name} - Training Accuracy")
    ax_train.set_xlabel("Epochs")
    ax_train.set_ylabel("Accuracy")
    ax_train.legend()
    ax_train.grid(True, alpha=0.3)
    ax_train.set_ylim(0, 1)

    # Plot test accuracy for this dataset
    ax_test = axes[2]
    for run_idx in range(num_runs):
        test_acc_history = all_results[f"run_{run_idx}"]["test_acc_history"]
        epochs = range(len(test_acc_history))
        color = colors[run_idx % len(colors)]
        ax_test.plot(
            epochs,
            test_acc_history,
            color=color,
            alpha=0.7,
            linewidth=0.8,
            label=f"Run {run_idx + 1}",
        )
    ax_test.set_title(f"{dataset_name} - Test Accuracy")
    ax_test.set_xlabel("Epochs")
    ax_test.set_ylabel("Accuracy")
    ax_test.legend()
    ax_test.grid(True, alpha=0.3)
    ax_test.set_ylim(0, 1)

    plt.tight_layout()
    plots_file = output_dir / f"{dataset_name}_training_plots.png"
    plt.savefig(plots_file, dpi=300, bbox_inches="tight")
    plt.close()

    path_for_args = output_dir / "args.txt"
    infos = {
        "data_source": source,
        "conv_circuit": conv_circuit,
        "dense_circuit": dense_circuit,
        "measure_subset": measure_subset,
        "dense_added_modes": dense_added_modes,
        "output_proba_type": output_proba_type,
        "output_formatting": output_formatting,
    }
    with open(path_for_args, "w", encoding="utf-8") as f:
        for key, value in infos.items():
            f.write(f"{key} = {value}\n")

    print(f"\nResults saved to {output_dir}")
    print(f"Training plots saved to {plots_file}")

    # Print summary
    print("\nSummary Results:")
    print("=" * 50)
    print(f"{dataset_name}:")
    print(
        f"  Train Accuracy: {summary['train_acc_mean']:.3f} \u00b1 {summary['train_acc_std']:.3f}"
    )
    print(
        f"  Test Accuracy:  {summary['test_acc_mean']:.3f} \u00b1 {summary['test_acc_std']:.3f}"
    )

    return summary


def run_bas_experiments(
    config: dict[str, Any] | None = None,
    training_params: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Entry point used by implementation.py and standalone CLI."""
    cfg = _prepare_config(config)
    base_outdir = _resolve_base_outdir(cfg.get("outdir", str(DEFAULT_RESULTS_DIR)))
    results_subdir = cfg.get("results_subdir", "BAS")
    dataset_display_name = "BAS"
    output_dir = _prepare_output_dir(base_outdir, results_subdir)

    random_states = _prepare_random_states(cfg)
    n_runs = len(random_states)
    if n_runs == 0:
        raise ValueError("At least one random state is required to run experiments.")

    print("Starting Photonic QCNN experiments...")
    print(f"Results will be saved to: {output_dir}")

    all_results: dict[str, dict[str, float]] = {}

    for i, random_state in enumerate(random_states):
        print(f"About to start experiment {i + 1}/{n_runs}")
        results = run_experiment(
            random_state,
            i,
            cfg["data_source"],
            cfg["conv_circuit"],
            cfg["dense_circuit"],
            cfg["measure_subset"],
            cfg["dense_added_modes"],
            cfg["output_proba_type"],
            cfg["output_formatting"],
            batch_size=int(cfg["batch_size"]),
            training_params=training_params,
            device=cfg.get("device"),
        )
        print(f"Experiment {i + 1}/{n_runs} completed")
        all_results[f"run_{i}"] = results

    summary = save_results(
        all_results,
        output_dir,
        cfg["data_source"],
        cfg["conv_circuit"],
        cfg["dense_circuit"],
        cfg["measure_subset"],
        cfg["dense_added_modes"],
        cfg["output_proba_type"],
        cfg["output_formatting"],
        dataset_display_name,
    )

    return {
        "dataset": dataset_display_name,
        "results_subdir": results_subdir,
        "summary": summary,
        "output_dir": str(output_dir),
    }


if __name__ == "__main__":  # pragma: no cover
    run_bas_experiments()
