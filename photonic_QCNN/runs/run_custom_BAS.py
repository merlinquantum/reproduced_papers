# ruff: noqa: N999
import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch

from photonic_QCNN.data.data import (
    convert_dataset_to_tensor,
    convert_tensor_to_loader,
    get_dataset,
)
from photonic_QCNN.src.merlin_pqcnn import HybridModel
from photonic_QCNN.training.train_model import train_model


def run_experiment(
    random_state,
    run_id,
    source,
    conv_circuit,
    dense_circuit,
    measure_subset,
    dense_added_modes,
    output_proba_type,
    output_formatting,
):
    """Run the complete experiment for one random state on Custom BAS"""
    print(f"Running experiment {run_id} with random state {random_state} on Custom BAS")

    # Set random seeds
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    # Load dataset
    print("Loading datasets")
    if source == "paper":
        x_train, x_test, y_train, y_test = get_dataset("Custom BAS", "paper", 42)
        x_train, x_test, y_train, y_test = convert_dataset_to_tensor(
            x_train, x_test, y_train, y_test
        )
        train_loader = convert_tensor_to_loader(x_train, y_train, batch_size=6)
    elif source == "scratch":
        x_train, x_test, y_train, y_test = get_dataset("Custom BAS", "scratch", 42)
        x_train, x_test, y_train, y_test = convert_dataset_to_tensor(
            x_train, x_test, y_train, y_test
        )
        train_loader = convert_tensor_to_loader(x_train, y_train, batch_size=6)
    else:
        raise ValueError(f"Unknown dataset source: {source}")

    # Train each dataset
    print("Training Custom BAS...")

    # Create model
    dims = (4, 4)

    model = HybridModel(
        dims,
        conv_circuit,
        dense_circuit,
        measure_subset,
        dense_added_modes,
        output_proba_type,
        output_formatting,
    )
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {num_params} trainable parameters")
    print(f"Output of circuit has size {model.qcnn_output_dim}")

    # Train model
    training_results = train_model(
        model, train_loader, x_train, x_test, y_train, y_test
    )

    print(
        f"Custom BAS - Final train: {training_results['final_train_acc']:.4f}, test: {training_results['final_test_acc']:.4f}"
    )
    return training_results


def save_results(
    all_results,
    output_dir,
    source,
    conv_circuit,
    dense_circuit,
    measure_subset,
    dense_added_modes,
    output_proba_type,
    output_formatting,
):
    """Save results to JSON file"""
    os.makedirs(output_dir, exist_ok=True)

    # Save detailed results
    results_file = os.path.join(output_dir, "detailed_results.json")
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)

    # Save summary statistics
    summary = {}
    num_runs = len(all_results)
    train_accs = [all_results[f"run_{i}"]["final_train_acc"] for i in range(num_runs)]
    test_accs = [all_results[f"run_{i}"]["final_test_acc"] for i in range(num_runs)]

    summary = {
        "train_acc_mean": np.mean(train_accs),
        "train_acc_std": np.std(train_accs),
        "test_acc_mean": np.mean(test_accs),
        "test_acc_std": np.std(test_accs),
        "train_accs": train_accs,
        "test_accs": test_accs,
    }

    summary_file = os.path.join(output_dir, "summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    # Create training plots for each dataset
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    colors = ["blue", "red", "green", "orange", "purple"]

    # Plot loss history for this dataset
    ax_loss = axes[0]
    for run_idx in range(num_runs):
        loss_history = all_results[f"run_{run_idx}"]["loss_history"]
        ax_loss.plot(
            loss_history,
            color=colors[run_idx],
            alpha=0.7,
            linewidth=0.8,
            label=f"Run {run_idx + 1}",
        )
    ax_loss.set_title("Custom BAS - Training Loss")
    ax_loss.set_xlabel("Training Steps")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)

    # Plot train accuracy for this dataset
    ax_train = axes[1]
    for run_idx in range(num_runs):
        train_acc_history = all_results[f"run_{run_idx}"]["train_acc_history"]
        epochs = range(len(train_acc_history))
        ax_train.plot(
            epochs,
            train_acc_history,
            color=colors[run_idx],
            alpha=0.7,
            linewidth=0.8,
            label=f"Run {run_idx + 1}",
        )
    ax_train.set_title("Custom BAS - Training Accuracy")
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
        ax_test.plot(
            epochs,
            test_acc_history,
            color=colors[run_idx],
            alpha=0.7,
            linewidth=0.8,
            label=f"Run {run_idx + 1}",
        )
    ax_test.set_title("Custom BAS - Test Accuracy")
    ax_test.set_xlabel("Epochs")
    ax_test.set_ylabel("Accuracy")
    ax_test.legend()
    ax_test.grid(True, alpha=0.3)
    ax_test.set_ylim(0, 1)

    plt.tight_layout()
    plots_file = os.path.join(output_dir, "custom_BAS_training_plots.png")
    plt.savefig(plots_file, dpi=300, bbox_inches="tight")
    plt.close()

    path_for_args = os.path.join(output_dir, "args.txt")
    infos = {
        "data_source": source,
        "conv_circuit": conv_circuit,
        "dense_circuit": dense_circuit,
        "measure_subset": measure_subset,
        "dense_added_modes": dense_added_modes,
        "output_proba_type": output_proba_type,
        "output_formatting": output_formatting,
    }
    with open(path_for_args, "w") as f:
        for key, value in infos.items():
            f.write(f"{key} = {value}\n")

    print(f"\nResults saved to {output_dir}")
    print(f"Training plots saved to {plots_file}")

    # Print summary
    print("\nSummary Results:")
    print("=" * 50)
    print("Custom BAS:")
    print(
        f"  Train Accuracy: {summary['train_acc_mean']:.3f} ± {summary['train_acc_std']:.3f}"
    )
    print(
        f"  Test Accuracy:  {summary['test_acc_mean']:.3f} ± {summary['test_acc_std']:.3f}"
    )


def main():
    """Main execution function"""
    # Hyperparameters
    n_runs = 5  # int from 1 to 5
    data_source = "paper"  # ['scratch', 'paper']
    conv_circuit = "BS"  # ['MZI', 'BS', 'BS_random_PS']
    dense_circuit = "BS"  # ['MZI', 'BS', 'BS_random_PS']
    measure_subset = 2
    dense_added_modes = 2
    output_proba_type = "mode"  # ['state', 'mode']  MerLin default is 'state'
    output_formatting = "Mod_grouping"  # ['Train_linear', 'No_train_linear', 'Lex_grouping', 'Mod_grouping']

    random_states = [42, 123, 456, 789, 999]

    # Create output directory with current date and time
    now = datetime.now()
    day = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H-%M-%S")
    output_dir = f"../results/custom_BAS/{day}/{time_str}"

    print("Starting Photonic QCNN experiments...")
    print(f"Results will be saved to: {output_dir}")

    # Run experiments with different random states
    random_states = [42, 123, 456, 789, 999]
    assert 0 < n_runs < 6, "Number of runs must be between 1 and 5"
    random_states = random_states[:n_runs]
    all_results = {}

    for i, random_state in enumerate(random_states):
        print(f"About to start experiment {i + 1}/{n_runs}")
        results = run_experiment(
            random_state,
            i,
            data_source,
            conv_circuit,
            dense_circuit,
            measure_subset,
            dense_added_modes,
            output_proba_type,
            output_formatting,
        )
        print(f"Experiment {i + 1}/{n_runs} completed")
        all_results[f"run_{i}"] = results

    # Save all results
    save_results(
        all_results,
        output_dir,
        data_source,
        conv_circuit,
        dense_circuit,
        measure_subset,
        dense_added_modes,
        output_proba_type,
        output_formatting,
    )


if __name__ == "__main__":
    main()
