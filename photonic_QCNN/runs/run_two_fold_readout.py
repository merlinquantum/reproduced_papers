"""
This is to reproduce the experiment on the training of the readout layer (measurement layer). This particular file
presents the first approach presented by the paper which is to consider all possibles two-modes configurations and to
associate each one to a label.

This experiment is conducted on Custom BAS with the same architecture as the paper so there are 6 final modes for
measurement. Thus, there are 15 2-modes configurations. k of which will be associated to label 0 and 15-k will be
associated to label 1. We run this experiment for k = 7 and 8 for reproduction purposes.
"""

import itertools
import json
import os
import random
import time
from datetime import datetime

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from photonic_QCNN.data.data import (
    convert_dataset_to_tensor,
    convert_tensor_to_loader,
    get_dataset,
)
from photonic_QCNN.src.merlin_pqcnn import HybridModelReadout
from photonic_QCNN.training.train_model import train_model


def run_experiment(
    random_state,
    run_id,
    source,
    conv_circuit,
    dense_circuit,
    dense_added_modes,
    k,
    list_label_0,
):
    """Run the complete experiment for one random state on Custom BAS"""
    print(f"Running experiment {run_id} with random state {random_state} on Custom BAS")

    # Set random seeds
    torch.manual_seed(random_state)
    np.random.seed(random_state)
    random.seed(random_state)

    # Load dataset
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

    model = HybridModelReadout(
        dims, conv_circuit, dense_circuit, dense_added_modes, list_label_0
    )
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {num_params} trainable parameters")
    print(f"Output of circuit has size {model.qcnn_output_dim}")

    # Train model
    training_results = train_model(
        model, train_loader, x_train, x_test, y_train, y_test
    )

    # Shorten training results
    training_results = {
        "final_train_acc": training_results["final_train_acc"],
        "final_test_acc": training_results["final_test_acc"],
    }

    print(
        f"Custom BAS - Final train: {training_results['final_train_acc']:.4f}, test: {training_results['final_test_acc']:.4f}"
    )
    return training_results


def save_results(all_results, output_dir, k):
    """Save results to JSON file"""
    os.makedirs(output_dir, exist_ok=True)

    # Save detailed results
    results_file = os.path.join(
        output_dir, f"first_readout_detailed_results_k_{k}.json"
    )
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)

    return


def save_confusion_matrix(output_dir, k):
    """Save confusion matrix to png file"""
    os.makedirs(output_dir, exist_ok=True)

    cm_percent = np.array([[100, 0], [0, 100]])

    labels = np.array([["100%", "0%"], ["0%", "100%"]])

    plt.figure(figsize=(4, 4))
    ax = sns.heatmap(
        cm_percent,
        annot=labels,
        fmt="",
        cmap="Blues",
        cbar=False,
        square=True,
        xticklabels=[r"$T_0$", r"$T_1$"],
        yticklabels=[r"$P_0$", r"$P_1$"],
    )

    # Add black border around the heatmap
    rect = patches.Rectangle(
        (0, 0),  # bottom left corner
        cm_percent.shape[1],  # width
        cm_percent.shape[0],  # height
        fill=False,  # no fill, just outline
        color="black",
        linewidth=2,
    )
    ax.add_patch(rect)

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12, rotation=0)
    plt.tight_layout()
    output_file = os.path.join(
        output_dir, f"first_readout_detailed_confusion_matrix_k_{k}.png"
    )
    plt.savefig(output_file, dpi=300)
    return


def all_possible_label0_sets(k):
    modes = list(range(6))
    # Step 1: all 15 binary pairs
    pairs = list(itertools.combinations(modes, 2))
    binary_pairs = []
    for i, j in pairs:
        vec = [0] * 6
        vec[i] = 1
        vec[j] = 1
        binary_pairs.append(tuple(vec))

    # Step 2: all possible choices of size k from these 15
    return list(itertools.combinations(binary_pairs, k))


def main():
    """Main execution function"""
    k = int(input("Enter the number of modes pairs to associate to label 0 (7 or 8): "))
    assert k == 7 or k == 8, f"k must be either 7 or 8, not {k}"

    # Hyperparameters
    data_source = "paper"  # ['scratch', 'paper']
    conv_circuit = "BS"  # ['MZI', 'BS', 'BS_random_PS']
    dense_circuit = "BS"  # ['MZI', 'BS', 'BS_random_PS']
    dense_added_modes = 2

    # Create output directory with current date and time
    now = datetime.now()
    day = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H-%M-%S")
    output_dir = f"../results/custom_BAS/{day}/{time_str}"

    print("Starting Photonic QCNN experiments...")
    print(f"Results will be saved to: {output_dir}")

    # Run experiments with different k and different list_label_0
    random_state = 42
    confusion_matrix = True
    all_results = {}
    all_possible_list_label_0 = all_possible_label0_sets(k)
    time_start = time.time()
    print(
        f"For k = {k}, there are {len(all_possible_list_label_0)} possible label 0 lists"
    )

    for i, list_label_0 in enumerate(all_possible_list_label_0):
        print(f"About to start experiment {i}")
        results = run_experiment(
            random_state,
            i,
            data_source,
            conv_circuit,
            dense_circuit,
            dense_added_modes,
            k,
            list_label_0,
        )
        print(f"Experiment {i} completed")

        # If not saved yet, save perfect confusion matrix
        if (
            confusion_matrix
            and results["final_train_acc"] == 1
            and results["final_test_acc"] == 1
        ):
            save_confusion_matrix(output_dir, k)
            confusion_matrix = False

        all_results[f"k_{k}_run_{i}"] = results

        # Time report
        time_end = time.time()
        time_elapsed = time_end - time_start
        time_left = time_elapsed / (i + 1) * (len(all_possible_list_label_0) - i - 1)
        print(
            f"Time elapsed: {time_elapsed:.2f} seconds\nEstimated time left for k = {k}: {time_left:.2f} seconds"
        )

    # Save all results
    save_results(all_results, output_dir, k)
    return


if __name__ == "__main__":
    main()
