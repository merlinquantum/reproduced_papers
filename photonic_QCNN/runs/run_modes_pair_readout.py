"""
This is to reproduce the experiment on the training of the readout layer (measurement layer). This particular file
presents the second approach presented by the paper which is to associate a pair of modes to label 0.

This experiment is conducted on Custom BAS with the same architecture as the paper so there are 6 final modes for
measurement. Thus, there are 15 pairs of modes configurations to consider.
"""

import itertools
import json
import os
import random
from datetime import datetime

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix

from photonic_QCNN.data.data import (
    convert_dataset_to_tensor,
    convert_tensor_to_loader,
    get_dataset,
)
from photonic_QCNN.src.merlin_pqcnn import HybridModelReadout
from photonic_QCNN.training.train_model import train_model_return_preds


def run_experiment(
    random_state,
    run_id,
    source,
    conv_circuit,
    dense_circuit,
    dense_added_modes,
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
    training_results, preds, labels = train_model_return_preds(
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
    return training_results, training_results["final_test_acc"], preds, labels


def save_results(all_results, output_dir):
    """Save results to JSON file"""
    os.makedirs(output_dir, exist_ok=True)

    # Save detailed results
    results_file = os.path.join(output_dir, "second_readout_detailed_results.json")
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)

    return


def save_confusion_matrix(output_dir, preds, labels):
    """Save confusion matrix to png file"""
    os.makedirs(output_dir, exist_ok=True)

    # Compute confusion matrix
    cm = confusion_matrix(labels, preds)

    # Convert to percentages
    cm_percent = cm / cm.sum(axis=1, keepdims=True) * 100

    # Create string labels for annotation
    annot_labels = np.array(
        [[f"{int(round(val))}%" for val in row] for row in cm_percent]
    )

    plt.figure(figsize=(4, 4))
    ax = sns.heatmap(
        cm_percent,
        annot=annot_labels,
        fmt="",
        cmap="Reds",
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
        output_dir, "second_readout_detailed_confusion_matrix.png"
    )
    plt.savefig(output_file, dpi=300)
    plt.close()


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

    # Run experiments with different list_label_0 and different random states
    all_results = {}
    all_possible_list_label_0 = all_possible_label0_sets(1)
    assert len(all_possible_list_label_0) == 15
    print(f"There are {len(all_possible_list_label_0)} possible label 0 modes pair")

    random_state = 42
    num_runs = 30
    random.seed(random_state)
    random_states = [int(random.uniform(0, 1000)) for _ in range(num_runs)]

    best_test_acc = 0
    best_preds = None
    best_labels = None

    for i, list_label_0 in enumerate(all_possible_list_label_0):
        print(f"For pair {i}: pair {list_label_0}")
        for j, random_state in enumerate(random_states):
            print(f"About to start experiment for run {j}: random_state {random_state}")
            results, test_acc, preds, labels = run_experiment(
                random_state,
                i,
                data_source,
                conv_circuit,
                dense_circuit,
                dense_added_modes,
                list_label_0,
            )
            print(f"Run {j} completed")

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_preds = preds
                best_labels = labels

            all_results[f"pair_{i}_run_{j}"] = results

    # Save confusion matrix of best test accuracy model
    save_confusion_matrix(output_dir, best_preds, best_labels)

    # Save all results
    save_results(all_results, output_dir)
    return


if __name__ == "__main__":
    main()
