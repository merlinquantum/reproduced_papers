"""
Reproduce the readout-layer experiment where all possible two-mode configurations are evaluated.

Original usage was fully interactive; this module now also exposes a callable API so other scripts
can drive the experiment programmatically (e.g., `python utils/figure4.py`).
"""

import argparse
import itertools
import json
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix

from photonic_QCNN.lib.data import (
    convert_dataset_to_tensor,
    convert_tensor_to_loader,
    get_dataset,
)
from photonic_QCNN.lib.src.merlin_pqcnn import HybridModelReadout
from photonic_QCNN.lib.training.train_model import train_model_return_preds

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "custom_BAS"


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
    training_results, pred_test, y_test = train_model_return_preds(
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
    return training_results, training_results["final_test_acc"], pred_test, y_test


def _default_timestamped_dir() -> Path:
    now = datetime.now()
    day = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H-%M-%S")
    return DEFAULT_OUTPUT_ROOT / day / time_str


def save_results(all_results, output_dir: Path, k: int) -> Path:
    """Save results to JSON file"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save detailed results
    results_file = output_dir / f"first_readout_detailed_results_k_{k}.json"
    with results_file.open("w") as f:
        json.dump(all_results, f, indent=2)

    return results_file


def save_confusion_matrix(output_dir: Path, k: int, preds, labels) -> Path:
    """Save confusion matrix to png file"""
    output_dir.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(labels, preds)
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_percent = cm / cm.sum(axis=1, keepdims=True) * 100
        cm_percent = np.nan_to_num(cm_percent)
    annot_labels = np.array(
        [[f"{int(round(val))}%" for val in row] for row in cm_percent]
    )

    plt.figure(figsize=(4, 4))
    ax = sns.heatmap(
        cm_percent,
        annot=annot_labels,
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
    output_file = output_dir / f"first_readout_detailed_confusion_matrix_k_{k}.png"
    plt.savefig(output_file, dpi=300)
    plt.close()
    return output_file


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


def run_two_fold_readout(
    k: int,
    output_dir: Optional[Path] = None,
    *,
    data_source: str = "paper",
    conv_circuit: str = "BS",
    dense_circuit: str = "BS",
    dense_added_modes: int = 2,
    max_iter: Optional[int] = None,
) -> dict[str, Path]:
    """Programmatically execute the two-fold readout sweep."""
    if k not in (7, 8):
        raise ValueError(f"k must be 7 or 8, not {k}")
    if max_iter is not None and max_iter <= 0:
        raise ValueError("max_iter must be a positive integer.")

    output_dir = Path(output_dir) if output_dir else _default_timestamped_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Hyperparameters
    header = f"[Two-fold Readout] Starting sweep for k={k}"
    print(header)
    print(f"[Two-fold Readout] Results directory: {output_dir}")
    if max_iter is not None:
        print(
            "[Two-fold Readout] WARNING: max_iter is set. "
            "Stopping early will produce incomplete statistics."
        )

    # Run experiments with different k and different list_label_0
    random_state = 42
    best_test_acc = -1.0
    best_preds = None
    best_labels = None
    all_results = {}
    all_possible_list_label_0 = all_possible_label0_sets(k)
    time_start = time.time()
    total_label_sets = len(all_possible_list_label_0)
    target_total = (
        min(total_label_sets, max_iter) if max_iter is not None else total_label_sets
    )
    print(f"For k = {k}, there are {total_label_sets} possible label 0 lists")
    if max_iter is not None:
        print(
            f"Will evaluate at most {max_iter} label assignments (out of the full set)."
        )
    print(f"Planned experiments this run: {target_total}")

    experiments_run = 0
    for i, list_label_0 in enumerate(all_possible_list_label_0):
        if max_iter is not None and experiments_run >= max_iter:
            print("[Two-fold Readout] Reached max_iter limit; stopping early.")
            break
        print(f"About to start experiment {i}")
        results, test_acc, preds, labels = run_experiment(
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
        experiments_run += 1

        all_results[f"k_{k}_run_{i}"] = results
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_preds = preds
            best_labels = labels

        # Time report
        time_end = time.time()
        time_elapsed = time_end - time_start
        remaining = max(target_total - experiments_run, 0)
        time_left = (
            (time_elapsed / experiments_run * remaining) if experiments_run else 0
        )
        print(
            f"Time elapsed: {time_elapsed:.2f} seconds\nEstimated time left for k = {k}: {time_left:.2f} seconds"
        )

    # Save best confusion matrix (even if not perfect)
    confusion_path = None
    if best_preds is not None and best_labels is not None:
        confusion_path = save_confusion_matrix(output_dir, k, best_preds, best_labels)

    # Save all results
    json_path = save_results(all_results, output_dir, k)
    return {
        "results_json": json_path,
        "output_dir": output_dir,
        "confusion_matrix": confusion_path,
        "incomplete": experiments_run < total_label_sets,
        "experiments_run": experiments_run,
        "total_label_sets": total_label_sets,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the two-fold readout sweep (Figure 4b/c)."
    )
    parser.add_argument(
        "--k",
        type=int,
        choices=[7, 8],
        help="Number of two-mode configurations assigned to label 0.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory to store JSON/plots (defaults to results/custom_BAS/<timestamp>).",
    )
    parser.add_argument(
        "--max-iter",
        "--max_iter",
        dest="max_iter",
        type=int,
        help="Limit the number of label assignments evaluated (incomplete results).",
    )
    return parser.parse_args()


def main():
    """Main execution function"""
    args = _parse_args()
    if args.k is None:
        user_input = input(
            "Enter the number of modes pairs to associate to label 0 (7 or 8): "
        )
        args.k = int(user_input.strip())
    run_two_fold_readout(args.k, args.output_dir, max_iter=args.max_iter)


if __name__ == "__main__":
    main()
