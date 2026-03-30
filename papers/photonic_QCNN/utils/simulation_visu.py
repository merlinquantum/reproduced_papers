"""
Generate the training/validation curves used in Figure 12 of the paper.

Usage examples:

    python simulation_visu.py --detailed-results path/to/detailed_results.json
    python simulation_visu.py --detailed-results ... --output custom_plot.png

If no CLI arguments are provided, the script will prompt for `detailed_results.json`
and write `simulation_results.png` next to it.
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
from matplotlib import pyplot as plt


def aggregate_loss_per_epoch(loss_history, num_batches):
    return [
        np.mean(loss_history[i * num_batches : (i + 1) * num_batches])
        for i in range(len(loss_history) // num_batches)
    ]


def generate_simulation_plot(
    detailed_results_path: Path, output_path: Optional[Path] = None
) -> Path:
    """Create the Figure 12-style visualization from a detailed_results.json file."""
    detailed_results_path = Path(detailed_results_path)
    if output_path is None:
        output_path = detailed_results_path.parent / "simulation_results.png"
    else:
        output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with detailed_results_path.open() as f:
        data = json.load(f)

    loss_histories = []
    test_loss_histories = []
    train_acc_histories = []
    test_acc_histories = []

    for run_data in data.values():
        losses = run_data["loss_history"]
        test_losses = run_data["test_loss_history"]
        train_accs = run_data["train_acc_history"]
        test_accs = run_data["test_acc_history"]

        n_epochs = len(train_accs)
        num_batches = len(losses) // (n_epochs - 1)

        # Convert batch losses → epoch losses
        epoch_losses = aggregate_loss_per_epoch(losses, num_batches)

        loss_histories.append(epoch_losses)
        test_loss_histories.append(test_losses)
        train_acc_histories.append(train_accs)
        test_acc_histories.append(test_accs)

    # Now all runs have same length
    loss_histories = np.array(loss_histories)
    test_loss_histories = np.array(test_loss_histories)
    train_acc_histories = np.array(train_acc_histories)
    test_acc_histories = np.array(test_acc_histories)

    loss_epochs = np.arange(loss_histories.shape[1])
    accs_epochs = np.arange(train_acc_histories.shape[1])

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    # ---- Loss Figure ----
    ax = axes[1]
    ax.plot(
        loss_epochs,
        loss_histories.mean(axis=0),
        color="blue",
        label="Mean Train Loss ± Std",
    )
    ax.fill_between(
        loss_epochs,
        loss_histories.mean(axis=0) - loss_histories.std(axis=0),
        loss_histories.mean(axis=0) + loss_histories.std(axis=0),
        color="blue",
        alpha=0.2,
    )
    ax.plot(
        loss_epochs,
        test_loss_histories.mean(axis=0),
        color="orange",
        label="Mean Test Loss ± Std",
    )
    ax.fill_between(
        loss_epochs,
        test_loss_histories.mean(axis=0) - test_loss_histories.std(axis=0),
        test_loss_histories.mean(axis=0) + test_loss_histories.std(axis=0),
        color="orange",
        alpha=0.2,
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True)

    # ---- Accuracy Figure ----
    ax = axes[0]
    ax.plot(
        accs_epochs,
        train_acc_histories.mean(axis=0) * 100,
        color="blue",
        label="Mean Train Acc ± Std",
    )
    ax.fill_between(
        accs_epochs,
        (train_acc_histories.mean(axis=0) - train_acc_histories.std(axis=0)) * 100,
        (train_acc_histories.mean(axis=0) + train_acc_histories.std(axis=0)) * 100,
        color="blue",
        alpha=0.2,
    )
    ax.plot(
        accs_epochs,
        test_acc_histories.mean(axis=0) * 100,
        color="orange",
        label="Mean Test Acc ± Std",
    )
    ax.fill_between(
        accs_epochs,
        (test_acc_histories.mean(axis=0) - test_acc_histories.std(axis=0)) * 100,
        (test_acc_histories.mean(axis=0) + test_acc_histories.std(axis=0)) * 100,
        color="orange",
        alpha=0.2,
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.legend()
    ax.grid(True)

    # ---- Individual Runs Figure ----
    ax = axes[2]
    colors = ["green", "red", "blue", "purple", "orange"]
    for i, (train_acc, test_acc) in enumerate(
        zip(train_acc_histories, test_acc_histories)
    ):
        ax.plot(
            accs_epochs,
            train_acc * 100,
            color=colors[i],
            linestyle="-",
            label=f"Train Run {i}",
        )
        ax.plot(
            accs_epochs,
            test_acc * 100,
            color=colors[i],
            linestyle="--",
            label=f"Test Run {i}",
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.legend()
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Figure 12-style plots from MerLin detailed results."
    )
    parser.add_argument(
        "--detailed-results",
        type=Path,
        dest="detailed_results",
        help="Path to detailed_results.json produced by a MerLin run.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional output path for the generated PNG.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    detailed_path = args.detailed_results
    if detailed_path is None:
        detailed_path = Path(input("Enter the path to detailed_results.json: ").strip())
    generate_simulation_plot(detailed_path, args.output)


if __name__ == "__main__":
    main()
