from pathlib import Path
from typing import List, Optional, Tuple
from torch.utils.data import TensorDataset

import numpy as np

import matplotlib.pyplot as plt


def plot_bas_run(
    accuracy_classical: List[float],
    accuracy_qiskit: List[float],
    accuracy_merlin: List[float],
    loss_classical: List[float],
    loss_qiskit: List[float],
    loss_merlin: List[float],
    run_dir: Optional[Path] = None,
):
    """
    Plot training accuracy and loss curves side by side. The plot is
    saved as a PDF file: results/training_metrics_graph.pdf (or to run_dir if set).

    Parameters
    ----------
    loss_list_epoch : list[float]
        Training loss per epoch.
    acc_list_epoch : list[float]
        Training accuracy per epoch.
    run_dir : pathlib.Path, optional
        Output directory for the PDF when running via the shared runtime. If None,
        the plot is saved under the local results folder.

    Returns
    -------
    None
    """

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(10, 3.6), sharex=True)
    fig.suptitle("Training Metrics", fontsize=12, fontweight="bold")

    ax_loss.plot(loss_classical, lw=2, color="tab:blue", label="Classical")
    ax_loss.plot(loss_qiskit, lw=2, color="tab:orange", label="Qiskit")
    ax_loss.plot(loss_merlin, lw=2, color="tab:green", label="Merlin")
    ax_loss.set_title("Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend(frameon=False)

    ax_acc.plot(accuracy_classical, lw=2, color="tab:blue", label="Classical")
    ax_acc.plot(accuracy_qiskit, lw=2, color="tab:orange", label="Qiskit")
    ax_acc.plot(accuracy_merlin, lw=2, color="tab:green", label="Merlin")
    ax_acc.set_title("Accuracy")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy (%)")
    ax_acc.legend(frameon=False)

    plt.tight_layout()
    if run_dir is None:
        output_path = (
            Path(__file__).parent.parent.resolve() / "results" / "bas_run_graph.pdf"
        )
        plt.savefig(output_path, format="pdf", bbox_inches="tight")
    else:
        plt.savefig(run_dir / "bas_run_graph.pdf", format="pdf", bbox_inches="tight")


def plot_amplitude_encoding_limitations(
    distances: List[float] | List[List[float]],
    dataset_unshuffled: TensorDataset,
    num_samples_per_class: int = 2000,
    fig_simulated: int = 1,
    run_dir: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """ """
    class1 = dataset_unshuffled.tensors[0][:num_samples_per_class]
    class2 = dataset_unshuffled.tensors[0][num_samples_per_class:]

    fig, (ax_scatter, ax_trace) = plt.subplots(1, 2, figsize=(10, 3.6))

    ax_scatter.scatter(class1[:, 0], class1[:, 1], s=10, color="tab:blue", alpha=0.7)
    ax_scatter.scatter(class2[:, 0], class2[:, 1], s=10, color="tab:red", alpha=0.7)
    ax_scatter.set_xlabel("x1")
    ax_scatter.set_ylabel("x2")
    ax_scatter.set_title("(a)")

    x_axis = np.arange(1, num_samples_per_class + 1)

    if fig_simulated == 1 or fig_simulated == 2:
        ax_trace.plot(x_axis, distances[0], color="tab:blue", lw=2, label="Class 1")
        ax_trace.plot(
            x_axis,
            distances[1],
            color="tab:red",
            lw=2,
            linestyle=":",
            label="Class 2",
        )
        ax_trace.legend(frameon=False)
    elif fig_simulated == 3:
        ax_trace.plot(x_axis, distances, color="tab:purple", lw=2)
    ax_trace.set_xlabel("Sample size per class")
    ax_trace.set_ylabel("Trace distance")
    ax_trace.set_title("(b)")

    plt.tight_layout()
    if run_dir is None:
        output_path = (
            Path(__file__).parent.parent.resolve()
            / "results"
            / f"fig{fig_simulated}_amplitude_encoding.pdf"
        )
        plt.savefig(output_path, format="pdf", bbox_inches="tight")
    else:
        plt.savefig(
            run_dir / f"fig{fig_simulated}_amplitude_encoding.pdf",
            format="pdf",
            bbox_inches="tight",
        )
