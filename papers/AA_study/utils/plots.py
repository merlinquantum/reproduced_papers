from pathlib import Path
from typing import List, Optional

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
