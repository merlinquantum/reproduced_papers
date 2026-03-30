from collections.abc import Sequence
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import TensorDataset


def plot_bas_run(
    accuracy_classical: list[float],
    accuracy_qiskit: list[float],
    accuracy_merlin: list[float],
    loss_classical: list[float],
    loss_qiskit: list[float],
    loss_merlin: list[float],
    run_dir: Optional[Path] = None,
):
    """
    Plot BAS training accuracy and loss curves side by side.

    Parameters
    ----------
    accuracy_classical : list[float]
        Classical model accuracy per epoch.
    accuracy_qiskit : list[float]
        Qiskit model accuracy per epoch.
    accuracy_merlin : list[float]
        Merlin model accuracy per epoch.
    loss_classical : list[float]
        Classical model loss per epoch.
    loss_qiskit : list[float]
        Qiskit model loss per epoch.
    loss_merlin : list[float]
        Merlin model loss per epoch.
    run_dir : pathlib.Path, optional
        Output directory for the PDF. If None, saves under the local results folder.

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
    distances: list[float] | list[list[float]],
    dataset_unshuffled: TensorDataset,
    num_samples_per_class: int = 2000,
    fig_simulated: int = 1,
    run_dir: Optional[Path] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Plot the amplitude-encoding limitation curves for Figs. 1 to 3.

    Parameters
    ----------
    distances : list[float] or list[list[float]]
        Trace-distance series. For Fig. 1 and Fig. 2 use a list of two curves.
        For Fig. 3 use a single curve.
    dataset_unshuffled : torch.utils.data.TensorDataset
        Dataset in class-ordered (unshuffled) form.
    num_samples_per_class : int, optional
        Number of samples per class used for the plot.
    fig_simulated : int, optional
        Figure index (1, 2, or 3) to control plotting style.
    run_dir : pathlib.Path, optional
        Output directory for the PDF. If None, saves under the local results folder.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Arrays corresponding to the class-1 and class-2 sample coordinates.
    """
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


def plot_fig_4(
    layers_tested: list[int],
    qiskit_accuracies: Sequence[
        Sequence[Union[Sequence[float], Sequence[Sequence[float]]]]
    ],
    amplitude_accuracies: Sequence[
        Sequence[Union[Sequence[float], Sequence[Sequence[float]]]]
    ],
    angle_accuracies: Sequence[
        Sequence[Union[Sequence[float], Sequence[Sequence[float]]]]
    ],
    qiskit_losses: Sequence[
        Sequence[Union[Sequence[float], Sequence[Sequence[float]]]]
    ],
    amplitude_losses: Sequence[
        Sequence[Union[Sequence[float], Sequence[Sequence[float]]]]
    ],
    angle_losses: Sequence[Sequence[Union[Sequence[float], Sequence[Sequence[float]]]]],
    run_dir: Optional[Path] = None,
):
    """
    Plot Fig. 4(b)-style grids of loss/accuracy over epochs.

    Parameters
    ----------
    layers_tested : list[int]
        Layer counts evaluated for each dataset.
    qiskit_accuracies : Sequence
        Accuracy curves for Qiskit models (datasets × layers × epochs).
    amplitude_accuracies : Sequence
        Accuracy curves for amplitude-encoding models.
    angle_accuracies : Sequence
        Accuracy curves for angle-encoding models.
    qiskit_losses : Sequence
        Loss curves for Qiskit models.
    amplitude_losses : Sequence
        Loss curves for amplitude-encoding models.
    angle_losses : Sequence
        Loss curves for angle-encoding models.
    run_dir : pathlib.Path, optional
        Output directory for the PDFs. If None, saves under the local results folder.

    Returns
    -------
    None
    """

    def _summarize_runs(
        losses: Union[Sequence[float], Sequence[Sequence[float]]],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        arr = np.asarray(losses, dtype=float)
        if arr.ndim == 1:
            arr = arr[None, :]
        mean = arr.mean(axis=0)
        low = arr.min(axis=0)
        high = arr.max(axis=0)
        return mean, low, high

    def _plot_grid(
        model_losses: Sequence[
            Sequence[Union[Sequence[float], Sequence[Sequence[float]]]]
        ],
        model_accuracies: Sequence[
            Sequence[Union[Sequence[float], Sequence[Sequence[float]]]]
        ],
        title: str,
        output_name: str,
    ) -> None:
        num_layers = len(layers_tested)
        num_datasets = len(model_losses)

        fig, axes = plt.subplots(
            num_layers,
            num_datasets,
            figsize=(3.2 * num_datasets, 2.8 * num_layers),
            squeeze=False,
            sharex=True,
            sharey=True,
        )
        fig.suptitle(title, fontsize=12, fontweight="bold")

        last_epoch = None
        for d_idx in range(num_datasets):
            for l_idx in range(num_layers):
                ax = axes[l_idx, d_idx]
                mean_loss, low_loss, high_loss = _summarize_runs(
                    model_losses[d_idx][l_idx]
                )
                mean_acc, _, _ = _summarize_runs(model_accuracies[d_idx][l_idx])
                epochs = np.arange(1, len(mean_loss) + 1)
                last_epoch = len(mean_loss)
                ax.plot(epochs, mean_loss, color="tab:blue", lw=2, label="Loss")
                ax.fill_between(
                    epochs, low_loss, high_loss, color="tab:blue", alpha=0.2
                )
                ax.plot(epochs, mean_acc, color="tab:red", lw=2, label="Accuracy")
                ax.axhline(
                    np.log(2),
                    color="tab:gray",
                    lw=1,
                    linestyle="--",
                    label="ln(2)",
                )
                ax.set_box_aspect(1)

                if l_idx == num_layers - 1:
                    ax.set_xlabel("Epoch")
                if d_idx == 0:
                    ax.set_ylabel("Loss")
                ax.set_title(f"Dataset {d_idx + 1}, L={layers_tested[l_idx]}")

        if last_epoch is not None:
            fig.text(
                0.5,
                0.01,
                f"Last epoch: {last_epoch}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        from matplotlib.lines import Line2D

        legend_handles = [
            Line2D([0], [0], color="tab:blue", lw=2, label="loss"),
            Line2D([0], [0], color="tab:red", lw=2, label="acc."),
            Line2D([0], [0], color="tab:gray", lw=1, linestyle="--", label="ln 2"),
        ]
        fig.legend(
            handles=legend_handles,
            loc="center right",
            bbox_to_anchor=(0.98, 0.5),
            frameon=True,
        )

        plt.tight_layout(rect=[0, 0.03, 0.9, 0.92])
        if run_dir is None:
            output_path = (
                Path(__file__).parent.parent.resolve() / "results" / output_name
            )
            plt.savefig(output_path, format="pdf", bbox_inches="tight")
        else:
            plt.savefig(run_dir / output_name, format="pdf", bbox_inches="tight")

    _plot_grid(
        qiskit_losses,
        qiskit_accuracies,
        title="Qiskit Model - Training Loss",
        output_name="fig4b_qiskit.pdf",
    )
    _plot_grid(
        amplitude_losses,
        amplitude_accuracies,
        title="Amplitude Encoding - Training Loss",
        output_name="fig4b_amplitude.pdf",
    )
    _plot_grid(
        angle_losses,
        angle_accuracies,
        title="Angle Encoding - Training Loss",
        output_name="fig4b_angle.pdf",
    )


def plot_fig_5(
    sample_sizes: Sequence[int],
    MNIST_trace_distances: list[float],
    CIFAR_10_trace_distances: list[float],
    PathMNIST_trace_distances: list[float],
    EuroSAT_trace_distances: list[float],
    run_dir: Optional[Path] = None,
    add_inset: bool = True,
) -> None:
    """
    Plot Fig. 5: trace distance between averaged encoded states.

    Parameters
    ----------
    sample_sizes : Sequence[int]
        Sample sizes per class.
    MNIST_trace_distances : list[float]
        Trace-distance curve for MNIST.
    CIFAR_10_trace_distances : list[float]
        Trace-distance curve for CIFAR-10.
    PathMNIST_trace_distances : list[float]
        Trace-distance curve for PathMNIST.
    EuroSAT_trace_distances : list[float]
        Trace-distance curve for EuroSAT.
    run_dir : pathlib.Path, optional
        Output directory for the PDF. If None, saves under the local results folder.
    add_inset : bool, optional
        Whether to include a small inset plot for early samples.

    Returns
    -------
    None
    """
    fig, ax = plt.subplots(1, 1, figsize=(5.2, 3.6))

    series = [
        ("MNIST", MNIST_trace_distances, "tab:blue"),
        ("CIFAR-10", CIFAR_10_trace_distances, "tab:orange"),
        ("PathMNIST", PathMNIST_trace_distances, "tab:green"),
        ("EuroSAT", EuroSAT_trace_distances, "tab:red"),
    ]
    for name, values, color in series:
        ax.plot(
            sample_sizes,
            values,
            label=name,
            color=color,
            lw=2,
            marker="o",
            ms=4,
        )

    ax.set_xlabel("Sample size per class")
    ax.set_ylabel("Trace distance")
    ax.set_xticks([sample_sizes[0], sample_sizes[-1]])
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="center left", bbox_to_anchor=(0.18, 0.5), frameon=False)

    if add_inset:
        try:
            from mpl_toolkits.axes_grid1.inset_locator import inset_axes

            axins = inset_axes(ax, width="40%", height="40%", loc="center right")
            for _, values, color in series:
                axins.plot(
                    sample_sizes,
                    values,
                    color=color,
                    lw=1.5,
                    marker="o",
                    ms=3,
                )
            axins.set_xlim(sample_sizes[0], sample_sizes[-2])
            axins.set_ylim(0.0, 0.03)
            axins.set_xticks([sample_sizes[0], sample_sizes[-2]])
            axins.set_yticks([0.01, 0.15])
        except Exception:
            pass

    plt.tight_layout()
    if run_dir is None:
        output_path = Path(__file__).parent.parent.resolve() / "results" / "fig5.pdf"
        plt.savefig(output_path, format="pdf", bbox_inches="tight")
    else:
        plt.savefig(run_dir / "fig5.pdf", format="pdf", bbox_inches="tight")


def plot_fig_7(
    sample_sizes: Sequence[int],
    training_losses: Sequence[Sequence[float]],
    generalization_errors: Sequence[float],
    testing_accuracies: Sequence[Sequence[float]],
    model_name: str = "merlin",
    run_dir: Optional[Path] = None,
) -> None:
    """
    Plot Fig. 7: MNIST performance under amplitude encoding.

    Parameters
    ----------
    sample_sizes : Sequence[int]
        Sample sizes per class.
    training_losses : Sequence[Sequence[float]]
        Training loss curves, one per sample size.
    generalization_errors : Sequence[float]
        Generalization error per sample size.
    testing_accuracies : Sequence[Sequence[float]]
        Testing accuracy curves, one per sample size.
    model_name : str, optional
        Model label used in the output filename.
    run_dir : pathlib.Path, optional
        Output directory for the PDF. If None, saves under the local results folder.

    Returns
    -------
    None
    """
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.2))

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    # (a) Training loss vs epoch
    ax = axes[0]
    for idx, (size, loss_curve) in enumerate(zip(sample_sizes, training_losses)):
        epochs = np.arange(1, len(loss_curve) + 1)
        ax.plot(
            epochs,
            loss_curve,
            color=colors[idx % len(colors)],
            lw=2,
            label=str(size),
        )
    ax.axhline(np.log(2), color="tab:gray", lw=1, linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training loss")
    ax.set_title("(a) Training loss")

    # (b) Generalization error vs sample size per class (log scale x)
    gen_error_values = []
    for value in generalization_errors:
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            if len(value) == 0:
                raise ValueError("generalization_errors contains an empty curve")
            gen_error_values.append(value[-1])
        else:
            gen_error_values.append(value)
    ax = axes[1]
    ax.plot(
        sample_sizes,
        gen_error_values,
        color="tab:blue",
        lw=2,
        marker="o",
    )
    ax.set_xscale("log")
    ax.set_xlabel("Sample size per class")
    ax.set_ylabel("Generalization error")
    ax.set_title("(b) Generalization error")

    # (c) Testing accuracy vs epoch
    ax = axes[2]
    for idx, (size, acc_curve) in enumerate(zip(sample_sizes, testing_accuracies)):
        epochs = np.arange(1, len(acc_curve) + 1)
        ax.plot(
            epochs,
            acc_curve,
            color=colors[idx % len(colors)],
            lw=2,
            label=str(size),
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Testing accuracy")
    ax.set_title("(c) Testing accuracy")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=len(labels),
            frameon=False,
            title="Sample size per class",
        )

    plt.tight_layout(rect=[0, 0, 1, 0.9])
    if run_dir is None:
        output_path = (
            Path(__file__).parent.parent.resolve()
            / "results"
            / f"fig7_{model_name}.pdf"
        )
        plt.savefig(output_path, format="pdf", bbox_inches="tight")
    else:
        plt.savefig(
            run_dir / f"fig7_{model_name}.pdf", format="pdf", bbox_inches="tight"
        )
