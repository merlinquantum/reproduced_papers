from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def _save_plot(fig: plt.Figure, filename: str, run_dir: Path | None = None) -> Path:
    if run_dir is None:
        output_path = Path(__file__).parent.parent.resolve() / "results" / filename
    else:
        output_path = run_dir / filename

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return output_path


def _to_plot_values(values):
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy()
    if isinstance(values, np.ndarray):
        return values
    return [
        value.detach().cpu().item() if isinstance(value, torch.Tensor) else value
        for value in values
    ]


### Fig 2b
def plot_trace_distance_comparison(
    train_pca_nqe: Sequence[float],
    train_nqe: Sequence[float],
    test_pca_nqe: Sequence[float],
    test_nqe: Sequence[float],
    baseline: float,
    *,
    iterations: Sequence[float] | None = None,
    train_title: str = "Training data",
    test_title: str = "Test data",
    pca_label: str = "PCA-NQE",
    nqe_label: str = "NQE",
    baseline_label: str = "Without NQE",
    figsize: tuple[float, float] = (8.0, 4.2),
    run_dir: Path | None = None,
    filename: str = "trace_distance_comparison.pdf",
) -> Path:
    if iterations is None:
        iterations = range(len(train_pca_nqe))

    iterations = list(iterations)

    series_lengths = {
        len(train_pca_nqe),
        len(train_nqe),
        len(test_pca_nqe),
        len(test_nqe),
        len(iterations),
    }
    if len(series_lengths) != 1:
        raise ValueError("All data series and iterations must have the same length.")

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=False)

    panel_specs = (
        (axes[0], train_title, train_pca_nqe, train_nqe),
        (axes[1], test_title, test_pca_nqe, test_nqe),
    )

    for ax, title, pca_values, nqe_values in panel_specs:
        ax.scatter(
            iterations,
            pca_values,
            color="#f26f87",
            s=18,
            label=pca_label,
            zorder=3,
        )
        ax.scatter(
            iterations,
            nqe_values,
            color="#57b33e",
            marker="^",
            s=24,
            label=nqe_label,
            zorder=3,
        )
        ax.axhline(
            baseline,
            color="#3390ff",
            linestyle="--",
            linewidth=1.2,
            label=baseline_label,
            zorder=2,
        )

        ax.set_title(title)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Trace Distance")
        ax.set_xlim(min(iterations) - 0.5, max(iterations) + 0.5)
        ax.legend(loc="lower right", frameon=True)

    fig.tight_layout()
    return _save_plot(fig, filename, run_dir)


### Fig 2c
def plot_qcnn_loss_history(
    pca_nqe_mean: Sequence[float],
    pca_nqe_std: Sequence[float],
    nqe_mean: Sequence[float],
    nqe_std: Sequence[float],
    without_nqe_mean: Sequence[float],
    without_nqe_std: Sequence[float],
    *,
    iterations: Sequence[float] | None = None,
    lower_bound_pca_nqe: float | None = None,
    lower_bound_nqe: float | None = None,
    lower_bound_without_nqe: float | None = None,
    accuracy_rows: Sequence[tuple[str, str]] | None = None,
    title: str = "Noiseless QCNN Loss History",
    figsize: tuple[float, float] = (9.0, 4.8),
    run_dir: Path | None = None,
    filename: str = "qcnn_loss_history.pdf",
) -> Path:
    if iterations is None:
        iterations = range(len(pca_nqe_mean))

    iterations = list(iterations)

    series_lengths = {
        len(iterations),
        len(pca_nqe_mean),
        len(pca_nqe_std),
        len(nqe_mean),
        len(nqe_std),
        len(without_nqe_mean),
        len(without_nqe_std),
    }
    if len(series_lengths) != 1:
        raise ValueError("All loss series, std series, and iterations must match.")

    fig, ax = plt.subplots(figsize=figsize)

    colors = {
        "pca": "#f26f87",
        "nqe": "#4daf3c",
        "baseline": "#339af0",
    }

    ax.plot(
        iterations,
        pca_nqe_mean,
        linestyle="--",
        linewidth=1.6,
        color=colors["pca"],
        label="PCA-NQE",
    )
    ax.fill_between(
        iterations,
        [m - s for m, s in zip(pca_nqe_mean, pca_nqe_std)],
        [m + s for m, s in zip(pca_nqe_mean, pca_nqe_std)],
        color=colors["pca"],
        alpha=0.28,
        linewidth=0,
    )

    ax.plot(
        iterations,
        nqe_mean,
        linestyle="-.",
        linewidth=1.6,
        color=colors["nqe"],
        label="NQE",
    )
    ax.fill_between(
        iterations,
        [m - s for m, s in zip(nqe_mean, nqe_std)],
        [m + s for m, s in zip(nqe_mean, nqe_std)],
        color=colors["nqe"],
        alpha=0.28,
        linewidth=0,
    )

    ax.plot(
        iterations,
        without_nqe_mean,
        linestyle="-",
        linewidth=1.6,
        color=colors["baseline"],
        label="Without NQE",
    )
    ax.fill_between(
        iterations,
        [m - s for m, s in zip(without_nqe_mean, without_nqe_std)],
        [m + s for m, s in zip(without_nqe_mean, without_nqe_std)],
        color=colors["baseline"],
        alpha=0.28,
        linewidth=0,
    )

    if lower_bound_pca_nqe is not None:
        ax.axhline(
            lower_bound_pca_nqe,
            color=colors["pca"],
            linestyle="--",
            linewidth=3,
            label="Lower Bound with PCA-NQE",
        )
    if lower_bound_nqe is not None:
        ax.axhline(
            lower_bound_nqe,
            color=colors["nqe"],
            linestyle="-.",
            linewidth=3,
            label="Lower Bound with NQE",
        )
    if lower_bound_without_nqe is not None:
        ax.axhline(
            lower_bound_without_nqe,
            color=colors["baseline"],
            linestyle="-",
            linewidth=3,
            label="Lower Bound without NQE",
        )

    ax.set_title(title)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.legend(loc="upper right", frameon=True, ncol=2)

    if accuracy_rows:
        table = ax.table(
            cellText=[[label, value] for label, value in accuracy_rows],
            colLabels=["", "Classification accuracy (%)"],
            cellLoc="center",
            bbox=[0.64, 0.30, 0.34, 0.27],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)

    fig.tight_layout()
    return _save_plot(fig, filename, run_dir)


def plot_figure_2_bc(
    train_pca_nqe: Sequence[Sequence[float]],
    train_nqe: Sequence[Sequence[float]],
    test_pca_nqe: Sequence[Sequence[float]],
    test_nqe: Sequence[Sequence[float]],
    baseline_trace_distance: Sequence[Sequence[float]] | Sequence[float] | float,
    pca_nqe_losses: Sequence[Sequence[float]],
    nqe_losses: Sequence[Sequence[float]],
    without_nqe_losses: Sequence[Sequence[float]],
    *,
    trace_iterations: Sequence[float] | None = None,
    loss_iterations: Sequence[float] | None = None,
    lower_bound_pca_nqe: Sequence[float] | float | None = None,
    lower_bound_nqe: Sequence[float] | float | None = None,
    lower_bound_without_nqe: Sequence[float] | float | None = None,
    accuracy_rows: Sequence[tuple[str, Sequence[float] | str]] | None = None,
    figsize: tuple[float, float] = (14.0, 4.8),
    run_dir: Path | None = None,
    filename: str = "figure_2_bc.pdf",
) -> Path:
    """Plot figure 2(b) and 2(c) side by side and save the result to PDF.

    Parameters
    ----------
    train_pca_nqe, train_nqe, test_pca_nqe, test_nqe
        Repeated trace-distance histories shaped like
        ``(num_repetitions, num_iterations)`` for the two NQE variants in
        subfigure (b). The mean curve is computed internally.
    baseline_trace_distance
        Repeated baseline trace-distance histories or scalars for the
        "Without NQE" reference in subfigure (b). The mean value is used.
    pca_nqe_losses, nqe_losses, without_nqe_losses
        Repeated loss histories for the three classifier settings used in
        subfigure (c). Each argument should be shaped like
        ``(num_repetitions, num_iterations)``. The function computes the mean
        curve and one-standard-deviation band internally.
    trace_iterations, loss_iterations
        Optional iteration indices for the trace-distance and loss-history plots.
        If omitted, they default to ``range(len(series))``.
    lower_bound_pca_nqe, lower_bound_nqe, lower_bound_without_nqe
        Optional repeated lower-bound values or scalars drawn on the
        loss-history panel. Repeated values are averaged internally.
    accuracy_rows
        Optional rows for the accuracy summary table on the loss-history panel.
        Each row must be ``(label, value)`` where ``value`` is either a prebuilt
        string or a sequence of repeated accuracy values. Repeated values are
        summarized as ``mean ± std`` internally.
    figsize, run_dir, filename
        Standard plotting/output configuration.
    """
    def _to_2d_float_array(values, name):
        arr = np.asarray(_to_plot_values(values), dtype=float)
        if arr.ndim != 2:
            raise ValueError(
                f"{name} must be a 2D array-like object shaped "
                "(num_repetitions, num_iterations)."
            )
        return arr

    def _mean_scalar(values):
        arr = np.asarray(_to_plot_values(values), dtype=float)
        return float(arr.mean())

    def _format_accuracy(value):
        if isinstance(value, str):
            return value
        arr = np.asarray(_to_plot_values(value), dtype=float)
        return f"{arr.mean():.1f} ± {arr.std():.1f}"

    train_pca_nqe = _to_2d_float_array(train_pca_nqe, "train_pca_nqe")
    train_nqe = _to_2d_float_array(train_nqe, "train_nqe")
    test_pca_nqe = _to_2d_float_array(test_pca_nqe, "test_pca_nqe")
    test_nqe = _to_2d_float_array(test_nqe, "test_nqe")

    pca_nqe_losses = np.asarray(_to_plot_values(pca_nqe_losses), dtype=float)
    nqe_losses = np.asarray(_to_plot_values(nqe_losses), dtype=float)
    without_nqe_losses = np.asarray(_to_plot_values(without_nqe_losses), dtype=float)

    if pca_nqe_losses.ndim != 2 or nqe_losses.ndim != 2 or without_nqe_losses.ndim != 2:
        raise ValueError(
            "Loss inputs must each be a 2D array-like object shaped "
            "(num_repetitions, num_iterations)."
        )

    pca_nqe_mean = pca_nqe_losses.mean(axis=0)
    pca_nqe_std = pca_nqe_losses.std(axis=0)
    nqe_mean = nqe_losses.mean(axis=0)
    nqe_std = nqe_losses.std(axis=0)
    without_nqe_mean = without_nqe_losses.mean(axis=0)
    without_nqe_std = without_nqe_losses.std(axis=0)
    train_pca_nqe_mean = train_pca_nqe.mean(axis=0)
    train_nqe_mean = train_nqe.mean(axis=0)
    test_pca_nqe_mean = test_pca_nqe.mean(axis=0)
    test_nqe_mean = test_nqe.mean(axis=0)
    baseline_trace_distance = _mean_scalar(baseline_trace_distance)

    if trace_iterations is None:
        trace_iterations = range(train_pca_nqe.shape[1])
    if loss_iterations is None:
        loss_iterations = range(len(pca_nqe_mean))

    trace_iterations = list(trace_iterations)
    loss_iterations = list(loss_iterations)

    trace_lengths = {
        train_pca_nqe.shape[1],
        train_nqe.shape[1],
        test_pca_nqe.shape[1],
        test_nqe.shape[1],
        len(trace_iterations),
    }
    if len(trace_lengths) != 1:
        raise ValueError("All trace-distance series and iterations must match.")

    loss_lengths = {
        len(loss_iterations),
        pca_nqe_losses.shape[1],
        nqe_losses.shape[1],
        without_nqe_losses.shape[1],
    }
    if len(loss_lengths) != 1:
        raise ValueError("All loss-history series and iterations must match.")

    fig = plt.figure(figsize=figsize)
    outer = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.35], wspace=0.18)
    left = outer[0].subgridspec(1, 2, wspace=0.28)

    ax_train = fig.add_subplot(left[0, 0])
    ax_test = fig.add_subplot(left[0, 1])
    ax_loss = fig.add_subplot(outer[1])

    trace_panel_specs = (
        (ax_train, "Training data", train_pca_nqe_mean, train_nqe_mean),
        (ax_test, "Test data", test_pca_nqe_mean, test_nqe_mean),
    )

    for ax, title, pca_values, nqe_values in trace_panel_specs:
        ax.scatter(
            trace_iterations,
            pca_values,
            color="#f26f87",
            s=18,
            label="PCA-NQE",
            zorder=3,
        )
        ax.scatter(
            trace_iterations,
            nqe_values,
            color="#57b33e",
            marker="^",
            s=24,
            label="NQE",
            zorder=3,
        )
        ax.axhline(
            baseline_trace_distance,
            color="#3390ff",
            linestyle="--",
            linewidth=1.2,
            label="Without NQE",
            zorder=2,
        )
        ax.set_title(title)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Trace Distance")
        ax.set_xlim(min(trace_iterations) - 0.5, max(trace_iterations) + 0.5)
        ax.legend(loc="lower right", frameon=True)

    colors = {
        "pca": "#f26f87",
        "nqe": "#4daf3c",
        "baseline": "#339af0",
    }

    ax_loss.plot(
        loss_iterations,
        pca_nqe_mean,
        linestyle="--",
        linewidth=1.6,
        color=colors["pca"],
        label="PCA-NQE",
    )
    ax_loss.fill_between(
        loss_iterations,
        [m - s for m, s in zip(pca_nqe_mean, pca_nqe_std)],
        [m + s for m, s in zip(pca_nqe_mean, pca_nqe_std)],
        color=colors["pca"],
        alpha=0.28,
        linewidth=0,
    )

    ax_loss.plot(
        loss_iterations,
        nqe_mean,
        linestyle="-.",
        linewidth=1.6,
        color=colors["nqe"],
        label="NQE",
    )
    ax_loss.fill_between(
        loss_iterations,
        [m - s for m, s in zip(nqe_mean, nqe_std)],
        [m + s for m, s in zip(nqe_mean, nqe_std)],
        color=colors["nqe"],
        alpha=0.28,
        linewidth=0,
    )

    ax_loss.plot(
        loss_iterations,
        without_nqe_mean,
        linestyle="-",
        linewidth=1.6,
        color=colors["baseline"],
        label="Without NQE",
    )
    ax_loss.fill_between(
        loss_iterations,
        [m - s for m, s in zip(without_nqe_mean, without_nqe_std)],
        [m + s for m, s in zip(without_nqe_mean, without_nqe_std)],
        color=colors["baseline"],
        alpha=0.28,
        linewidth=0,
    )

    if lower_bound_pca_nqe is not None:
        ax_loss.axhline(
            _mean_scalar(lower_bound_pca_nqe),
            color=colors["pca"],
            linestyle="--",
            linewidth=3,
            label="Lower Bound with PCA-NQE",
        )
    if lower_bound_nqe is not None:
        ax_loss.axhline(
            _mean_scalar(lower_bound_nqe),
            color=colors["nqe"],
            linestyle="-.",
            linewidth=3,
            label="Lower Bound with NQE",
        )
    if lower_bound_without_nqe is not None:
        ax_loss.axhline(
            _mean_scalar(lower_bound_without_nqe),
            color=colors["baseline"],
            linestyle="-",
            linewidth=3,
            label="Lower Bound without NQE",
        )

    ax_loss.set_title("Noiseless QCNN Loss History")
    ax_loss.set_xlabel("Iteration")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend(loc="upper right", frameon=True, ncol=2)

    if accuracy_rows:
        table = ax_loss.table(
            cellText=[[label, _format_accuracy(value)] for label, value in accuracy_rows],
            colLabels=["", "Classification accuracy (%)"],
            cellLoc="center",
            bbox=[0.63, 0.29, 0.35, 0.28],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)

    fig.tight_layout()
    return _save_plot(fig, filename, run_dir)


### Simple trace distance plot
def plot_trace_distance(
    train_distances,
    test_distances,
    *,
    run_dir: Path | None = None,
    filename: str = "trace_distance.pdf",
) -> Path:
    train_distances = _to_plot_values(train_distances)
    test_distances = _to_plot_values(test_distances)
    iterations = list(range(len(train_distances)))

    series_lengths = {
        len(train_distances),
        len(test_distances),
        len(iterations),
    }
    if len(series_lengths) != 1:
        raise ValueError("All data series and iterations must have the same length.")

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 4.2), sharey=False)

    panel_specs = (
        (axes[0], "Training distances", train_distances),
        (axes[1], "Testing distances", test_distances),
    )

    for (
        ax,
        title,
        values,
    ) in panel_specs:
        ax.scatter(
            iterations,
            values,
            color="#f26f87",
            s=18,
            zorder=3,
        )

        ax.set_title(title)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Trace Distance")
        ax.set_xlim(min(iterations) - 0.5, max(iterations) + 0.5)
    fig.tight_layout()
    return _save_plot(fig, filename, run_dir)


### Quick loss plot
def quick_loss_plot(
    loss: Sequence[float],
    *,
    run_dir: Path | None = None,
    filename: str = "quick_loss_plot.pdf",
) -> Path:
    loss = _to_plot_values(loss)
    iterations = list(range(len(loss)))
    fig, ax = plt.subplots()
    ax.plot(iterations, loss, color="#339af0", linewidth=1.8)
    ax.set_title("Loss History")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    fig.tight_layout()
    return _save_plot(fig, filename, run_dir)


### Accuracies
def plot_accuracies(
    train_accuracies: Sequence[float],
    test_accuracies: Sequence[float],
    *,
    run_dir: Path | None = None,
    filename: str = "accuracies.pdf",
) -> Path:
    train_accuracies = _to_plot_values(train_accuracies)
    test_accuracies = _to_plot_values(test_accuracies)
    iterations = list(range(len(train_accuracies)))

    if len(train_accuracies) != len(test_accuracies):
        raise ValueError(
            "train_accuracies and test_accuracies must have the same length."
        )

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 4.2), sharey=True)

    panel_specs = (
        (axes[0], "Training Accuracy", train_accuracies),
        (axes[1], "Testing Accuracy", test_accuracies),
    )

    for ax, title, values in panel_specs:
        ax.plot(iterations, values, color="#4daf3c", linewidth=1.8)
        ax.set_title(title)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Accuracy")
        ax.set_xlim(min(iterations), max(iterations) if iterations else 0)

    fig.tight_layout()
    return _save_plot(fig, filename, run_dir)
