from collections.abc import Sequence

import matplotlib.pyplot as plt


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
) -> tuple[plt.Figure, Sequence[plt.Axes]]:
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
    return fig, axes


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
) -> tuple[plt.Figure, plt.Axes]:
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
    return fig, ax


### Simple trac distance plot
def plot_trace_distance(
    train_distances,
    test_distances,
) -> tuple[plt.Figure, Sequence[plt.Axes]]:
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
        ax.legend(loc="lower right", frameon=True)

    fig.tight_layout()
    return fig, axes


### Quick loss plot
def quick_loss_plot(loss: Sequence[float]) -> tuple[plt.Figure, plt.Axes]:
    iterations = list(range(len(loss)))
    fig, ax = plt.subplots()
    ax.plot(iterations, loss, color="#339af0", linewidth=1.8)
    ax.set_title("Loss History")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    fig.tight_layout()
    return fig, ax


### Accuracies
def plot_accuracies(
    train_accuracies: Sequence[float],
    test_accuracies: Sequence[float],
) -> tuple[plt.Figure, Sequence[plt.Axes]]:
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
    return fig, axes
