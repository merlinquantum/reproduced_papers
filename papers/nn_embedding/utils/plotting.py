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
        ax_loss.text(
            0.805,
            0.59,
            "Classification accuracy (%)",
            transform=ax_loss.transAxes,
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="semibold",
            bbox={
                "facecolor": "white",
                "edgecolor": "#b0b0b0",
                "boxstyle": "round,pad=0.2",
                "alpha": 0.92,
            },
        )
        table = ax_loss.table(
            cellText=[
                [label, _format_accuracy(value)] for label, value in accuracy_rows
            ],
            colLabels=["Method", "Accuracy (%)"],
            cellLoc="center",
            colWidths=[0.5, 0.5],
            bbox=[0.60, 0.29, 0.38, 0.26],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_text_props(weight="semibold")

    fig.subplots_adjust(left=0.06, right=0.985, bottom=0.12, top=0.90, wspace=0.22)
    return _save_plot(fig, filename, run_dir)


def plot_figure_3(
    loss_lists_classifier: dict[str, Sequence[Sequence[float]]],
    test_accuracies: dict[str, Sequence[Sequence[float]]],
    *,
    layers_to_test: Sequence[int] | None = None,
    loss_iterations: Sequence[float] | None = None,
    figsize: tuple[float, float] = (10.2, 5.6),
    run_dir: Path | None = None,
    filename: str = "figure_3.pdf",
) -> Path:
    """Plot figure 3 style QCNN loss comparison with repeated runs.

    Parameters
    ----------
    loss_lists_classifier
        Mapping from method name to repeated classifier loss histories. Each
        value must be shaped like ``(num_repetitions, num_iterations)``.
        Expected keys are ``"pca_nqe"``, ``"nqe"``, and any number of
        ``"layer_<k>"`` entries.
    test_accuracies
        Mapping from method name to repeated test-accuracy histories. Each
        value must be shaped like ``(num_repetitions, num_iterations)``. The
        final accuracy from each repetition is summarized in the table.
    layers_to_test
        Optional ordered list of layer indices to show. If omitted, layer keys
        are inferred from ``loss_lists_classifier`` and sorted numerically.
    loss_iterations
        Optional x-axis values for the loss histories. Defaults to
        ``range(num_iterations)``.
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

    def _format_accuracy(values):
        arr = np.asarray(_to_plot_values(values), dtype=float)
        if arr.ndim == 2:
            arr = arr[:, -1]
        return f"{arr.mean():.1f} ± {arr.std():.1f}"

    def _mean_and_std(values, name):
        arr = _to_2d_float_array(values, name)
        return arr.mean(axis=0), arr.std(axis=0), arr

    if layers_to_test is None:
        layers_to_test = sorted(
            int(key.split("_", 1)[1])
            for key in loss_lists_classifier
            if key.startswith("layer_")
        )

    layer_keys = [f"layer_{layer}" for layer in layers_to_test]
    required_keys = ["pca_nqe", "nqe", *layer_keys]

    missing_loss = [key for key in required_keys if key not in loss_lists_classifier]
    missing_acc = [key for key in required_keys if key not in test_accuracies]
    if missing_loss:
        raise ValueError(f"Missing loss histories for keys: {missing_loss}")
    if missing_acc:
        raise ValueError(f"Missing test accuracies for keys: {missing_acc}")

    series = {}
    for key in required_keys:
        mean, std, arr = _mean_and_std(loss_lists_classifier[key], key)
        series[key] = {"mean": mean, "std": std, "arr": arr}

    num_iterations = series["pca_nqe"]["arr"].shape[1]
    if loss_iterations is None:
        loss_iterations = list(range(num_iterations))
    loss_iterations = list(loss_iterations)

    if len(loss_iterations) != num_iterations:
        raise ValueError(
            "loss_iterations must have the same length as the loss curves."
        )

    for key in required_keys[1:]:
        if series[key]["arr"].shape[1] != num_iterations:
            raise ValueError("All loss-history series must have the same length.")

    colors = {
        "layer_1": "#36b37e",
        "layer_2": "#39a8d0",
        "layer_3": "#c86bf0",
        "pca_nqe": "#f26f87",
        "nqe": "#b89d17",
    }
    line_styles = {
        "pca_nqe": "--",
        "nqe": "-.",
    }

    fig, ax = plt.subplots(figsize=figsize)

    for key in layer_keys:
        color = colors.get(key, None)
        label = f"Layer={key.split('_', 1)[1]}"
        mean = series[key]["mean"]
        std = series[key]["std"]
        ax.plot(
            loss_iterations,
            mean,
            color=color,
            linewidth=2.4,
            label=label,
        )
        ax.fill_between(
            loss_iterations,
            mean - std,
            mean + std,
            color=color,
            alpha=0.28,
            linewidth=0,
        )

    for key, label in (("pca_nqe", "PCA-NQE"), ("nqe", "NQE")):
        mean = series[key]["mean"]
        std = series[key]["std"]
        ax.plot(
            loss_iterations,
            mean,
            color=colors[key],
            linestyle=line_styles[key],
            linewidth=1.7,
            label=label,
        )
        ax.fill_between(
            loss_iterations,
            mean - std,
            mean + std,
            color=colors[key],
            alpha=0.28,
            linewidth=0,
        )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.legend(loc="upper right", frameon=True, ncol=2)

    accuracy_rows = [
        (f"Layer={layer}", _format_accuracy(test_accuracies[f"layer_{layer}"]))
        for layer in layers_to_test
    ]
    accuracy_rows.extend(
        [
            ("PCA-NQE", _format_accuracy(test_accuracies["pca_nqe"])),
            ("NQE", _format_accuracy(test_accuracies["nqe"])),
        ]
    )

    ax.text(
        0.09,
        0.25,
        "Classification accuracy",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=9,
        fontweight="semibold",
        bbox={
            "facecolor": "white",
            "edgecolor": "#b0b0b0",
            "boxstyle": "round,pad=0.2",
            "alpha": 0.92,
        },
    )
    table = ax.table(
        cellText=[[label, value] for label, value in accuracy_rows],
        colLabels=["Method", "Accuracy"],
        cellLoc="center",
        colWidths=[0.5, 0.5],
        bbox=[0.00, 0.00, 0.19, 0.24],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="semibold")

    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.12, top=0.95)
    return _save_plot(fig, filename, run_dir)


def plot_figure_4(
    effective_dimension: dict[str, Sequence[Sequence[float]]],
    *,
    samples_per_dataset: int | None = None,
    n_values: Sequence[float] | None = None,
    figsize: tuple[float, float] = (6.0, 4.5),
    run_dir: Path | None = None,
    filename: str = "figure_4.pdf",
) -> Path:
    """Plot figure 4: local effective dimension with and without NQE.

    Parameters
    ----------
    effective_dimension
        Mapping from method name to repeated effective-dimension curves. Each
        value must be shaped like ``(num_repetitions, num_n_values)``. Expected
        keys are ``"nqe"`` and ``"without_nqe"``.
    samples_per_dataset
        Number of samples used per dataset; used to reconstruct x-axis values
        as ``range(1, samples_per_dataset + 1, 1000)`` when *n_values* is not
        given.
    n_values
        Optional explicit x-axis values. Overrides *samples_per_dataset*.
    figsize, run_dir, filename
        Standard plotting/output configuration.
    """

    def _to_2d_float_array(values, name):
        arr = np.asarray(_to_plot_values(values), dtype=float)
        if arr.ndim != 2:
            raise ValueError(
                f"{name} must be a 2D array-like object shaped "
                "(num_repetitions, num_n_values)."
            )
        return arr

    required_keys = ("nqe", "without_nqe")
    missing = [k for k in required_keys if k not in effective_dimension]
    if missing:
        raise ValueError(f"Missing effective dimension for keys: {missing}")

    arrays = {k: _to_2d_float_array(effective_dimension[k], k) for k in required_keys}
    num_n = arrays["nqe"].shape[1]

    if n_values is not None:
        n_values = np.asarray(n_values, dtype=float)
    elif samples_per_dataset is not None:
        n_values = np.array(list(range(1, samples_per_dataset + 1, 1000)), dtype=float)
    else:
        n_values = np.arange(num_n, dtype=float)

    if len(n_values) != num_n:
        raise ValueError("n_values must have the same length as the data curves.")

    specs = (
        ("nqe", "With NQE", "#8b8b2a", "-"),
        ("without_nqe", "Without NQE", "#7b7bc4", "--"),
    )

    fig, ax = plt.subplots(figsize=figsize)

    for key, label, color, linestyle in specs:
        mean = arrays[key].mean(axis=0)
        std = arrays[key].std(axis=0)
        ax.plot(
            n_values,
            mean,
            color=color,
            linestyle=linestyle,
            linewidth=1.8,
            label=label,
        )
        ax.fill_between(
            n_values,
            mean - std,
            mean + std,
            color=color,
            alpha=0.25,
            linewidth=0,
        )

    ax.set_xlabel("Number of data")
    ax.set_ylabel("Local effective dimension")
    ax.legend(loc="center left", frameon=True)

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


def plot_figure_5(
    generalization_error: dict[str, Sequence[Sequence[float]]],
    *,
    weights: Sequence[float] | None = None,
    figsize: tuple[float, float] = (6.0, 4.5),
    run_dir: Path | None = None,
    filename: str = "figure_5.pdf",
) -> Path:
    """Plot figure 5: generalization error bound vs regularization weight.

    Parameters
    ----------
    generalization_error
        Mapping from method name to repeated error-bound lists. Each value
        must be shaped like ``(num_repetitions, num_weights)``. Expected keys
        are ``"pca_nqe"``, ``"nqe"``, and ``"without_nqe"``.
    weights
        Optional regularization weight values for the x-axis. If omitted,
        inferred from the data length as ``np.linspace(0.1, 0.9, num_weights)``.
    figsize, run_dir, filename
        Standard plotting/output configuration.
    """

    def _to_2d_float_array(values, name):
        arr = np.asarray(_to_plot_values(values), dtype=float)
        if arr.ndim != 2:
            raise ValueError(
                f"{name} must be a 2D array-like object shaped "
                "(num_repetitions, num_weights)."
            )
        return arr

    required_keys = ("pca_nqe", "nqe", "without_nqe")
    missing = [k for k in required_keys if k not in generalization_error]
    if missing:
        raise ValueError(f"Missing generalization error for keys: {missing}")

    arrays = {k: _to_2d_float_array(generalization_error[k], k) for k in required_keys}
    num_weights = arrays["pca_nqe"].shape[1]

    if weights is None:
        weights = np.linspace(0.1, 0.9, num_weights)
    weights = np.asarray(weights, dtype=float)

    if len(weights) != num_weights:
        raise ValueError("weights must have the same length as the error curves.")

    specs = (
        ("pca_nqe", "PCA-NQE", "#f26f87", "o"),
        ("nqe", "NQE", "#4daf3c", "^"),
        ("without_nqe", "Without NQE", "#339af0", "s"),
    )

    fig, ax = plt.subplots(figsize=figsize)

    for key, label, color, marker in specs:
        mean = arrays[key].mean(axis=0)
        std = arrays[key].std(axis=0)
        ax.errorbar(
            weights,
            mean,
            yerr=std,
            fmt=marker,
            color=color,
            markersize=6,
            capsize=3,
            label=label,
            zorder=3,
        )

    ax.set_xlabel(r"Regularization Weight ($\lambda$)")
    ax.set_ylabel("Generalization Error Bound (G)")
    ax.legend(loc="upper right", frameon=True)

    fig.tight_layout()
    return _save_plot(fig, filename, run_dir)


def plot_figure_6(
    train_deviation: dict[str, Sequence[float]],
    test_deviation: dict[str, Sequence[float]],
    train_kernel_var: dict[str, Sequence[float]],
    test_kernel_var: dict[str, Sequence[float]],
    *,
    figsize: tuple[float, float] = (10.0, 4.5),
    run_dir: Path | None = None,
    filename: str = "figure_6.pdf",
) -> Path:
    """Plot figure 6: deviation from 2-design and kernel variance.

    Parameters
    ----------
    train_deviation, test_deviation
        Mapping from method name to repeated scalar deviation values shaped
        ``(num_repetitions,)``. Expected keys: ``"pca_nqe"``, ``"nqe"``,
        ``"without_nqe"``.
    train_kernel_var, test_kernel_var
        Same structure for kernel variance values.
    figsize, run_dir, filename
        Standard plotting/output configuration.
    """

    required_keys = ("pca_nqe", "nqe", "without_nqe")
    for name, d in (
        ("train_deviation", train_deviation),
        ("test_deviation", test_deviation),
        ("train_kernel_var", train_kernel_var),
        ("test_kernel_var", test_kernel_var),
    ):
        missing = [k for k in required_keys if k not in d]
        if missing:
            raise ValueError(f"Missing {name} for keys: {missing}")

    labels = ("PCA-NQE", "NQE", "Without NQE")
    x = np.arange(len(labels))
    bar_width = 0.3

    # ── (a) Deviation from 2-design ──
    train_dev_means = np.array([np.mean(train_deviation[k]) for k in required_keys])
    train_dev_stds = np.array([np.std(train_deviation[k]) for k in required_keys])
    test_dev_means = np.array([np.mean(test_deviation[k]) for k in required_keys])
    test_dev_stds = np.array([np.std(test_deviation[k]) for k in required_keys])

    # ── (b) Kernel variance (train + test averaged) ──
    kernel_means = np.array(
        [
            np.mean(list(train_kernel_var[k]) + list(test_kernel_var[k]))
            for k in required_keys
        ]
    )
    kernel_stds = np.array(
        [
            np.std(list(train_kernel_var[k]) + list(test_kernel_var[k]))
            for k in required_keys
        ]
    )

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=figsize)

    # ── Panel (a) ──
    bars_train = ax_a.bar(
        x - bar_width / 2,
        train_dev_means,
        bar_width,
        yerr=train_dev_stds,
        color="black",
        capsize=3,
        label="Train data",
        zorder=3,
    )
    bars_test = ax_a.bar(
        x + bar_width / 2,
        test_dev_means,
        bar_width,
        yerr=test_dev_stds,
        color="white",
        edgecolor="black",
        linewidth=1.0,
        capsize=3,
        label="Test data",
        zorder=3,
    )

    for bar, val in zip(bars_train, train_dev_means):
        ax_a.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.008,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    for bar, val in zip(bars_test, test_dev_means):
        ax_a.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.008,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax_a.set_xticks(x)
    ax_a.set_xticklabels(labels)
    ax_a.set_ylabel("Deviation from 2-design")
    ax_a.set_ylim(bottom=0.0)
    ax_a.legend(loc="upper left", frameon=True)
    ax_a.set_title("(a)")

    # ── Panel (b) ──
    ax_b.bar(
        x,
        kernel_means,
        bar_width * 1.5,
        yerr=kernel_stds,
        color="gray",
        capsize=3,
        zorder=3,
    )

    ax_b.set_xticks(x)
    ax_b.set_xticklabels(labels)
    ax_b.set_ylabel("Kernel Variance")
    ax_b.set_ylim(bottom=0.0)
    ax_b.set_title("(b)")

    fig.tight_layout()
    return _save_plot(fig, filename, run_dir)
