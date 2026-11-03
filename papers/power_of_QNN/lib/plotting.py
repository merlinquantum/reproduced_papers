"""Figures in the style of the paper's Figs. 2, 3a, 3b and 6, drawn from run artifacts."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .display_names import model_label  # noqa: E402

# Result-scope labels from the scientific-validity policy, stamped on every figure.
SCOPE_LABELS = (
    "paper-accurate",
    "partial RED-tier run",
    "preliminary",
    "substitute-data",
    "synthetic-data",
)


def _stamp(figure, scope: str | None) -> None:
    if scope is None:
        return
    if scope not in SCOPE_LABELS:
        raise ValueError(
            f"Unknown result scope {scope!r}; expected one of {SCOPE_LABELS}"
        )
    figure.text(0.99, 0.01, scope, ha="right", va="bottom", fontsize=8, color="gray")


def _label(name: str, entry: dict) -> str:
    """Readable model name, using the run's recorded mode count for photonic models."""
    return model_label(
        name, detailed=True, modes=entry.get("hardware", {}).get("modes")
    )


def plot_spectra(
    run_dir: Path, metrics: dict, bins: int, output: Path, scope: str | None = None
) -> None:
    """Fig. 2 style: average normalized histogram of F_hat eigenvalues per model.

    Counts are averaged over the sampled parameter sets of every draw and divided by
    ``d``, so each panel's bars sum to 1 (the paper's "normalised counts").
    """
    names = list(metrics["models"])
    figure, axes = plt.subplots(
        1, len(names), figsize=(4 * len(names), 3.2), squeeze=False
    )
    for axis, name in zip(axes[0], names):
        with np.load(run_dir / metrics["models"][name]["arrays"]) as arrays:
            eigenvalues = arrays["eigenvalues"].reshape(
                -1, arrays["eigenvalues"].shape[-1]
            )
        upper = eigenvalues.max()
        counts = np.mean(
            [
                np.histogram(row, bins=bins, range=(0.0, upper))[0]
                for row in eigenvalues
            ],
            axis=0,
        )
        edges = np.linspace(0.0, upper, bins + 1)
        axis.bar(
            edges[:-1],
            counts / eigenvalues.shape[1],
            width=np.diff(edges),
            align="edge",
            edgecolor="black",
        )
        axis.set_title(
            f"{_label(name, metrics['models'][name])} (d={eigenvalues.shape[1]})"
        )
        axis.set_xlabel(f"eigenvalue size (bins = {bins})")
        axis.set_ylim(0.0, 1.0)
    axes[0][0].set_ylabel("normalised counts")
    figure.tight_layout()
    _stamp(figure, scope)
    figure.savefig(output, dpi=150)
    plt.close(figure)


def plot_effective_dimension(
    metrics: dict,
    output: Path,
    scope: str | None = None,
    reference: dict[str, tuple[list[float], list[float]]] | None = None,
) -> None:
    """Fig. 3a style: normalized effective dimension vs n, mean +- std over draws.

    ``reference`` maps a label to ``(n_values, values)`` drawn as dashed lines, for
    the paper's published curves.
    """
    n_values = np.asarray(metrics["n_values"])
    figure, axis = plt.subplots(figsize=(6, 4))
    for name, entry in metrics["models"].items():
        mean = np.asarray(entry["normalized_effective_dimension_mean"])
        std = np.asarray(entry["normalized_effective_dimension_std"])
        axis.plot(
            n_values,
            mean,
            marker="o",
            markersize=3,
            label=f"{_label(name, entry)} (d={entry['d']})",
        )
        axis.fill_between(n_values, mean - std, mean + std, alpha=0.2)
    for label, (reference_n, values) in (reference or {}).items():
        axis.plot(reference_n, values, linestyle="--", color="gray", linewidth=1)
        axis.annotate(label, (reference_n[-1], values[-1]), fontsize=7, color="gray")
    axis.set_xscale("log")
    axis.set_xlabel("number of data n")
    axis.set_ylabel(r"normalised effective dimension $d_{\gamma,n}/d$")
    axis.set_ylim(top=1.0)
    axis.legend()
    figure.tight_layout()
    _stamp(figure, scope)
    figure.savefig(output, dpi=150)
    plt.close(figure)


def plot_training_losses(
    metrics: dict,
    output: Path,
    scope: str | None = None,
    reference: dict[str, float] | None = None,
) -> None:
    """Fig. 3b style: mean training loss +- std over seeds per model.

    ``reference`` maps a label to the paper's final average loss (Table 1), drawn
    as a marker at the last iteration.
    """
    figure, axis = plt.subplots(figsize=(6, 4))
    for name, entry in metrics["models"].items():
        mean = np.asarray(entry["loss_curve_mean"])
        std = np.asarray(entry["loss_curve_std"])
        steps = np.arange(len(mean))
        axis.plot(steps, mean, label=f"{_label(name, entry)} (d={entry['d']})")
        axis.fill_between(steps, mean - std, mean + std, alpha=0.15)
    for label, value in (reference or {}).items():
        axis.scatter([steps[-1]], [value], marker="x", color="black", zorder=5)
        axis.annotate(
            label,
            (steps[-1], value),
            fontsize=7,
            xytext=(-60, 4),
            textcoords="offset points",
        )
    axis.set_xlabel("training iteration")
    axis.set_ylabel("cross-entropy loss")
    axis.legend()
    figure.tight_layout()
    _stamp(figure, scope)
    figure.savefig(output, dpi=150)
    plt.close(figure)


def plot_generalisation(
    metrics: dict,
    output: Path,
    scope: str | None = None,
    reference: tuple[list[float], list[float], list[float]] | None = None,
) -> None:
    """Fig. 6 style: normalized effective dimension at the trained theta vs randomization."""
    fractions = [entry["fraction"] for entry in metrics["fractions"]]
    mean = np.asarray(
        [entry["normalized_effective_dimension_mean"] for entry in metrics["fractions"]]
    )
    std = np.asarray(
        [entry["normalized_effective_dimension_std"] for entry in metrics["fractions"]]
    )
    test_loss = [entry["test_loss_mean"] for entry in metrics["fractions"]]
    figure, (left, right) = plt.subplots(1, 2, figsize=(9, 3.5))
    left.errorbar(
        fractions, mean, yerr=std, marker="o", capsize=3, label="this reproduction"
    )
    if reference is not None:
        reference_fractions, reference_mean, reference_std = reference
        left.errorbar(
            reference_fractions,
            reference_mean,
            yerr=reference_std,
            marker="s",
            capsize=3,
            linestyle="--",
            color="gray",
            label="authors' cached data",
        )
        left.legend(fontsize=7)
    left.set_xlabel("fraction of labels overwritten at random")
    left.set_ylabel("normalised effective dimension")
    right.plot(fractions, test_loss, marker="o")
    right.set_xlabel("fraction of labels overwritten at random")
    right.set_ylabel("test loss")
    figure.tight_layout()
    _stamp(figure, scope)
    figure.savefig(output, dpi=150)
    plt.close(figure)


def plot_sensitivity(
    metrics: dict, n_index: int, output: Path, scope: str | None = None
) -> None:
    """Fig. 8 style: normalized effective dimension at one n vs the number of samples."""
    figure, axis = plt.subplots(figsize=(6, 4))
    n_value = metrics["n_values"][n_index]
    for name, entry in metrics["models"].items():
        mean = np.asarray(entry["normalized_effective_dimension_mean"])[:, n_index]
        std = np.asarray(entry["normalized_effective_dimension_std"])[:, n_index]
        axis.errorbar(
            entry["sample_sizes"],
            mean,
            yerr=std,
            marker="o",
            capsize=3,
            label=f"{_label(name, entry)} (d={entry['d']})",
        )
    axis.set_xlabel("number of parameter and data samples")
    axis.set_ylabel(f"normalised effective dimension (n = {n_value:.0e})")
    axis.legend()
    figure.tight_layout()
    _stamp(figure, scope)
    figure.savefig(output, dpi=150)
    plt.close(figure)
