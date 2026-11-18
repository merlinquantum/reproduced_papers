"""Figure helpers for the reproduction (paper Figs. 2-5 analogues).

Every figure carries a result-scope label in its title or footer, as required by
the scientific-validity policy (e.g. ``paper-accurate``, ``preliminary``).
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

COLORS = {
    "ff": "#c0392b",
    "tf": "#2471a3",
    "ffs": "#7d3c98",
    "mlp": "#7f8c8d",
    "fourier": "#16a085",
}


FALLBACK_COLORS = ("#e67e22", "#27ae60", "#8e44ad", "#2c3e50", "#d35400", "#1abc9c")


def color_for(candidate: str, index: int) -> str:
    """Use the family colour for a plain family name, a distinct colour otherwise."""
    if candidate in COLORS:
        return COLORS[candidate]
    return FALLBACK_COLORS[index % len(FALLBACK_COLORS)]


def completed_runs(sweep_dir: Path) -> dict[str, dict[int, Path]]:
    """Return ``{candidate: {seed: run_dir}}`` for completed runs of a sweep."""
    status = json.loads((sweep_dir / "sweep_status.json").read_text(encoding="utf-8"))
    runs: dict[str, dict[int, Path]] = {}
    for entry in status["runs"]:
        if entry["status"] == "COMPLETED":
            runs.setdefault(entry["candidate"], {})[entry["seed"]] = (
                sweep_dir / entry["run_dir"]
            )
    return runs


def read_metrics(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))


def median_seed(runs: dict[int, Path], metric: str) -> int:
    """Return the seed whose ``metric`` is the (lower) median over seeds."""
    values = sorted((read_metrics(path)[metric], seed) for seed, path in runs.items())
    return values[(len(values) - 1) // 2][1]


def save_figure(figure, name: str, destinations: list[Path]) -> list[Path]:
    """Save ``figure`` as PNG in every destination folder."""
    written = []
    for folder in destinations:
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{name}.png"
        figure.savefig(path, dpi=160, bbox_inches="tight")
        written.append(path)
    plt.close(figure)
    return written


def plot_fit_and_spectrum(
    runs: dict[str, dict[int, Path]],
    candidates: list[str],
    data_frequencies: list[float],
    scope_label: str,
) -> Any:
    """Fig. 2 analogue: median-seed predictions and their DFT spectra."""
    figure, (fit_axis, spectrum_axis) = plt.subplots(1, 2, figsize=(13, 4.2))
    first = True
    for index, candidate in enumerate(candidates):
        seed = median_seed(runs[candidate], "train_mse")
        run_dir = runs[candidate][seed]
        predictions = np.load(run_dir / "predictions.npz")
        metrics = read_metrics(run_dir)
        color = color_for(candidate, index)
        if first:
            fit_axis.plot(
                predictions["train_x"], predictions["train_y"], "k.", ms=3, label="data"
            )
            first = False
        fit_axis.plot(
            predictions["dense_x"],
            predictions["dense_prediction"],
            color=color,
            label=f"{candidate.upper()} (seed {seed}, MSE {metrics['train_mse']:.2g})",
        )
        spectrum_path = run_dir / "spectrum.npz"
        if spectrum_path.exists():
            spectrum = np.load(spectrum_path)
            keep = spectrum["angular_frequency"] <= 1.5 * max(data_frequencies) + 0.5
            spectrum_axis.plot(
                spectrum["angular_frequency"][keep],
                spectrum["magnitude"][keep],
                color=color,
                label=candidate.upper(),
            )
    for frequency in data_frequencies:
        spectrum_axis.axvline(frequency, color="#5dade2", ls="--", lw=1)
    fit_axis.set(xlabel="x", ylabel="y(x)", title=f"Fit to Ω_d = {data_frequencies}")
    fit_axis.legend(fontsize=8, loc="upper right")
    spectrum_axis.set(
        xlabel="angular frequency ω",
        ylabel="|DFT| / n",
        title="Spectrum of the trained model (median seed)",
    )
    spectrum_axis.legend(fontsize=8)
    figure.suptitle(scope_label, fontsize=9, y=1.02)
    return figure


def plot_metric_by_candidate(
    values: dict[str, list[float]],
    ylabel: str,
    title: str,
    scope_label: str,
    log_scale: bool = True,
) -> Any:
    """Strip plot of per-seed values with mean markers for each candidate."""
    figure, axis = plt.subplots(figsize=(max(5, 1.1 * len(values)), 4))
    for index, (candidate, series) in enumerate(values.items()):
        jitter = np.linspace(-0.12, 0.12, len(series)) if len(series) > 1 else [0.0]
        color = color_for(candidate, index)
        axis.scatter(index + np.asarray(jitter), series, color=color, alpha=0.7, s=18)
        axis.scatter([index], [statistics.fmean(series)], color="k", marker="_", s=400)
    axis.set_xticks(range(len(values)), list(values), rotation=30, ha="right")
    axis.set(ylabel=ylabel, title=title)
    if log_scale:
        axis.set_yscale("log")
    figure.text(0.01, -0.08, scope_label, fontsize=8)
    return figure


def plot_fig3_boxes(per_dataset: dict[str, dict[int, float]], scope_label: str) -> Any:
    """Fig. 3 analogue: one box per model over the datasets (seed-averaged MSE)."""
    figure, axis = plt.subplots(figsize=(7, 4.2))
    models = list(per_dataset)
    data = [list(per_dataset[m].values()) for m in models]
    axis.boxplot(
        data, showmeans=True, meanprops={"marker": "^", "markerfacecolor": "green"}
    )
    axis.set_xticks(
        range(1, len(models) + 1),
        [m.upper() if m in ("tf", "ffs") else m for m in models],
    )
    axis.set_yscale("log")
    axis.set(
        ylabel="train MSE (mean over seeds)",
        title="Spectral richness: 7 datasets, |Ω_d| = 1..7 in [1, 3]",
    )
    figure.text(0.01, -0.06, scope_label, fontsize=8)
    return figure


def plot_pressure_fields(
    reference: np.ndarray,
    predictions: dict[str, np.ndarray],
    x,
    y,
    time_value,
    scope_label,
) -> Any:
    """Fig. 4 analogue: reference and predicted pressure on the 100 x 50 grid."""
    panels = {"reference (FEM)": reference, **predictions}
    figure, axes = plt.subplots(
        1, len(panels), figsize=(4.4 * len(panels), 2.8), sharey=True
    )
    # Shared levels from the reference range; predictions outside it take the
    # end colours (extend="both") so every panel uses the same colour scale.
    levels = np.linspace(reference.min(), reference.max(), 41)
    reference_image = None
    for axis, (title, field) in zip(np.atleast_1d(axes), panels.items()):
        image = axis.tricontourf(
            x, y, field, levels=levels, cmap="RdBu_r", extend="both"
        )
        reference_image = reference_image or image
        axis.set(title=title, xlabel="x")
        axis.set_aspect("equal")
    np.atleast_1d(axes)[0].set_ylabel("y")
    figure.colorbar(reference_image, ax=axes, shrink=0.8, label="p")
    figure.suptitle(
        f"Pressure at t = {time_value}  ·  {scope_label}", fontsize=9, y=1.04
    )
    return figure


def plot_maerm_over_time(
    per_candidate: dict[str, dict[int, list[dict[str, Any]]]], scope_label: str
) -> Any:
    """MAERM of u, v, p against time: mean over seeds with min/max band."""
    figure, axes = plt.subplots(1, 3, figsize=(14, 3.6))
    for index, (candidate, by_seed) in enumerate(per_candidate.items()):
        color = color_for(candidate, index)
        for axis, field in zip(axes, ("u", "v", "p")):
            times = [row["t"] for row in next(iter(by_seed.values()))]
            values = np.array(
                [
                    [row[f"{field}_maerm_abs_median"] for row in rows]
                    for rows in by_seed.values()
                ]
            )
            axis.plot(
                times,
                values.mean(0),
                color=color,
                label=f"{candidate} (n={len(values)})",
            )
            axis.fill_between(
                times, values.min(0), values.max(0), color=color, alpha=0.15
            )
            axis.set(title=f"{field}: MAERM (%), median(|y|) denominator", xlabel="t")
    axes[0].legend(fontsize=8)
    figure.suptitle(scope_label, fontsize=9, y=1.03)
    return figure
