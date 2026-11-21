"""Plot the figures of a completed run directory.

Usage (from the repository root)::

    python3 papers/photonic_trainability/utils/plot_results.py <RUN_DIR> \
        --name fig4 --label paper-accurate

The script reads ``metrics.json`` to find the experiment type and the CSV
tables it wrote, and saves PNG figures in the run directory and in the paper's
``results/`` folder as ``<name>_<figure>.png``. ``--label`` is the result-scope
label from the scientific-validity policy (``paper-accurate``, ``partial
RED-tier run``, ``substitute-data``, ``synthetic-data``, ``preliminary``,
``extension``); it is printed in every figure title. Every figure has a table
view: the CSV in the same run directory.

Styling follows the repository's data-visualisation rules: categorical hues in
a fixed order (reference palette, light mode), markers as secondary encoding,
one y-axis per panel, a recessive grid, and text in ink colours rather than
series colours.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

logger = logging.getLogger(__name__)

SERIES = [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#6250d6",
    "#e34948",
]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": TEXT_SECONDARY,
            "axes.labelcolor": TEXT_PRIMARY,
            "axes.titlecolor": TEXT_PRIMARY,
            "xtick.color": TEXT_SECONDARY,
            "ytick.color": TEXT_SECONDARY,
            "text.color": TEXT_PRIMARY,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.linewidth": 2.0,
            "lines.markersize": 5.0,
            "legend.frameon": False,
            "font.size": 10,
        }
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float(value: str) -> float:
    return float(value) if value not in ("", "None") else float("nan")


RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
_CONTEXT = {"name": "run", "label": "preliminary", "targets": []}


def _save(figure, out_dir: Path, name: str) -> Path:
    figure.suptitle(
        f"[{_CONTEXT['label']}]", fontsize=8, color=TEXT_SECONDARY, x=0.99, ha="right"
    )
    figure.tight_layout()
    filename = f"{_CONTEXT['name']}_{name}"
    paths = [target / filename for target in _CONTEXT["targets"]] or [
        out_dir / filename
    ]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=150)
        logger.info("FIGURE_WRITTEN | path=%s", path)
    plt.close(figure)
    return paths[0]


def plot_first_moment(run_dir: Path, out_dir: Path) -> list[Path]:
    """Paper Fig. 10: ``mu1`` versus weight; exact (solid) and M -> inf (dotted)."""
    rows = _read_csv(run_dir / "first_moment_table.csv")
    figure, axis = plt.subplots(figsize=(6, 4))
    by_density = defaultdict(list)
    for row in rows:
        by_density[float(row["density"])].append(row)
    for index, (density, group) in enumerate(sorted(by_density.items())):
        weights = [int(row["weight"]) for row in group]
        color, marker = SERIES[index], MARKERS[index]
        axis.semilogy(
            weights,
            [_float(r["mu1_exact"]) for r in group],
            color=color,
            marker=marker,
            label=f"nu = {density:g}, M = {group[0]['modes']}",
        )
        axis.semilogy(
            weights,
            [_float(r["mu1_limit"]) for r in group],
            color=color,
            linestyle=":",
            label=f"nu = {density:g}, M -> inf",
        )
    axis.set_xlabel("Weight |alpha|")
    axis.set_ylabel("First moment mu1(alpha)")
    axis.set_title("PNM first moment versus weight (K = 5)")
    axis.legend(fontsize=8)
    return [_save(figure, out_dir, "first_moment.png")]


def plot_variance_scaling(run_dir: Path, out_dir: Path) -> list[Path]:
    """Paper Figs. 3-4 style: sample and circuit variance versus M per observable."""
    rows = _read_csv(run_dir / "variance_table.csv")
    by_observable = defaultdict(list)
    for row in rows:
        by_observable[row["observable"]].append(row)
    figure, (left, right) = plt.subplots(1, 2, figsize=(12.5, 4.2))
    for index, (label, group) in enumerate(by_observable.items()):
        modes = [int(row["modes"]) for row in group]
        color, marker = SERIES[index % len(SERIES)], MARKERS[index % len(MARKERS)]
        left.semilogy(
            modes,
            [_float(r["sample_variance"]) for r in group],
            color=color,
            marker=marker,
            label=label,
        )
        left.axhline(
            _float(group[-1]["sample_variance_limit"]),
            color=color,
            linestyle=":",
            linewidth=1.0,
        )
        circuit = np.array([_float(r["circuit_variance"]) for r in group])
        positive = circuit > 0
        right.semilogy(
            np.array(modes)[positive],
            circuit[positive],
            color=color,
            marker=marker,
            label=label,
        )
    left.set_title("Sample variance (dotted: mu1(2 alpha) as M -> inf)")
    right.set_title("Circuit variance")
    all_modes = [int(row["modes"]) for row in rows]
    # Ranges well beyond the paper's are shown log-log, where a power law is a line.
    log_modes = min(all_modes) >= 100 and max(all_modes) / min(all_modes) >= 5
    for axis in (left, right):
        axis.set_xlabel("Number of modes M" + (" (log scale)" if log_modes else ""))
        if log_modes:
            axis.set_xscale("log")
    left.set_ylabel("Sample variance")
    right.set_ylabel("Circuit variance")
    right.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    return [_save(figure, out_dir, "variance_scaling.png")]


def plot_pnp_variances(run_dir: Path, out_dir: Path) -> list[Path]:
    """Paper Fig. 5 style: per-instance crosses and the mean line versus M."""
    rows = _read_csv(run_dir / "pnp_variance_table.csv")
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for axis, key, title, color in (
        (axes[0], "sample_variance", "Sample variance", SERIES[1]),
        (axes[1], "circuit_variance", "Circuit variance", SERIES[0]),
    ):
        modes = np.array([int(row["modes"]) for row in rows])
        values = np.array([_float(row[key]) for row in rows])
        axis.scatter(
            modes, values, marker="x", color=color, alpha=0.45, s=18, label="instances"
        )
        grid = sorted(set(modes))
        axis.plot(
            grid,
            [values[modes == m].mean() for m in grid],
            color=color,
            marker="o",
            label="average",
        )
        axis.set_xlabel("Number of modes M")
        axis.set_ylabel(title)
        axis.set_title(title)
        axis.legend(fontsize=8)
    return [_save(figure, out_dir, "pnp_variances.png")]


def plot_output_probability(run_dir: Path, out_dir: Path) -> list[Path]:
    """Projector variance ratio versus Hilbert dimension, with scaling references."""
    rows = _read_csv(run_dir / "projector_table.csv")
    dimension = np.array([_float(r["hilbert_dimension"]) for r in rows])
    order = np.argsort(dimension)
    figure, axis = plt.subplots(figsize=(6, 4))
    for index, (key, label) in enumerate(
        (
            ("variance_ratio", "exact ratio (averaged over outcomes)"),
            ("mhiri_scaling_dim_over_M_over_n_plus_1", "|H| / (M/n + 1)"),
            (
                "paper_scaling_dim_over_n_plus_n_over_M_plus_1",
                "|H| / (n + n/M + 1) (paper Eq. 23)",
            ),
        )
    ):
        values = np.array([_float(r[key]) for r in rows])
        axis.loglog(
            dimension[order],
            values[order],
            color=SERIES[index],
            marker=MARKERS[index],
            linestyle="-" if index == 0 else "--",
            label=label,
        )
    axis.set_xlabel("Hilbert dimension |H|")
    axis.set_ylabel("Var_samp / Var_circ")
    axis.set_title("Single-outcome projectors")
    axis.legend(fontsize=8)
    return [_save(figure, out_dir, "projector_ratio.png")]


def plot_sample_complexity(run_dir: Path, out_dir: Path) -> list[Path]:
    """Ordering success versus N / ratio, with Prop. 1's guarantee."""
    rows = _read_csv(run_dir / "ordering_table.csv")
    by_case = defaultdict(list)
    for row in rows:
        by_case[row["case"]].append(row)
    figure, axis = plt.subplots(figsize=(6.5, 4))
    for index, (case, group) in enumerate(by_case.items()):
        factors = [_float(r["shot_factor"]) for r in group]
        axis.semilogx(
            factors,
            [_float(r["success_resolvable_pairs"]) for r in group],
            color=SERIES[index],
            marker=MARKERS[index],
            label=case,
        )
    first = next(iter(by_case.values()))
    axis.semilogx(
        [_float(r["shot_factor"]) for r in first],
        [_float(r["prop1_guarantee_resolvable"]) for r in first],
        color=TEXT_SECONDARY,
        linestyle="--",
        label="Prop. 1 guarantee",
    )
    axis.set_xlabel("Shots N / (Var_samp / Var_circ)")
    axis.set_ylabel("P(correct ordering), resolvable pairs")
    axis.set_ylim(0.0, 1.02)
    axis.set_title("Loss ordering with N shots")
    axis.legend(fontsize=8)
    return [_save(figure, out_dir, "ordering_success.png")]


def plot_training(run_dir: Path, out_dir: Path) -> list[Path]:
    """Paper Figs. 7 and 9 style: exact loss per iteration and parameter trajectories."""
    rows = _read_csv(run_dir / "losses.csv")
    by_method = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_method[row["method"]][int(row["seed"])].append(_float(row["exact_loss"]))
    figure, axis = plt.subplots(figsize=(6.5, 4))
    for index, (method, seeds) in enumerate(by_method.items()):
        curves = np.array(list(seeds.values()))
        iterations = np.arange(1, curves.shape[1] + 1)
        mean = curves.mean(axis=0)
        axis.plot(
            iterations,
            mean,
            color=SERIES[index],
            label=f"{method} (mean of {len(curves)})",
        )
        if len(curves) > 1:
            spread = curves.std(axis=0, ddof=1)
            axis.fill_between(
                iterations,
                mean - spread,
                mean + spread,
                color=SERIES[index],
                alpha=0.15,
                linewidth=0,
            )
    axis.set_xlabel("Iteration")
    axis.set_ylabel("Exact loss")
    axis.set_title("Training loss (band: +/- 1 std over seeds)")
    axis.legend(fontsize=8)
    paths = [_save(figure, out_dir, "training_loss.png")]
    trajectory_path = run_dir / "trajectories.csv"
    if trajectory_path.exists():
        rows = [r for r in _read_csv(trajectory_path) if r["method"] == "gpsr_shots"]
        if rows:
            figure, axis = plt.subplots(figsize=(6.5, 4))
            by_parameter = defaultdict(list)
            for row in rows:
                by_parameter[int(row["parameter"])].append(
                    (int(row["iteration"]), _float(row["value"]))
                )
            for index, points in enumerate(by_parameter.values()):
                iterations, values = zip(*points)
                axis.plot(iterations, values, color=SERIES[index % 3], linewidth=1.2)
            axis.set_xlabel("Iteration")
            axis.set_ylabel("Parameter value (wrapped to [-pi, pi])")
            axis.set_title(
                "Trajectories of 15 random parameters (gpsr_shots, first seed)"
            )
            axis.grid(True)
            paths.append(_save(figure, out_dir, "training_trajectories.png"))
    return paths


def plot_merlin_variance(run_dir: Path, out_dir: Path) -> list[Path]:
    """Phase 04: variance ratio versus M per hardware condition, one panel per observable."""
    rows = _read_csv(run_dir / "merlin_variance_table.csv")
    observables = list(dict.fromkeys(row["observable"] for row in rows))
    conditions = list(dict.fromkeys(row["condition"] for row in rows))
    figure, axes = plt.subplots(
        1, len(observables), figsize=(5.2 * len(observables) + 2.2, 4.2)
    )
    for axis, observable in zip(np.atleast_1d(axes), observables):
        for index, condition in enumerate(conditions):
            group = sorted(
                (
                    r
                    for r in rows
                    if r["observable"] == observable and r["condition"] == condition
                ),
                key=lambda r: int(r["modes"]),
            )
            modes = [int(r["modes"]) for r in group]
            ratio = np.array([_float(r["variance_ratio"]) for r in group])
            # Batch-means errors propagated to the ratio (first order).
            relative = np.sqrt(
                (
                    np.array([_float(r["sample_variance_se"]) for r in group])
                    / np.array([_float(r["sample_variance"]) for r in group])
                )
                ** 2
                + (
                    np.array([_float(r["circuit_variance_se"]) for r in group])
                    / np.array([_float(r["circuit_variance"]) for r in group])
                )
                ** 2
            )
            axis.errorbar(
                modes,
                ratio,
                yerr=ratio * relative,
                color=SERIES[index % len(SERIES)],
                marker=MARKERS[index % len(MARKERS)],
                capsize=3,
                label=condition,
            )
        axis.set_yscale("log")
        axis.set_xlabel("Number of modes M")
        axis.set_ylabel("Var_samp / Var_circ")
        axis.set_title(observable)
    np.atleast_1d(axes)[-1].legend(
        fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0)
    )
    return [_save(figure, out_dir, "hardware_variance_ratio.png")]


def plot_estimator_scaling(run_dir: Path, out_dir: Path) -> list[Path]:
    """Phase 05 C9: samples versus |A| and classical wall-clock versus |A| (separate axes)."""
    rows = _read_csv(run_dir / "estimator_cases.csv")
    families = list(dict.fromkeys(r["family"] for r in rows))
    figure, (left, right) = plt.subplots(1, 2, figsize=(12.5, 4.4))
    series = []
    for family in families:
        series.append((f"quantum shots, {family}", family, "n_quantum", "-"))
        series.append((f"Lim-Oh samples (B1), {family}", family, "n_lim_oh", "--"))
    for index, (label, family, column, style) in enumerate(series):
        sizes = sorted({int(r["terms"]) for r in rows})
        means, errors = [], []
        for size in sizes:
            values = np.array(
                [
                    float(r[column])
                    for r in rows
                    if r["family"] == family and int(r["terms"]) == size
                ]
            )
            means.append(values.mean())
            errors.append(values.std(ddof=1))
        left.errorbar(
            sizes,
            means,
            yerr=errors,
            color=SERIES[index],
            marker=MARKERS[index],
            linestyle=style,
            capsize=3,
            label=label,
        )
    left.set_xscale("log")
    left.set_yscale("log")
    left.set_xlabel("Number of PNP terms |A| (log scale)")
    left.set_ylabel("Samples for additive error eps = 0.1")
    left.set_title("Samples needed (mean ± std, 3 U x 5 PNPs)")
    left.legend(fontsize=8)
    for index, (label, column) in enumerate(
        (
            ("B1 Lim-Oh (N_LO x measured time/sample)", "lim_oh_classical_seconds"),
            ("B2 exact evaluation (permanents)", "b2_exact_seconds"),
        )
    ):
        sizes = sorted({int(r["terms"]) for r in rows})
        means = [
            np.mean([float(r[column]) for r in rows if int(r["terms"]) == size])
            for size in sizes
        ]
        right.plot(
            sizes,
            means,
            color=SERIES[4 + index],
            marker=MARKERS[4 + index],
            label=label,
        )
    right.set_xscale("log")
    right.set_yscale("log")
    right.set_xlabel("Number of PNP terms |A| (log scale)")
    right.set_ylabel("Classical wall-clock (s, Apple M2)")
    right.set_title("Classical cost of the same loss (both families)")
    right.legend(fontsize=8)
    return [_save(figure, out_dir, "estimator_scaling.png")]


PLOTTERS = {
    "first_moment": plot_first_moment,
    "variance_scaling": plot_variance_scaling,
    "pnp_variances": plot_pnp_variances,
    "output_probability": plot_output_probability,
    "sample_complexity": plot_sample_complexity,
    "training": plot_training,
    "merlin_variance": plot_merlin_variance,
    "estimator_scaling": plot_estimator_scaling,
}


def main(argv: list[str] | None = None) -> int:
    """Plot one run directory; see the module docstring."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--name", required=True, help="Figure-set name, e.g. fig4")
    parser.add_argument(
        "--label", required=True, help="Result-scope label for the titles"
    )
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    metrics = json.loads((args.run_dir / "metrics.json").read_text(encoding="utf-8"))
    experiment = metrics["experiment"]
    if experiment not in PLOTTERS:
        raise SystemExit(f"No plotter for experiment {experiment!r}")
    _CONTEXT.update(
        name=args.name, label=args.label, targets=[args.run_dir, args.results_dir]
    )
    _style()
    PLOTTERS[experiment](args.run_dir, args.run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
