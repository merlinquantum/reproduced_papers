"""Figures built from ``utils/compare_results.py``'s ``comparison.json``.

Every figure reads only the structured comparison artifact, carries its
result-scope label in the title, and is saved by the ``utils/plot_*.py``
scripts to both the comparison run directory and ``results/``.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.lines  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402

# Reference categorical palette, slots 1-4 in fixed order (documented as
# passing the adjacent-pair CVD and normal-vision checks on the light
# surface). Slots 3-4 sit below 3:1 contrast, so every series is also
# direct-labelled and the tables in comparison.md give the values.
SERIES_COLOURS = {
    "qultsf": "#2a78d6",
    "linear": "#eb6834",
    "nlinear": "#1baf7a",
    "dlinear": "#eda100",
}
SERIES_LABELS = {
    "qultsf": "QuLTSF",
    "linear": "Linear",
    "nlinear": "NLinear",
    "dlinear": "DLinear",
    "persistence": "Persistence",
    "identity_circuit": "No circuit (identity)",
    "frozen_circuit": "Frozen random circuit",
    "rank_bottleneck": "Rank-10 linear bottleneck",
    "mlp": "Parameter-matched MLP",
    "qultsf_merlin_e20": "MerLin, trained mesh (1 seed, 20 ep)",
    "identity_interferometer_merlin": "MerLin, no interferometer",
}
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
NEUTRAL = "#8a8985"
GRID = "#e6e5e1"
MODELS = ("qultsf", "linear", "nlinear", "dlinear")


def _style_axes(axes) -> None:
    axes.set_facecolor(SURFACE)
    axes.grid(axis="y", color=GRID, linewidth=0.8)
    axes.set_axisbelow(True)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(NEUTRAL)
    axes.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    axes.yaxis.label.set_color(TEXT_SECONDARY)
    axes.xaxis.label.set_color(TEXT_SECONDARY)


def _new_figure(n_panels: int, width: float = 6.4):
    figure, axes = plt.subplots(
        1, n_panels, figsize=(width * n_panels, 4.2), facecolor=SURFACE, squeeze=False
    )
    for panel in axes[0]:
        _style_axes(panel)
    return figure, list(axes[0])


def _end_labels(axes, x, values: dict[str, float], min_gap_fraction=0.045) -> None:
    """Label line ends, spreading labels vertically so they never overlap."""
    low, high = axes.get_ylim()
    gap = (high - low) * min_gap_fraction
    placed: list[tuple[str, float, float]] = []
    for name, value in sorted(values.items(), key=lambda item: item[1]):
        label_y = value if not placed else max(value, placed[-1][2] + gap)
        placed.append((name, value, label_y))
    for name, _, label_y in placed:
        axes.annotate(
            SERIES_LABELS[name],
            (x, label_y),
            xytext=(12, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=TEXT_PRIMARY,
            annotation_clip=False,
        )


def plot_table1(comparison: dict):
    """Test MSE against horizon, ours (mean ± std) and the paper's values.

    Parameters
    ----------
    comparison : dict
        Parsed ``comparison.json``.

    Returns
    -------
    matplotlib.figure.Figure
        One panel; filled markers are ours, hollow markers the paper's.
    """
    figure, (axes,) = _new_figure(1, width=7.2)
    horizons = [block["T"] for block in comparison["table1"]]
    positions = range(len(horizons))
    last_values: dict[str, float] = {}
    for name in MODELS:
        means = [block["rows"][name]["mse_mean"] for block in comparison["table1"]]
        stds = [block["rows"][name]["mse_std"] or 0.0 for block in comparison["table1"]]
        paper = [block["rows"][name]["paper"][0] for block in comparison["table1"]]
        colour = SERIES_COLOURS[name]
        axes.plot(
            positions, means, color=colour, linewidth=2, marker="o", markersize=8,
            markeredgecolor=SURFACE, markeredgewidth=2,
            label=f"{SERIES_LABELS[name]} (ours)",
        )  # fmt: skip
        axes.errorbar(
            positions, means, yerr=stds, fmt="none", ecolor=colour, capsize=3,
            linewidth=1,
        )  # fmt: skip
        axes.scatter(
            positions, paper, facecolors="none", edgecolors=colour, s=110,
            linewidths=1.5, zorder=3,
        )  # fmt: skip
        last_values[name] = means[-1]
    axes.scatter(
        [], [], facecolors="none", edgecolors=NEUTRAL, s=90, label="Paper Table 1"
    )
    axes.set_xticks(list(positions), [str(h) for h in horizons])
    axes.set_xlim(-0.3, len(horizons) - 0.3)
    _end_labels(axes, len(horizons) - 1, last_values)
    axes.set_xlabel("Forecast horizon T (steps)")
    axes.set_ylabel("Test MSE (standardised)")
    axes.set_title(
        "Weather, L = 336: QuLTSF vs linear baselines\n"
        + textwrap.fill(comparison["labels"]["table1"], width=78),
        fontsize=10, color=TEXT_PRIMARY, loc="left",
    )  # fmt: skip
    axes.legend(frameon=False, fontsize=9, labelcolor=TEXT_SECONDARY, loc="upper left")
    figure.tight_layout()
    return figure


def plot_lookback(comparison: dict):
    """Test MSE against lookback L, one panel per horizon (paper Figs 2-3).

    Parameters
    ----------
    comparison : dict
        Parsed ``comparison.json``.

    Returns
    -------
    matplotlib.figure.Figure
        One panel per horizon present in the lookback table.
    """
    horizons = sorted({block["T"] for block in comparison["lookback"]})
    figure, panels = _new_figure(len(horizons))
    for axes, horizon in zip(panels, horizons, strict=True):
        blocks = sorted(
            (b for b in comparison["lookback"] if b["T"] == horizon),
            key=lambda b: b["L"],
        )
        lookbacks = [block["L"] for block in blocks]
        last_values: dict[str, float] = {}
        for name in MODELS:
            values = [block["rows"][name]["mse_mean"] for block in blocks]
            last_values[name] = values[-1]
            axes.plot(
                lookbacks, values, color=SERIES_COLOURS[name], linewidth=2, marker="o",
                markersize=8, markeredgecolor=SURFACE, markeredgewidth=2,
                label=SERIES_LABELS[name],
            )  # fmt: skip
        axes.set_xticks(lookbacks, [str(length) for length in lookbacks])
        axes.set_xlim(lookbacks[0] - 30, lookbacks[-1] + 130)
        _end_labels(axes, lookbacks[-1], last_values)
        axes.set_xlabel("Lookback window L (steps)")
        axes.set_ylabel("Test MSE (standardised)")
        axes.set_title(
            f"T = {horizon}", fontsize=11, color=TEXT_PRIMARY, loc="left"
        )  # fmt: skip
    panels[0].legend(frameon=False, fontsize=9, labelcolor=TEXT_SECONDARY)
    figure.suptitle(
        f"Weather: test MSE vs lookback, {comparison['labels']['lookback']}",
        fontsize=11, color=TEXT_PRIMARY, x=0.01, ha="left",
    )  # fmt: skip
    figure.tight_layout()
    return figure


def plot_ablations(comparison: dict):
    """Controls' test MSE (mean ± std) against QuLTSF and the best baseline.

    A dot plot rather than bars: the values sit close together far from
    zero, and truncated bars would exaggerate the differences.

    Parameters
    ----------
    comparison : dict
        Parsed ``comparison.json``.

    Returns
    -------
    matplotlib.figure.Figure
        One panel per ablated horizon; QuLTSF and the best same-budget
        linear baseline are vertical reference lines.
    """
    blocks = comparison["ablations"]
    figure, panels = _new_figure(len(blocks))
    table1 = {block["T"]: block for block in comparison["table1"]}
    for axes, block in zip(panels, blocks, strict=True):
        names = list(block["rows"])
        means = [block["rows"][name]["mse_mean"] for name in names]
        stds = [block["rows"][name]["mse_std"] or 0.0 for name in names]
        positions = list(range(len(names)))
        reference = table1[block["T"]]
        quantum = reference["rows"]["qultsf"]["mse_mean"]
        best = reference["best_baseline"]
        baseline = reference["rows"][best]["mse_mean"]
        axes.axvline(
            quantum, color=SERIES_COLOURS["qultsf"], linewidth=2,
            label=f"QuLTSF {quantum:.4f}",
        )  # fmt: skip
        axes.axvline(
            baseline, color=SERIES_COLOURS[best], linewidth=2, linestyle="--",
            label=f"{SERIES_LABELS[best]} (best linear) {baseline:.4f}",
        )  # fmt: skip
        axes.errorbar(
            means, positions, xerr=stds, fmt="o", color=TEXT_SECONDARY,
            markersize=8, markeredgecolor=SURFACE, markeredgewidth=2,
            ecolor=TEXT_SECONDARY, elinewidth=1, capsize=3, zorder=3,
        )  # fmt: skip
        for position, mean in zip(positions, means, strict=True):
            axes.annotate(
                f"{mean:.4f}", (mean, position), xytext=(0, 9),
                textcoords="offset points", ha="center", fontsize=8,
                color=TEXT_SECONDARY, zorder=4,
                bbox={"boxstyle": "square,pad=0.15", "facecolor": SURFACE, "edgecolor": "none"},
            )  # fmt: skip
        axes.set_yticks(positions, [SERIES_LABELS[name] for name in names])
        axes.set_ylim(len(names) - 0.5, -0.7)
        span = max(max(m + e for m, e in zip(means, stds, strict=True)), baseline)
        floor = min(min(m - e for m, e in zip(means, stds, strict=True)), quantum)
        pad = (span - floor) * 0.08
        axes.set_xlim(floor - pad, span + pad)
        axes.grid(axis="y", visible=False)
        axes.grid(axis="x", color=GRID, linewidth=0.8)
        axes.set_xlabel("Test MSE (standardised)")
        axes.set_title(f"T = {block['T']}", fontsize=11, color=TEXT_PRIMARY, loc="left")
        axes.legend(
            frameon=False, fontsize=8, labelcolor=TEXT_SECONDARY, ncol=2,
            loc="upper left", bbox_to_anchor=(0.0, -0.18),
        )  # fmt: skip
    figure.suptitle(
        f"Weather, L = 336: ablations, {comparison['labels']['ablations']}",
        fontsize=11, color=TEXT_PRIMARY, x=0.01, ha="left",
    )  # fmt: skip
    figure.tight_layout()
    return figure


def plot_photonic(comparison: dict):
    """Photonic adaptation against QuLTSF, its qubit control and the best linear model.

    Parameters
    ----------
    comparison : dict
        Parsed ``comparison.json`` with a ``photonic`` section.

    Returns
    -------
    matplotlib.figure.Figure
        One dot-plot panel per horizon with photonic results; QuLTSF and the
        best same-budget linear model are vertical reference lines.
    """
    blocks = comparison["photonic"]
    figure, panels = _new_figure(len(blocks), width=4.8)
    table1 = {block["T"]: block for block in comparison["table1"]}
    qubit_control = {
        block["T"]: block["rows"]["identity_circuit"]
        for block in comparison["ablations"]
        if "identity_circuit" in block["rows"]
    }
    # Same rows in every panel, so the panels can be read across; a missing
    # model (not run at that horizon) leaves its row empty.
    row_names = (
        "identity_circuit",
        "qultsf_merlin_e20",
        "identity_interferometer_merlin",
    )
    row_labels = ["Qubit, no circuit (Phase 3)"] + [
        SERIES_LABELS[n] for n in row_names[1:]
    ]
    for axes, block in zip(panels, blocks, strict=True):
        rows = dict(block["rows"])
        if block["T"] in qubit_control:
            rows["identity_circuit"] = qubit_control[block["T"]]
        present = [(i, name) for i, name in enumerate(row_names) if name in rows]
        positions = [i for i, _ in present]
        means = [rows[name]["mse_mean"] for _, name in present]
        stds = [rows[name]["mse_std"] or 0.0 for _, name in present]
        reference = table1[block["T"]]
        quantum = reference["rows"]["qultsf"]["mse_mean"]
        best = reference["best_baseline"]
        baseline = reference["rows"][best]["mse_mean"]
        axes.axvline(
            quantum, color=SERIES_COLOURS["qultsf"], linewidth=2, label="QuLTSF"
        )
        axes.axvline(
            baseline, color=NEUTRAL, linewidth=2, linestyle="--",
            label="Best same-budget linear model (named in panel title)",
        )  # fmt: skip
        axes.errorbar(
            means, positions, xerr=stds, fmt="o", color=TEXT_SECONDARY,
            markersize=8, markeredgecolor=SURFACE, markeredgewidth=2,
            ecolor=TEXT_SECONDARY, elinewidth=1, capsize=3, zorder=3,
        )  # fmt: skip
        for position, mean in zip(positions, means, strict=True):
            axes.annotate(
                f"{mean:.4f}", (mean, position), xytext=(0, 9),
                textcoords="offset points", ha="center", fontsize=8,
                color=TEXT_SECONDARY, zorder=4,
                bbox={"boxstyle": "square,pad=0.15", "facecolor": SURFACE, "edgecolor": "none"},
            )  # fmt: skip
        axes.set_yticks(
            range(len(row_names)),
            row_labels if axes is panels[0] else [""] * len(row_names),
        )
        axes.set_ylim(len(row_names) - 0.5, -0.7)
        span = max(max(m + e for m, e in zip(means, stds, strict=True)), baseline)
        floor = min(min(m - e for m, e in zip(means, stds, strict=True)), quantum)
        pad = (span - floor) * 0.12
        axes.set_xlim(floor - pad, span + pad)
        axes.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
        axes.grid(axis="y", visible=False)
        axes.grid(axis="x", color=GRID, linewidth=0.8)
        axes.set_xlabel("Test MSE (standardised)")
        axes.set_title(
            f"T = {block['T']} (best linear: {SERIES_LABELS[best]} {baseline:.4f})",
            fontsize=10, color=TEXT_PRIMARY, loc="left",
        )  # fmt: skip
    panels[0].legend(
        frameon=False, fontsize=8, labelcolor=TEXT_SECONDARY, ncol=2,
        loc="upper left", bbox_to_anchor=(0.0, -0.18),
    )  # fmt: skip
    figure.suptitle(
        "Weather, L = 336: MerLin photonic adaptation (simulation)\n"
        + textwrap.fill(comparison["labels"]["merlin"], width=150),
        fontsize=10, color=TEXT_PRIMARY, x=0.01, ha="left",
    )  # fmt: skip
    figure.tight_layout()
    return figure


FAIR_ROW_LABELS = {
    "qultsf": "QuLTSF (reproduced)",
    "linear": "Linear",
    "nlinear": "NLinear",
    "dlinear": "DLinear",
    "linear_ltsf_budget": "Linear, LTSF budget",
    "nlinear_ltsf_budget": "NLinear, LTSF budget",
    "dlinear_ltsf_budget": "DLinear, LTSF budget",
    "persistence": "Persistence",
    "mlp": "Parameter-matched MLP",
    "rank_bottleneck": "Rank-10 bottleneck",
    "identity_circuit": "QuLTSF, no circuit",
    "frozen_circuit": "QuLTSF, frozen circuit",
    "identity_interferometer_merlin": "MerLin, no interferometer",
    "qultsf_merlin_e20": "MerLin, trained (1 seed, 20 ep)",
}


def plot_fair_comparison(table: dict):
    """Every locally run system against QuLTSF, with the paper's values.

    Parameters
    ----------
    table : dict
        Parsed ``fair_comparison.json``.

    Returns
    -------
    matplotlib.figure.Figure
        One dot-plot panel per horizon. Filled dots are local means with
        seed std; hollow circles are paper-reported values; the vertical line
        is the reproduced QuLTSF mean. Persistence is omitted (off-scale).
    """
    horizons = table["horizons"]
    names = [name for name in FAIR_ROW_LABELS if name != "persistence"]
    figure, panels = _new_figure(len(horizons), width=3.9)
    figure.set_size_inches(3.9 * len(horizons) + 1.8, 6.2)
    for axes, block in zip(panels, horizons, strict=True):
        local = {
            row["system"]: row
            for row in block["rows"]
            if row["provenance"] == "local" and row.get("successful_runs")
        }
        paper = {
            row["system"].split(" ")[0]: row["mse"]
            for row in block["rows"]
            if row["provenance"] == "paper-reported"
        }
        quantum = local["qultsf"]["mse_mean"]
        axes.axvline(quantum, color=SERIES_COLOURS["qultsf"], linewidth=2)
        values = []
        for position, name in enumerate(names):
            if name in paper:
                axes.scatter(
                    [paper[name]], [position], facecolors="none",
                    edgecolors=TEXT_SECONDARY, s=80, linewidths=1.2, zorder=3,
                )  # fmt: skip
                values.append(paper[name])
            if name not in local:
                continue
            row = local[name]
            colour = SERIES_COLOURS.get(name, TEXT_SECONDARY)
            axes.errorbar(
                [row["mse_mean"]], [position], xerr=[row["mse_std"] or 0.0],
                fmt="o", color=colour, markersize=7, markeredgecolor=SURFACE,
                markeredgewidth=1.5, ecolor=colour, elinewidth=1, capsize=3, zorder=4,
            )  # fmt: skip
            values.append(row["mse_mean"] + (row["mse_std"] or 0.0))
            values.append(row["mse_mean"] - (row["mse_std"] or 0.0))
        labels = []
        for name in names:
            row = local.get(name)
            params = f" ({row['trainable_parameters'] / 1000:.0f}k)" if row else ""
            labels.append(FAIR_ROW_LABELS[name] + params)
        axes.set_yticks(range(len(names)), labels, fontsize=8)
        axes.set_ylim(len(names) - 0.5, -0.7)
        low, high = min(values), max(values)
        pad = (high - low) * 0.08
        axes.set_xlim(low - pad, high + pad)
        axes.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
        axes.grid(axis="y", visible=False)
        axes.grid(axis="x", color=GRID, linewidth=0.8)
        axes.set_xlabel("Test MSE (standardised)")
        axes.set_title(f"T = {block['T']}", fontsize=11, color=TEXT_PRIMARY, loc="left")
    handles = [
        matplotlib.lines.Line2D([], [], color=SERIES_COLOURS["qultsf"], linewidth=2),
        matplotlib.lines.Line2D(
            [], [], marker="o", linestyle="none", color=TEXT_SECONDARY, markersize=7
        ),
        matplotlib.lines.Line2D(
            [], [], marker="o", linestyle="none", markerfacecolor="none",
            markeredgecolor=TEXT_SECONDARY, markersize=8,
        ),
    ]  # fmt: skip
    figure.legend(
        handles,
        [
            "Reproduced QuLTSF mean",
            "Local runs: mean ± seed std (3 seeds unless noted)",
            "Paper-reported value (Table 1; linear rows quoted from Zeng et al.)",
        ],
        frameon=False, fontsize=8, labelcolor=TEXT_SECONDARY, ncol=3,
        loc="lower left", bbox_to_anchor=(0.01, 0.0),
    )  # fmt: skip
    figure.suptitle(
        "Weather, L = 336: fair comparison on one pipeline (same budget and seeds "
        "unless labelled; MerLin rows are a simulated adaptation)",
        fontsize=10, color=TEXT_PRIMARY, x=0.01, ha="left",
    )  # fmt: skip
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    return figure


def save_figure(figure, name: str, run_dir: Path, results_dir: Path) -> list[Path]:
    """Save ``figure`` as PNG to the run directory and ``results/``.

    Parameters
    ----------
    figure : matplotlib.figure.Figure
        Figure to save.
    name : str
        File stem, e.g. ``"table1_mse"``.
    run_dir : pathlib.Path
        Comparison run directory (holds ``comparison.json``).
    results_dir : pathlib.Path
        Curated results directory.

    Returns
    -------
    list[pathlib.Path]
        Written file paths.
    """
    written = []
    for directory in (run_dir, results_dir):
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{name}.png"
        figure.savefig(path, dpi=150, facecolor=SURFACE)
        written.append(path)
    plt.close(figure)
    return written
