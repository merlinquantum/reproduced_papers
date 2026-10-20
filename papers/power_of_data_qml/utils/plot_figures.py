"""Plot the reproduced figures against the journal Source Data.

Usage (from the paper folder)::

    python utils/plot_figures.py --aggregate results/study_extended.json.gz \
        --label "partial: n<=16, 10 seeds" --protocol paper

Reads an aggregate written by ``utils/aggregate_study.py`` (never a hard-coded
run), and writes PNG figures to ``results/figures/`` and to ``figures/`` inside
every run directory the aggregate lists.

Encoding (fixed across figures; palette validated with the dataviz validator):
colour = model (blue QK, orange PQK, aqua best classical), marker shape repeats
the model for colour-vision safety; solid line with filled markers and a
+/-1 std band = this reproduction; dashed line with hollow markers = paper
(Source Data). Each figure carries its result-scope label.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parents[1]
sys.path[:0] = [str(REPO_ROOT), str(PROJECT_DIR)]

from lib.result_io import load_json  # noqa: E402

SERIES = {
    "quantum_kernel": ("Quantum kernel (QK)", "#2a78d6", "o"),
    "projected_quantum_kernel": ("Projected QK (PQK)", "#eb6834", "s"),
    "best_classical": ("Best classical ML", "#1baf7a", "^"),
    "prediction_error_bound": ("Bound s_K,λ (Eq. K.1)", "#2a78d6", "D"),
}
# Set in main() from the embeddings present in the aggregate (gate E1-E3 or photonic PA/PB).
EMBEDDINGS: tuple[str, ...] = ("E1", "E2", "E3")
TEXT_PRIMARY, TEXT_SECONDARY, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def style_axes(ax) -> None:
    ax.set_facecolor("#fcfcfb")
    ax.grid(True, color=GRID, linewidth=0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(TEXT_SECONDARY)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def select(rows, component, quantity, embedding, protocol, train_size=None):
    chosen = [
        r
        for r in rows
        if r["component"] == component
        and r["quantity"] == quantity
        and r["embedding"] == embedding
        and r["protocol"] == protocol
        and (train_size is None or r["train_size"] == train_size)
    ]
    return sorted(chosen, key=lambda r: r["n"])


def plot_series(ax, rows, series_key, label_suffix=""):
    if not rows:
        return
    name, color, marker = SERIES[series_key]
    n = [r["n"] for r in rows]
    mean = [r["mean"] for r in rows]
    std = [0.0 if r["std"] != r["std"] else r["std"] for r in rows]
    ax.plot(
        n,
        mean,
        color=color,
        marker=marker,
        markersize=4.5,
        linewidth=1.6,
        label=f"{name}{label_suffix} (ours)",
    )
    ax.fill_between(
        n,
        [m - s for m, s in zip(mean, std)],
        [m + s for m, s in zip(mean, std)],
        color=color,
        alpha=0.15,
        linewidth=0,
    )
    paper = [
        (r["n"], r["paper_value"]) for r in rows if r.get("paper_value") is not None
    ]
    if paper:
        ax.plot(
            *zip(*paper),
            color=color,
            marker=marker,
            markersize=4.5,
            linewidth=1.2,
            linestyle="--",
            markerfacecolor="none",
            label=f"{name}{label_suffix} (paper)",
        )


def finish(fig, axes, title, label, path_stem, out_dirs):
    handles, labels = axes.flat[0].get_legend_handles_labels()
    seen = dict(zip(labels, handles))
    fig.legend(
        seen.values(),
        seen.keys(),
        loc="lower center",
        ncol=min(4, len(seen)),
        fontsize=8,
        frameon=False,
        labelcolor=TEXT_PRIMARY,
    )
    fig.suptitle(f"{title}\n[{label}]", fontsize=10, color=TEXT_PRIMARY)
    fig.tight_layout(rect=(0, 0.08, 1, 0.93))
    for directory in out_dirs:
        directory.mkdir(parents=True, exist_ok=True)
        fig.savefig(directory / f"{path_stem}.png", dpi=150, facecolor="white")
    plt.close(fig)


def figure_dimension_geometry(rows, label, out_dirs):
    fig, axes = plt.subplots(
        2, len(EMBEDDINGS), figsize=(3.7 * len(EMBEDDINGS), 6.2), sharex=True
    )
    for col, embedding in enumerate(EMBEDDINGS):
        for row, (prefix, ylabel) in enumerate(
            (("d", "effective dimension d"), ("g", "geometric difference g"))
        ):
            ax = axes[row, col]
            style_axes(ax)
            for kernel in ("quantum_kernel", "projected_quantum_kernel"):
                plot_series(
                    ax,
                    select(
                        rows,
                        "dimension_geometry",
                        f"{prefix}_{kernel}",
                        embedding,
                        None,
                    ),
                    kernel,
                )
            ax.set_title(f"{embedding}", fontsize=9, color=TEXT_PRIMARY)
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=9, color=TEXT_PRIMARY)
            if row == 1:
                ax.set_xlabel("n (system size)", fontsize=9, color=TEXT_PRIMARY)
    finish(
        fig,
        axes,
        "Fig. 1a (journal Fig. 3a): d and g vs n, N = 800",
        label,
        "fig1a_dimension_geometry",
        out_dirs,
    )


def figure_prediction_error(rows, label, protocol, out_dirs):
    fig, axes = plt.subplots(
        2, len(EMBEDDINGS), figsize=(3.7 * len(EMBEDDINGS), 6.2), sharex=True
    )
    for col, embedding in enumerate(EMBEDDINGS):
        ax = axes[0, col]
        style_axes(ax)
        for key in ("quantum_kernel", "projected_quantum_kernel", "best_classical"):
            plot_series(
                ax, select(rows, "quantum_dataset", key, embedding, protocol, 600), key
            )
        ax.set_title(
            f"Dataset (Q, {embedding}) - regression", fontsize=9, color=TEXT_PRIMARY
        )
        if col == 0:
            ax.set_ylabel("prediction error (MAE)", fontsize=9, color=TEXT_PRIMARY)
        ax = axes[1, col]
        style_axes(ax)
        plot_series(
            ax,
            select(
                rows,
                "classical_dataset",
                f"quantum_kernel_{embedding}",
                None,
                protocol,
                600,
            ),
            "quantum_kernel",
        )
        plot_series(
            ax,
            select(
                rows,
                "classical_dataset",
                f"projected_quantum_kernel_{embedding}",
                None,
                protocol,
                600,
            ),
            "projected_quantum_kernel",
        )
        plot_series(
            ax,
            select(rows, "classical_dataset", "best_classical", None, protocol, 600),
            "best_classical",
        )
        ax.set_title(
            f"Dataset (C), quantum kernels with {embedding}",
            fontsize=9,
            color=TEXT_PRIMARY,
        )
        ax.set_xlabel("n (system size)", fontsize=9, color=TEXT_PRIMARY)
        if col == 0:
            ax.set_ylabel("classification error", fontsize=9, color=TEXT_PRIMARY)
    finish(
        fig,
        axes,
        f"Fig. 1b (journal Fig. 3b): prediction error, N = 600, protocol '{protocol}'",
        label,
        f"fig1b_prediction_error_{protocol}",
        out_dirs,
    )


def figure_engineered(rows, component, label, protocol, out_dirs):
    fig, axes = plt.subplots(
        2,
        len(EMBEDDINGS),
        figsize=(3.7 * len(EMBEDDINGS), 6.2),
        sharex=True,
        sharey=True,
    )
    for row, train_size in enumerate((100, 600)):
        for col, embedding in enumerate(EMBEDDINGS):
            ax = axes[row, col]
            style_axes(ax)
            for key in ("quantum_kernel", "projected_quantum_kernel", "best_classical"):
                plot_series(
                    ax,
                    select(rows, component, key, embedding, protocol, train_size),
                    key,
                )
            ax.axhline(0.5, color=TEXT_SECONDARY, linewidth=0.8, linestyle=":")
            ax.set_title(
                f"{embedding}, N = {train_size}", fontsize=9, color=TEXT_PRIMARY
            )
            if col == 0:
                ax.set_ylabel("test accuracy", fontsize=9, color=TEXT_PRIMARY)
            if row == 1:
                ax.set_xlabel("n (system size)", fontsize=9, color=TEXT_PRIMARY)
    reference = "PQK" if component == "engineered_pqk" else "QK"
    number = (
        "Fig. 2 (journal Fig. 4)"
        if component == "engineered_pqk"
        else "Fig. S1 (journal Supp. Fig. 2)"
    )
    finish(
        fig,
        axes,
        f"{number}: labels engineered against {reference}; dotted = random guessing; protocol '{protocol}'",
        label,
        f"{'fig2' if component == 'engineered_pqk' else 'figS1'}_{component}_{protocol}",
        out_dirs,
    )


def figure_bound(rows, label, out_dirs):
    fig, axes = plt.subplots(
        1, len(EMBEDDINGS), figsize=(3.7 * len(EMBEDDINGS), 3.6), sharey=True
    )
    for col, embedding in enumerate(EMBEDDINGS):
        ax = axes[col]
        style_axes(ax)
        plot_series(
            ax,
            select(rows, "bound", "prediction_error_bound", embedding, None, 600),
            "prediction_error_bound",
        )
        error_rows = [
            r
            for r in select(
                rows, "quantum_dataset", "best_classical", embedding, "paper", 600
            )
            if r["n"] <= 12
        ]
        plot_series(ax, error_rows, "best_classical")
        ax.set_title(f"Dataset (Q, {embedding})", fontsize=9, color=TEXT_PRIMARY)
        ax.set_xlabel("n (system size)", fontsize=9, color=TEXT_PRIMARY)
        if col == 0:
            ax.set_ylabel("bound / prediction error", fontsize=9, color=TEXT_PRIMARY)
    finish(
        fig,
        axes[None, :],
        "Fig. S3 (journal Supp. Fig. 4): classical bound vs best classical error, N = 600",
        label,
        "figS3_bound_vs_error",
        out_dirs,
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--aggregate", required=True, type=Path)
    parser.add_argument(
        "--label", required=True, help="Result-scope label shown on every figure"
    )
    parser.add_argument("--protocol", default="paper", choices=("paper", "cv"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/figures"))
    args = parser.parse_args(argv)
    aggregate = load_json(args.aggregate)
    rows = aggregate["summary"]
    global EMBEDDINGS
    EMBEDDINGS = tuple(sorted({r["embedding"] for r in rows if r["embedding"]}))
    out_dirs = [args.results_dir] + [Path(run) / "figures" for run in aggregate["runs"]]
    components = {r["component"] for r in rows}
    if "dimension_geometry" in components:
        figure_dimension_geometry(rows, args.label, out_dirs)
    if "quantum_dataset" in components:
        figure_prediction_error(rows, args.label, args.protocol, out_dirs)
    for component in ("engineered_pqk", "engineered_qk"):
        if component in components:
            figure_engineered(rows, component, args.label, args.protocol, out_dirs)
    if "bound" in components:
        figure_bound(rows, args.label, out_dirs)
    print(f"figures written to {', '.join(str(d) for d in out_dirs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
