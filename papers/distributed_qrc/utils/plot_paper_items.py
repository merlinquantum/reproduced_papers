"""One PNG per table and per figure of the paper, paper vs this reproduction.

Writes ``results/paper_items/TABLE_<n>.png`` for Tables I-VIII, B1-B2 and C1-C12,
and ``results/paper_items/FIGURE_<n>.png`` for Figures 1-7 and A1. Numbers follow
the paper's printed numbering (IEEE Roman numerals for the main tables, written
here in Arabic: TABLE_4 is the paper's Table IV).

Every table has a **Paper** column group (blue) and a **Reproduction** column
group (green), so that the origin of each value is explicit.

* Result tables (IV-VIII, C1-C12): the paper's reported values and ours (mean
  ± sample std over seeds 0-2, sweeps ``E{1..4}-R2``, full data).
* Descriptive tables (I-III, B1-B2): the paper's content and what is implemented
  (configurations, data rows, device mapping, calibration values used).
* Figures 4-7: the paper's per-cell values (left column, from Tables C1-C12)
  and ours (right column, seed mean), drawn in the paper's style (x ordered by
  total neurons, linear y), MAE and RMSE rows on a shared y axis. Figures 1-3 and A1 are diagrams and circuits:
  they show this implementation (circuits drawn by Qiskit from the same gates as
  ``lib/quantum_neuron.py`` and ``lib/kernels.py``); the paper's drawings are not
  copied, compare with the paper.

Paper values come from the cached arXiv LaTeX source (Tables I-VIII, B1-B2) and
from ``results/data/paper_tables.json`` (Tables C1-C12, extracted by
``utils/extract_paper_tables.py``).

Usage (from the repository root)::

    python papers/distributed_qrc/utils/plot_paper_items.py papers/distributed_qrc/outdir \\
        data/distributed_qrc/sources/arxiv_source_v1/tqe_tran.tex
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

PAPER_DIR = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PAPER_DIR.parents[1]), str(PAPER_DIR.parent)]

OUT = PAPER_DIR / "results" / "paper_items"
ARCHS = {"srsr": "E1", "mrsr": "E2", "srmr": "E3", "mrmr": "E4"}
ARCH_NAMES = {"srsr": "SRSR", "mrsr": "MRSR", "srmr": "SRMR", "mrmr": "MRMR"}
MAIN_TABLE = {"srsr": 4, "mrsr": 5, "srmr": 6, "mrmr": 7}
ROMAN = {4: "IV", 5: "V", 6: "VI", 7: "VII", 8: "VIII"}
LATEX_LABEL = {
    4: "tabArc1tot",
    5: "tabArc2tot",
    6: "tabArc3tot",
    7: "tabArc4tot",
    8: "tab12",
}
C_TABLES = {
    1: ("srsr", "linear", "classical Ridge output"),
    2: ("srsr", "quantum_kernel_ideal", "quantum Ridge output, ideal simulator"),
    3: ("srsr", "quantum_kernel_noisy", "quantum Ridge output, noisy simulator"),
    4: ("mrsr", "linear", "classical Ridge output"),
    5: ("mrsr", "quantum_kernel_ideal", "quantum Ridge output, ideal simulator"),
    6: ("mrsr", "quantum_kernel_noisy", "quantum Ridge output, noisy simulator"),
    7: ("srmr", "linear", "multiple classical Ridge output instances"),
    8: (
        "srmr",
        "quantum_kernel_ideal",
        "multiple quantum Ridge instances, ideal simulator",
    ),
    9: (
        "srmr",
        "quantum_kernel_noisy",
        "multiple quantum Ridge instances, noisy simulators",
    ),
    10: ("mrmr", "linear", "multiple classical Ridge output instances"),
    11: (
        "mrmr",
        "quantum_kernel_ideal",
        "multiple quantum Ridge instances, ideal simulator",
    ),
    12: (
        "mrmr",
        "quantum_kernel_noisy",
        "multiple quantum Ridge instances, noisy simulators",
    ),
}
RES_LABEL = {
    "classical": "Classical",
    "quantum_ideal": "Quantum (Sim.)",
    "quantum_noisy": "Quantum (Noisy)",
}
RIDGE_LABEL = {
    "linear": "Classical",
    "quantum_kernel_ideal": "Quantum (Sim.)",
    "quantum_kernel_noisy": "Quantum (Noisy)",
}
RESERVOIRS = ("classical", "quantum_ideal", "quantum_noisy")
READOUTS = ("linear", "quantum_kernel_ideal", "quantum_kernel_noisy")
FOOT = "Paper: values as printed (one value per cell; the paper reports no seeds or spread). Reproduction: mean ± sample std over seeds 0, 1, 2 (reproduction sweeps; full data; full training set for every kernel; simulation)."

# Column groups: None (row labels / common), "paper", "repro".
GROUP_COLOURS = {
    None: ("#e6e6e6", "#ffffff"),
    "paper": ("#c9daf0", "#eef4fb"),
    "repro": ("#c8e6cf", "#eef8f0"),
}
GROUP_TITLES = {"paper": "Paper", "repro": "Reproduction"}


# ------------------------------------------------------------------ data access
def load_ours(outdir: Path) -> dict:
    """{arch: {(config, reservoir, readout): {metric: (mean, std)}}} from the R2 sweeps."""

    ours = {}
    for arch, exp in ARCHS.items():
        summary = json.loads((outdir / f"sweep_{exp}-R2" / "summary.json").read_text())[
            "cells"
        ]["default"]
        ours[arch] = {
            tuple(key.split("|")): {
                m: (cell[f"test.{m}"]["mean"], cell[f"test.{m}"]["std"])
                for m in ("mae", "rmse", "r2")
            }
            for key, cell in summary.items()
        }
    return ours


def load_paper_cells() -> dict:
    cells = json.loads(
        (PAPER_DIR / "results" / "data" / "paper_tables.json").read_text()
    )["cells"]
    return {
        (
            c["architecture"],
            c["configuration"],
            c["reservoir_type"],
            c["readout_type"],
        ): c
        for c in cells
    }


def latex_rows(tex: str, label: str) -> list[list[str]]:
    """Data rows of a tabular whose \\label is ``label`` (bold and spacing removed)."""

    start = tex.index(f"\\label{{table:{label}}}")
    end = tex.index("\\end{tabular}", start)
    rows = []
    for line in tex[start:end].splitlines():
        line = re.sub(r"\\textbf\{([^}]*)\}", r"\1", line).replace("\\\\", "").strip()
        if "&" not in line or line.startswith("%"):
            continue
        cells = [c.strip() for c in line.split("&")]
        if re.match(r"^\d", cells[0]):
            rows.append(cells)
    return rows


def latex_spec_rows(tex: str, label: str, n_cells: int) -> list[list[str]]:
    """Specification rows (label first, no leading backslash) of the calibration tables."""

    start = tex.index(f"\\label{{table:{label}}}")
    rows = []
    for line in tex[start : tex.index("\\end{tabular}", start)].splitlines():
        cells = [c.strip().replace("\\\\", "").strip() for c in line.split("&")]
        if (
            len(cells) == n_cells
            and cells[0]
            and not cells[0].startswith("\\")
            and "Specifications" not in cells[0]
        ):
            rows.append(cells)
    return rows


def config_order(arch: str) -> list[str]:
    cfg = json.loads(
        (
            PAPER_DIR / "configs" / f"{ARCHS[arch].lower()}_{arch}_original.json"
        ).read_text()
    )
    from distributed_qrc.lib.architectures import build_cell_spec

    return [build_cell_spec(arch, size).label for size in cfg["experiment"]["sizes"]]


def total_neurons(arch: str, label: str) -> int:
    if arch == "srsr":
        return int(label)
    if arch == "srmr":
        return int(label.split("/")[0])
    k, n = label.split("x")
    return int(k) * int(n)


def fmt(mean_std: tuple[float, float], digits: int = 4) -> str:
    mean, std = mean_std
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def sci(value: float) -> str:
    """Scientific notation as printed in the paper's Table B1 (e.g. 7.519e-3)."""

    return f"{value:.3e}".replace("e-0", "e-")


def match_cell(
    paper: dict, arch: str, config: str, mae: float, rmse: float, r2: float
) -> tuple[str, str] | None:
    """The C-table cell whose printed values equal a main-table row (resolves label typos)."""

    for res in RESERVOIRS:
        for ro in READOUTS:
            c = paper.get((arch, config, res, ro))
            if (
                c
                and abs(c["mae"] - mae) < 1e-9
                and abs(c["rmse"] - rmse) < 1e-9
                and abs(c["r2"] - r2) < 1e-9
            ):
                return res, ro
    return None


# ------------------------------------------------------------------ legends

LEGEND_METRICS = "MAE, RMSE, R²: test-set mean absolute error, root mean squared error and coefficient of determination, on the min-max-normalized load (lower MAE/RMSE and higher R² are better)."
LEGEND_TYPES = "Reservoir and ridge types: Classical = ReservoirPy reservoir / linear ridge; Quantum (Sim.) = ideal simulator; Quantum (Noisy) = simulator with the hardware-informed noise model (Table III mapping, Tables B1-B2)."
LEGEND_CONFIG = "Configuration labels: N = one reservoir of N neurons; kxn = k reservoirs of n neurons each; N/m = one reservoir of N neurons read by m ridge instances."
LEGENDS = {
    "TABLE_1": [
        "No., Details: the paper's Table I. architecture: the same architecture in this reproduction; as implemented: how its reservoirs and ridge readouts are connected (lib/architectures.py, lib/runner.py); config: the file with its 8 configurations.",
    ],
    "TABLE_2": [
        "Each row is one model input/target pair: the loads of the 4 past hours (hour-3 … current hour, MW) and the load of the next hour (future, MW), before normalization.",
        "first / second window: the first two pairs of the series; last window: the final pair. identical: our values equal the paper's.",
    ],
    "TABLE_3": [
        "Rows: number of noisy quantum components in a model (reservoirs, or ridge instances). Columns: how many of these components use each QPU noise profile; '-' = none.",
        "Example: a model with 3 reservoirs runs one on each profile; a single component always uses IBM Brisbane.",
    ],
    "TABLE_MAIN": [
        "config: configuration label. " + LEGEND_CONFIG,
        "Paper columns: the cell the paper reports as best for this configuration (reservoir type, ridge type) and its test metrics. 'Same model' is that cell, identified by its printed values in Tables C1-C12.",
        "Reproduction, same model as the paper: our test metrics for that same cell. Reproduction, our best model: test metrics of our cell with the highest seed-mean R² among the 8 quantum/hybrid cells (the paper's selection rule); this cell can differ from the paper's (the cell of every seed is listed in results/data/headline_statistics.json).",
        LEGEND_TYPES,
        LEGEND_METRICS,
    ],
    "TABLE_8": [
        "neurons: total number of reservoir neurons. architecture (config): the architecture and configuration the paper reports as best at this size.",
        "reservoir / ridge in the paper's wording: Single- = one component, Distributed- = several; -Quantum / -Classical = component type.",
        "Reproduction, same model as the paper: our metrics for that cell. Reproduction, our best model: our best cell over all 4 architectures at this size (highest seed-mean R²) and its MAE.",
        LEGEND_TYPES,
        LEGEND_METRICS,
    ],
    "TABLE_B1": [
        "specification: calibration quantity printed in the paper's App. B for each IBM device. Paper columns: the printed values.",
        "value used: the error rate our noise model takes from that row; how it enters our noise model: where the error is applied, or why the row is not used.",
    ],
    "TABLE_B2": [
        "specification: calibration quantity printed in the paper's App. B for IonQ Aria-1. Paper: the printed value.",
        "error rate used: the rate our noise model takes from that row; how it enters our noise model: where the error is applied, or why the row is not used.",
    ],
    "TABLE_C": [
        "config: configuration label. " + LEGEND_CONFIG,
        "Column groups: the reservoir type of the model (Classical, Quantum (Sim.) = ideal simulator, Quantum (Noisy) = noisy simulator); the ridge type is fixed by the table (see the title).",
        LEGEND_METRICS,
        "Reproduction cells: mean on the first line, ± sample std over seeds 0, 1, 2 on the second line.",
    ],
}


def legend_for(name: str) -> list[str]:
    if name.startswith("TABLE_C"):
        return LEGENDS["TABLE_C"]
    if name in ("TABLE_4", "TABLE_5", "TABLE_6", "TABLE_7"):
        return LEGENDS["TABLE_MAIN"]
    return LEGENDS.get(name, [])


# ------------------------------------------------------------------ drawing
def draw_table(
    name: str,
    title: str,
    header: list[list[tuple[str, int, str | None]]],
    rows: list[list[str]],
    groups: list[str | None],
    widths: list[float],
    notes: list[str],
    highlight: dict[int, str] | None = None,
    fontsize: float = 8.0,
) -> None:
    """Grouped-header table. ``header`` rows hold (label, column span, group); ``groups`` colours each column."""

    highlight = highlight or {}
    legend = legend_for(name)
    notes = [
        *(["Legend:"] + [f"  • {line}" for line in legend] if legend else []),
        *(f"Row {rows[i][0]}: {text}" for i, text in highlight.items()),
        *notes,
    ]
    lines = max(cell.count("\n") + 1 for row in rows for cell in row)
    header_lines = [max(label.count("\n") + 1 for label, _, _ in r) for r in header]
    header_hs = [0.2 * n + 0.14 for n in header_lines]
    row_h = 0.24 * lines + 0.08
    total_w = sum(widths)
    total_h = sum(header_hs) + row_h * len(rows)
    fig = plt.figure(figsize=(total_w + 0.4, total_h + 0.9 + 0.17 * len(notes)))
    ax = fig.add_axes(
        (
            0.2 / (total_w + 0.4),
            (0.25 + 0.17 * len(notes)) / (total_h + 0.9 + 0.17 * len(notes)),
            total_w / (total_w + 0.4),
            total_h / (total_h + 0.9 + 0.17 * len(notes)),
        )
    )
    ax.set_xlim(0, total_w)
    ax.set_ylim(total_h, 0)
    ax.axis("off")
    x_edges = np.concatenate([[0], np.cumsum(widths)])

    def cell(
        x0,
        x1,
        y0,
        h,
        text,
        face,
        weight="normal",
        edge="#8c8c8c",
        lw=0.6,
        colour="black",
    ):
        ax.add_patch(
            Rectangle(
                (x0, y0),
                x1 - x0,
                h,
                facecolor=face,
                edgecolor=edge,
                lw=lw,
                clip_on=False,
            )
        )
        ax.text(
            (x0 + x1) / 2,
            y0 + h / 2,
            text,
            ha="center",
            va="center",
            fontsize=fontsize,
            weight=weight,
            color=colour,
        )

    y = 0.0
    for level, header_row in enumerate(header):
        col = 0
        for label, span, group in header_row:
            face = GROUP_COLOURS[group][0] if group else GROUP_COLOURS[None][0]
            size_weight = "bold"
            cell(
                x_edges[col],
                x_edges[col + span],
                y,
                header_hs[level],
                label,
                face,
                size_weight,
            )
            col += span
        y += header_hs[level]
    for r, row in enumerate(rows):
        for c, text in enumerate(row):
            face = GROUP_COLOURS[groups[c]][1]
            if r in highlight and c == 0:
                face = "#ffe8a3"
            cell(x_edges[c], x_edges[c + 1], y, row_h, text, face)
        y += row_h
    # Thick separators between column groups.
    for c in range(1, len(groups)):
        if groups[c] != groups[c - 1]:
            ax.plot(
                [x_edges[c]] * 2, [0, total_h], color="#333333", lw=1.6, clip_on=False
            )
    ax.add_patch(
        Rectangle(
            (0, 0),
            total_w,
            total_h,
            fill=False,
            edgecolor="#333333",
            lw=1.2,
            clip_on=False,
        )
    )
    fig.text(
        0.5,
        1 - 0.12 / (total_h + 0.9 + 0.17 * len(notes)),
        title,
        ha="center",
        va="top",
        fontsize=10.5,
        weight="bold",
    )
    fig.text(
        0.2 / (total_w + 0.4),
        0.02,
        "\n".join(notes),
        fontsize=7.5,
        va="bottom",
        ha="left",
        color="#333333",
    )
    fig.savefig(OUT / f"{name}.png", dpi=150, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print(f"wrote {os.path.relpath(OUT / name)}.png")


def box(ax, x, y, w, h, text, color="#dbe9f6"):
    ax.add_patch(
        FancyBboxPatch(
            (x - w / 2, y - h / 2),
            w,
            h,
            boxstyle="round,pad=0.02",
            fc=color,
            ec="#333333",
        )
    )
    ax.text(x, y, text, ha="center", va="center", fontsize=8)


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(
        FancyArrowPatch(
            (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=10, color="#333333"
        )
    )


def architecture_panel(ax, kind: str, k: int = 3) -> None:
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ys = np.linspace(4.8, 1.2, k)
    box(ax, 0.8, 3.0, 1.2, 0.8, "x(t)\n4 lags", "#f2f2f2")
    if kind == "srsr":
        box(ax, 3.3, 3, 2.0, 0.9, "Reservoir\n(N neurons)")
        box(ax, 6.3, 3, 1.8, 0.9, "Readout", "#fde3c8")
        box(ax, 9.0, 3, 0.9, 0.7, "ŷ", "#f2f2f2")
        arrow(ax, 1.4, 3, 2.3, 3)
        arrow(ax, 4.3, 3, 5.4, 3)
        arrow(ax, 7.2, 3, 8.55, 3)
    elif kind == "mrsr":
        for i, y in enumerate(ys):
            box(ax, 3.3, y, 1.9, 0.8, f"Reservoir {i + 1}\n(n neurons)")
            arrow(ax, 1.4, 3, 2.35, y)
            arrow(ax, 4.25, y, 5.3, 3)
        box(ax, 5.8, 3, 1.0, 0.8, "concat", "#eeeeee")
        box(ax, 7.6, 3, 1.4, 0.9, "Readout", "#fde3c8")
        box(ax, 9.4, 3, 0.8, 0.7, "ŷ", "#f2f2f2")
        arrow(ax, 6.3, 3, 6.9, 3)
        arrow(ax, 8.3, 3, 9.0, 3)
    elif kind == "srmr":
        box(ax, 2.9, 3, 1.6, 0.9, "Reservoir\n(N neurons)")
        arrow(ax, 1.4, 3, 2.1, 3)
        for i, y in enumerate(ys):
            box(
                ax,
                5.5,
                y,
                1.9,
                0.8,
                f"Readout {i + 1}\n(feature part {i + 1})",
                "#fde3c8",
            )
            arrow(ax, 3.7, 3, 4.55, y)
            arrow(ax, 6.45, y, 7.55, 3)
        box(ax, 8.0, 3, 0.9, 0.8, "mean", "#eeeeee")
        box(ax, 9.4, 3, 0.8, 0.7, "ŷ", "#f2f2f2")
        arrow(ax, 8.45, 3, 9.0, 3)
    else:
        for i, y in enumerate(ys):
            box(ax, 3.0, y, 1.8, 0.8, f"Reservoir {i + 1}")
            box(ax, 5.6, y, 1.7, 0.8, f"Readout {i + 1}", "#fde3c8")
            arrow(ax, 1.4, 3, 2.1, y)
            arrow(ax, 3.9, y, 4.75, y)
            arrow(ax, 6.45, y, 7.55, 3)
        box(ax, 8.0, 3, 0.9, 0.8, "mean", "#eeeeee")
        box(ax, 9.4, 3, 0.8, 0.7, "ŷ", "#f2f2f2")
        arrow(ax, 8.45, 3, 9.0, 3)
    ax.set_title(
        f"{ARCH_NAMES[kind]} as implemented (lib/architectures.py, lib/runner.py)",
        fontsize=9,
    )


def circuit_text_figure(
    name: str, title: str, circuits: list[tuple[str, object]], notes: list[str]
) -> None:
    texts = [(label, str(qc.draw(output="text", fold=-1))) for label, qc in circuits]
    n_lines = sum(t.count("\n") + 3 for _, t in texts)
    width = max(len(line) for _, t in texts for line in t.splitlines()) * 0.075 + 1
    fig = plt.figure(
        figsize=(min(max(width, 8), 30), 0.17 * n_lines + 1.2 + 0.2 * len(notes))
    )
    step = 1.0 / (n_lines + 6 + len(notes))
    fig.text(0.01, 0.995, title, fontsize=10, va="top")
    y = 0.995 - 2.5 * step
    for label, text in texts:
        fig.text(0.01, y, label, fontsize=9, weight="bold", va="top")
        y -= 1.5 * step
        fig.text(0.01, y, text, family="monospace", fontsize=7, va="top")
        y -= (text.count("\n") + 1.5) * step
    fig.text(0.01, 0.01, "\n".join(notes), fontsize=7, va="bottom")
    fig.savefig(OUT / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {os.path.relpath(OUT / name)}.png")


# ------------------------------------------------------------------ tables
def table_1(tex: str) -> None:
    details = {row[0]: row[1] for row in latex_rows(tex, "tab1")}
    implemented = {
        "srsr": "Single reservoir, single ridge readout",
        "mrsr": "k reservoirs, features concatenated, single ridge readout",
        "srmr": "Single reservoir, features split over m ridge readouts, predictions averaged",
        "mrmr": "k reservoirs, reservoir i feeding ridge readout i, predictions averaged",
    }
    rows = [
        [
            str(i),
            details[str(i)],
            ARCH_NAMES[arch],
            implemented[arch],
            f"configs/{ARCHS[arch].lower()}_{arch}_original.json",
        ]
        for i, arch in enumerate(ARCHS, 1)
    ]
    header = [
        [("", 1, None), ("Paper", 1, "paper"), ("Reproduction", 3, "repro")],
        [
            ("No.", 1, None),
            ("Details", 1, "paper"),
            ("architecture", 1, "repro"),
            ("as implemented (lib/architectures.py)", 1, "repro"),
            ("config", 1, "repro"),
        ],
    ]
    draw_table(
        "TABLE_1",
        "Table I — Reservoir architecture details",
        header,
        rows,
        [None, "paper", "repro", "repro", "repro"],
        [0.5, 4.6, 1.1, 4.8, 2.6],
        [],
    )


def table_2(tex: str) -> None:
    from distributed_qrc.lib.data import dataset_dir, load_hourly_series

    cfg = json.loads((PAPER_DIR / "configs" / "defaults.json").read_text())
    series = load_hourly_series(
        dataset_dir(cfg), cfg["dataset"]["start"], cfg["dataset"]["end"]
    )
    paper = latex_rows(tex, "tab2")
    ours = [series[0:5], series[1:6], series[-5:]]
    rows = []
    for label, p, o in zip(
        ("first window", "second window", "last window"), paper, ours
    ):
        p_vals = [float(v.replace(",", "")) for v in p]
        rows.append(
            [
                label,
                *[f"{int(v):,}" for v in p_vals],
                *[f"{int(round(v)):,}" for v in o],
                "yes" if np.allclose(p_vals, np.round(o)) else "no",
            ]
        )
    cols = ["hour-3", "hour-2", "hour-1", "current", "future"]
    header = [
        [
            ("", 1, None),
            ("Paper", 5, "paper"),
            ("Reproduction (our data)", 5, "repro"),
            ("", 1, None),
        ],
        [
            ("row", 1, None),
            *[(c, 1, "paper") for c in cols],
            *[(c, 1, "repro") for c in cols],
            ("identical", 1, None),
        ],
    ]
    draw_table(
        "TABLE_2",
        "Table II — 4-hour windows and next-hour load (MW, before normalization)",
        header,
        rows,
        [None] + ["paper"] * 5 + ["repro"] * 5 + [None],
        [1.2] + [0.85] * 10 + [0.8],
        [
            f"Reproduction: IPTO RealTimeSCADASystemLoad, {cfg['dataset']['start']} … {cfg['dataset']['end']} ({series.size:,} hourly values), lib/data.py::load_hourly_series."
        ],
    )


def table_3(tex: str) -> None:
    from distributed_qrc.lib.noise import PAPER_DEVICE_MAPPING

    rows = []
    for p in latex_rows(tex, "tab3"):
        devices = PAPER_DEVICE_MAPPING[int(p[0])]
        ours = [
            str(devices.count(d)) if devices.count(d) else "-"
            for d in ("ibm_marrakesh", "ibm_brisbane", "ionq_aria1")
        ]
        rows.append([p[0], *p[1:], *ours, "yes" if p[1:] == ours else "no"])
    devs = ["IBM Marrakesh", "IBM Brisbane", "IonQ Aria-1"]
    header = [
        [
            ("", 1, None),
            ("Paper", 3, "paper"),
            ("Reproduction (lib/noise.py)", 3, "repro"),
            ("", 1, None),
        ],
        [
            ("reservoirs / ridge\ninstances", 1, None),
            *[(d, 1, "paper") for d in devs],
            *[(d, 1, "repro") for d in devs],
            ("identical", 1, None),
        ],
    ]
    draw_table(
        "TABLE_3",
        "Table III — Distribution of noisy reservoirs / ridge instances over QPU noise profiles",
        header,
        rows,
        [None] + ["paper"] * 3 + ["repro"] * 3 + [None],
        [1.5] + [1.2] * 6 + [0.8],
        [
            "Reproduction: PAPER_DEVICE_MAPPING, used with --noise-mapping paper. Noise profiles only: neither the paper nor this reproduction ran on QPUs."
        ],
    )


def table_main(number: int, arch: str, tex: str, ours: dict, paper: dict) -> None:
    rows, highlight = [], {}
    has_instances = arch in ("srmr", "mrmr")
    for i, p in enumerate(latex_rows(tex, LATEX_LABEL[number])):
        config = f"{p[0]}/{p[1]}" if arch == "srmr" else p[0]
        res_l, ridge_l = (p[2], p[3]) if has_instances else (p[1], p[2])
        mae, rmse, r2 = (float(v) for v in p[-3:])
        cell = match_cell(paper, arch, config, mae, rmse, r2)
        o = ours[arch][(config, *cell)] if cell else None
        candidates = {
            k[1:]: v
            for k, v in ours[arch].items()
            if k[0] == config and k[1:] != ("classical", "linear")
        }
        best = max(candidates, key=lambda k: candidates[k]["r2"][0])
        printed = (
            res_l.replace("(Noisy.)", "(Noisy)").replace(
                "Classical (Sim.)", "Classical"
            ),
            ridge_l.replace("(Noisy.)", "(Noisy)"),
        )
        if cell and (RES_LABEL[cell[0]], RIDGE_LABEL[cell[1]]) != printed:
            highlight[i] = (
                f"the paper prints {res_l} / {ridge_l}, but these values are those of the {RES_LABEL[cell[0]]} / {RIDGE_LABEL[cell[1]]} cell "
                "of the paper's Table C; the reproduction compares with that cell."
            )
        rows.append(
            [
                config,
                res_l,
                ridge_l,
                f"{mae:.4f}",
                f"{rmse:.4f}",
                f"{r2:.4f}",
                fmt(o["mae"]) if o else "n/a",
                fmt(o["rmse"]) if o else "n/a",
                fmt(o["r2"]) if o else "n/a",
                fmt(candidates[best]["mae"]),
                fmt(candidates[best]["rmse"]),
                fmt(candidates[best]["r2"]),
            ]
        )
    header = [
        [
            ("", 1, None),
            ("Paper: best model", 5, "paper"),
            ("Reproduction", 6, "repro"),
        ],
        [
            ("", 1, None),
            ("", 5, "paper"),
            ("same model as the paper", 3, "repro"),
            ("our best model", 3, "repro"),
        ],
        [
            ("config", 1, None),
            ("reservoir", 1, "paper"),
            ("ridge", 1, "paper"),
            ("MAE", 1, "paper"),
            ("RMSE", 1, "paper"),
            ("R²", 1, "paper"),
            ("MAE", 1, "repro"),
            ("RMSE", 1, "repro"),
            ("R²", 1, "repro"),
            ("MAE", 1, "repro"),
            ("RMSE", 1, "repro"),
            ("R²", 1, "repro"),
        ],
    ]
    draw_table(
        f"TABLE_{number}",
        f"Table {ROMAN[number]} — {ARCH_NAMES[arch]}: best-performing model per configuration",
        header,
        rows,
        [None] + ["paper"] * 5 + ["repro"] * 6,
        [0.7, 1.3, 1.3, 0.75, 0.75, 0.75, 1.45, 1.45, 1.45, 1.45, 1.45, 1.45],
        [
            FOOT,
        ],
        highlight=highlight,
    )


def table_8(tex: str, ours: dict, paper: dict) -> None:
    labels = {
        arch: {total_neurons(arch, c): c for c in config_order(arch)} for arch in ARCHS
    }
    rows = []
    for p in latex_rows(tex, "tab12"):
        n, arch = int(p[0]), p[1].lower()
        config = labels[arch][n]
        mae, rmse, r2 = (float(v) for v in p[-3:])
        cell = match_cell(paper, arch, config, mae, rmse, r2)
        o = ours[arch][(config, *cell)] if cell else None
        pool = {
            (a, labels[a][n], *k[1:]): v
            for a in ARCHS
            for k, v in ours[a].items()
            if k[0] == labels[a][n] and k[1:] != ("classical", "linear")
        }
        best = max(pool, key=lambda k: pool[k]["r2"][0])
        rows.append(
            [
                p[0],
                f"{p[1]} ({config})",
                p[2],
                p[3],
                f"{mae:.4f}",
                f"{rmse:.4f}",
                f"{r2:.4f}",
                fmt(o["mae"]) if o else "n/a",
                fmt(o["r2"]) if o else "n/a",
                f"{best[0].upper()} {best[1]}\n{RES_LABEL[best[2]]} / {RIDGE_LABEL[best[3]]}",
                fmt(pool[best]["mae"]),
            ]
        )
    header = [
        [
            ("", 1, None),
            ("Paper: best model", 6, "paper"),
            ("Reproduction", 4, "repro"),
        ],
        [
            ("", 1, None),
            ("", 6, "paper"),
            ("same model as the paper", 2, "repro"),
            ("our best model, all architectures", 2, "repro"),
        ],
        [
            ("neurons", 1, None),
            ("architecture", 1, "paper"),
            ("reservoir", 1, "paper"),
            ("ridge", 1, "paper"),
            ("MAE", 1, "paper"),
            ("RMSE", 1, "paper"),
            ("R²", 1, "paper"),
            ("MAE", 1, "repro"),
            ("R²", 1, "repro"),
            ("architecture / cell", 1, "repro"),
            ("MAE", 1, "repro"),
        ],
    ]
    draw_table(
        "TABLE_8",
        "Table VIII — Best-performing model per total number of reservoir neurons",
        header,
        rows,
        [None] + ["paper"] * 6 + ["repro"] * 4,
        [0.75, 1.3, 1.75, 1.6, 0.75, 0.75, 0.75, 1.45, 1.45, 2.3, 1.45],
        [
            FOOT,
            "Our best model: highest seed-mean test R² among the quantum/hybrid cells of the 4 architectures at this total number of neurons.",
        ],
    )


# How each App. B calibration quantity enters lib/noise.py (or why it does not).
SINGLE_QUBIT_ROLE = (
    "1-qubit depolarizing error after every native 1-qubit gate"
    "\n(2 per RX, 1 per H, 1 per qubit around each CX; RZ and P are virtual, error free)"
)
TWO_QUBIT_ROLE = "2-qubit depolarizing error after every CX"
READOUT_ROLE = "each measured qubit is flipped with this probability"
DECOHERENCE_ROLE = (
    "not used: the gate errors above (randomized benchmarking) already include"
    "\ndecoherence during the gates; idle time is neglected"
)


def table_b(tex: str) -> None:
    cfg = json.loads((PAPER_DIR / "configs" / "defaults.json").read_text())["noise"][
        "devices"
    ]
    # spec -> (device whose value is used, or None for both; config key; role)
    used = {
        "Median ECR error": (
            "ibm_brisbane",
            "error_2q",
            TWO_QUBIT_ROLE + "\n(ECR is Brisbane's native 2-qubit gate)",
        ),
        "Median CZ error": (
            "ibm_marrakesh",
            "error_2q",
            TWO_QUBIT_ROLE + "\n(CZ is Marrakesh's native 2-qubit gate)",
        ),
        "Median SX error": (None, "error_1q", SINGLE_QUBIT_ROLE),
        "Median readout error": (None, "readout_error", READOUT_ROLE),
    }
    not_used = {
        "2Q error (best)": "not used: best qubit pair only; we use the median"
        "\n2-qubit error of each device (ECR / CZ rows)",
        "2Q error (layered)": "not used: error per layered gate (whole-chip benchmark);"
        "\nwe apply a per-gate error instead (ECR / CZ rows)",
        "Median T1": DECOHERENCE_ROLE,
        "Median T2": DECOHERENCE_ROLE,
        "Processor Type": "not used",
    }
    rows = []
    for spec, bris, marr in latex_spec_rows(tex, "tabAppA", 3):
        if spec in used:
            device, key, role = used[spec]
            ours_b = (
                sci(cfg["ibm_brisbane"][key])
                if device in (None, "ibm_brisbane")
                else "-"
            )
            ours_m = (
                sci(cfg["ibm_marrakesh"][key])
                if device in (None, "ibm_marrakesh")
                else "-"
            )
            rows.append([spec, bris, marr, ours_b, ours_m, role])
        else:
            rows.append([spec, bris, marr, "-", "-", not_used[spec]])
    header = [
        [
            ("", 1, None),
            ("Paper", 2, "paper"),
            ("Reproduction", 3, "repro"),
        ],
        [
            ("specification", 1, None),
            ("IBM Brisbane", 1, "paper"),
            ("IBM Marrakesh", 1, "paper"),
            ("value used\nBrisbane", 1, "repro"),
            ("value used\nMarrakesh", 1, "repro"),
            ("how it enters our noise model", 1, "repro"),
        ],
    ]
    draw_table(
        "TABLE_B1",
        "Table B1 — Calibration data for IBM Brisbane and IBM Marrakesh",
        header,
        rows,
        [None, "paper", "paper", "repro", "repro", "repro"],
        [2.0, 1.3, 1.3, 1.3, 1.3, 5.6],
        [
            "The paper does not say how it built its noise models. Ours uses only these medians"
            " (configs/defaults.json, lib/noise.py), so it is a simplified version of a full device noise model.",
            "'-' = value not used for that device. The quantum_noisy cells use these rates as printed;"
            " the noise side check (Fair Baselines) also multiplies them by 10.",
        ],
    )
    ion = cfg["ionq_aria1"]
    keys = {
        "Measurement fidelity (median)": ("readout_error", READOUT_ROLE),
        "1Q fidelity (median)": ("error_1q", SINGLE_QUBIT_ROLE),
        "2Q fidelity (median)": ("error_2q", TWO_QUBIT_ROLE),
    }
    rows = []
    for spec, value in latex_spec_rows(tex, "tabAppB", 2):
        if spec in keys:
            key, role = keys[spec]
            rows.append([spec, value, f"{ion[key]:.4g}\n(= 1 − {value})", role])
        else:
            rows.append([spec, value, "-", DECOHERENCE_ROLE])
    header = [
        [
            ("", 1, None),
            ("Paper", 1, "paper"),
            ("Reproduction", 2, "repro"),
        ],
        [
            ("specification", 1, None),
            ("IonQ Aria-1", 1, "paper"),
            ("error rate used", 1, "repro"),
            ("how it enters our noise model", 1, "repro"),
        ],
    ]
    draw_table(
        "TABLE_B2",
        "Table B2 — Calibration data for IonQ Aria-1",
        header,
        rows,
        [None, "paper", "repro", "repro"],
        [2.6, 1.2, 1.5, 5.6],
        [
            "IonQ reports fidelities; the error rate used is 1 − fidelity. The same gate counts as for IBM are applied"
            " (the IonQ native gate set is not modelled).",
            "The paper does not say how it built its noise models. Ours uses only these medians"
            " (configs/defaults.json, lib/noise.py).",
        ],
    )


def table_c(number: int, ours: dict, paper: dict) -> None:
    arch, readout, what = C_TABLES[number]
    rows = []
    for config in config_order(arch):
        row = [config]
        row += [
            f"{paper[(arch, config, res, readout)][m]:.4f}"
            for res in RESERVOIRS
            for m in ("mae", "rmse", "r2")
        ]
        row += [
            f"{ours[arch][(config, res, readout)][m][0]:.4f}\n± {ours[arch][(config, res, readout)][m][1]:.4f}"
            for res in RESERVOIRS
            for m in ("mae", "rmse", "r2")
        ]
        rows.append(row)
    res_titles = [
        "Classical reservoir",
        "Quantum reservoir (Sim.)",
        "Quantum reservoir (Noisy)",
    ]
    metrics = ["MAE", "RMSE", "R²"]
    header = [
        [("", 1, None), ("Paper", 9, "paper"), ("Reproduction", 9, "repro")],
        [
            ("", 1, None),
            *[(t, 3, "paper") for t in res_titles],
            *[(t, 3, "repro") for t in res_titles],
        ],
        [
            ("config", 1, None),
            *[(m, 1, "paper") for _ in res_titles for m in metrics],
            *[(m, 1, "repro") for _ in res_titles for m in metrics],
        ],
    ]
    draw_table(
        f"TABLE_C{number}",
        f"Table C{number} — {ARCH_NAMES[arch]} model performance with {what}",
        header,
        rows,
        [None] + ["paper"] * 9 + ["repro"] * 9,
        [0.7] + [0.68] * 9 + [0.85] * 9,
        [
            FOOT,
            f"Relative differences per cell: results/data/{arch}_vs_paper.csv (column paper_table = C{number}).",
        ],
        fontsize=7.5,
    )


# ------------------------------------------------------------------ figures
# Style of the paper's Figs. 4-7 (Figures/comb-arch*.png in the arXiv source):
# one colour, marker and line style per (reservoir, ridge) model, legend order
# grouped by ridge type.
PAPER_FIGURE_STYLE = {
    ("classical", "linear"): ("tab:blue", "o", "-"),
    ("quantum_ideal", "linear"): ("tab:orange", "s", "-"),
    ("quantum_noisy", "linear"): ("tab:green", "s", "--"),
    ("classical", "quantum_kernel_ideal"): ("tab:red", "o", "-"),
    ("quantum_ideal", "quantum_kernel_ideal"): ("tab:purple", "d", "-"),
    ("quantum_noisy", "quantum_kernel_ideal"): ("tab:brown", "d", "--"),
    ("classical", "quantum_kernel_noisy"): ("tab:pink", "*", "-"),
    ("quantum_ideal", "quantum_kernel_noisy"): ("tab:gray", "*", "-"),
    ("quantum_noisy", "quantum_kernel_noisy"): ("tab:olive", "^", "--"),
}
PAPER_FIGURE_LABEL = {
    "classical": "Classical Reservoir",
    "quantum_ideal": "Quantum Reservoir(Sim)",
    "quantum_noisy": "Quantum Reservoir(Noisy)",
    "linear": "Classical Ridge",
    "quantum_kernel_ideal": "Quantum Ridge(Sim)",
    "quantum_kernel_noisy": "Quantum Ridge(Noisy)",
}


def figures_4_to_7(ours: dict, paper: dict) -> None:
    # Paper on the left, reproduction on the right; each row shares its y axis.
    sides = (
        ("paper", "Paper (as printed)"),
        ("repro", "Reproduction (mean over seeds 0, 1, 2)"),
    )
    for arch, number in zip(ARCHS, (4, 5, 6, 7)):
        # The paper orders configurations by total neurons. SRSR and SRMR use a
        # numeric axis of total neurons, MRSR and MRMR one tick per k x n label.
        configs = sorted(config_order(arch), key=lambda c: total_neurons(arch, c))
        totals = [total_neurons(arch, c) for c in configs]
        if arch in ("srsr", "srmr"):
            x, ticks, tick_labels = totals, totals, [str(t) for t in totals]
            xlabel = "Number of Reservoir Neurons"
        else:
            x = list(range(len(configs)))
            ticks, tick_labels = x, configs
            xlabel = "Number of Reservoirs x Number of Reservoir Neurons"
        first_c = 3 * number - 11
        fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True, sharey="row")
        for row, metric in enumerate(("mae", "rmse")):
            for col, (side, _) in enumerate(sides):
                ax = axes[row, col]
                ax.set_facecolor(GROUP_COLOURS[side][1])
                for ro in READOUTS:
                    for res in RESERVOIRS:
                        colour, marker, line_style = PAPER_FIGURE_STYLE[(res, ro)]
                        if side == "paper":
                            y = [paper[(arch, c, res, ro)][metric] for c in configs]
                        else:
                            y = [ours[arch][(c, res, ro)][metric][0] for c in configs]
                        label = f"{PAPER_FIGURE_LABEL[res]}-{PAPER_FIGURE_LABEL[ro]}"
                        if (res, ro) == ("classical", "linear"):
                            label += "(Baseline)"
                        ax.plot(
                            x,
                            y,
                            color=colour,
                            marker=marker,
                            ls=line_style,
                            ms=6 if marker == "*" else 5,
                            lw=1.4,
                            label=label,
                        )
                ax.set_xticks(ticks, tick_labels)
                ax.grid(True)
                if col == 0:
                    ax.set_ylabel(f"test {metric.upper()} (normalized)", fontsize=11)
                if row == 1:
                    ax.set_xlabel(xlabel, fontsize=11)
        for col, (side, title) in enumerate(sides):
            axes[0, col].set_title(
                title,
                fontsize=11,
                fontweight="bold",
                pad=10,
                bbox={
                    "boxstyle": "square,pad=0.4",
                    "facecolor": GROUP_COLOURS[side][0],
                    "edgecolor": "none",
                },
            )
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.095),
            ncol=3,
            fontsize=9,
            frameon=True,
            shadow=True,
        )
        fig.suptitle(
            f"Figure {number} — MAE and RMSE for all {ARCH_NAMES[arch]} models"
            f" (paper values from Tables C{first_c}-C{first_c + 2})",
            fontsize=12,
        )
        # Architecture-specific notes, one footnote line each.
        extra_notes = {
            "srmr": "SRMR: the number of ridge instances is 2 (10, 20, 40 neurons),"
            " 3 (15, 30, 60) or 5 (25, 50).\n",
            # The paper's own Fig. 5 (comb-arch2.png) draws this RMSE curve with
            # the SRSR values of Table C1, not those of Table C4.
            "mrsr": "Note: the left column plots Table C4. In the paper's printed Fig. 5,"
            " the RMSE curve of Quantum Reservoir(Sim)-Classical Ridge repeats the SRSR"
            " values of Table C1 (0.034, 0.0269, 0.0262, 0.038, ...), so it differs here.\n",
        }
        fig.text(
            0.01,
            0.01,
            "Legend: left column = values printed in the paper (Tables C); right column = our reproduction. "
            "Both panels of a row share the same y axis (linear, as in the paper).\n"
            "x axis = configuration, ordered by total neurons as in the paper ("
            + LEGEND_CONFIG[len("Configuration labels: ") :]
            + ").\n"
            + extra_notes.get(arch, "")
            + "y axis = test MAE (top) and RMSE (bottom) on the min-max-normalized load; lower is better. "
            "Colours, markers and line styles follow the paper's figure.\n"
            "Reproduction values are seed means; the seed spread is in the Tables C images (TABLE_C*.png) and results/data/<architecture>_vs_paper.csv.",
            fontsize=8,
            va="bottom",
        )
        fig.tight_layout(rect=(0, 0.2, 1, 0.96))
        fig.savefig(OUT / f"FIGURE_{number}.png", dpi=150)
        plt.close(fig)
        print(f"wrote {os.path.relpath(OUT)}/FIGURE_{number}.png")


def figures_diagrams() -> None:
    for name, kinds, title in (
        (
            "FIGURE_1",
            ("mrsr", "srmr"),
            "Figure 1 — (a) MRSR and (b) SRMR architectures, as implemented",
        ),
        (
            "FIGURE_A1",
            ("srsr", "mrmr"),
            "Figure A1 — (a) SRSR and (b) MRMR architectures, as implemented",
        ),
    ):
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        for ax, kind in zip(axes, kinds):
            architecture_panel(ax, kind)
        fig.suptitle(
            f"{title}. Arrows carry classical data only; compare with the paper's figure (not reproduced here).",
            fontsize=10,
        )
        fig.text(
            0.01,
            0.01,
            "Legend: grey box = input window x(t) (4 past hourly loads), aggregation step (concat or mean) or forecast ŷ; blue box = reservoir (fixed, classical or quantum);\n"
            "orange box = ridge readout (trained, linear or quantum kernel). Arrows = classical data (measured features or predictions). k = 3 shown for illustration (the paper uses 2, 3 or 5).",
            fontsize=7.5,
            va="bottom",
        )
        fig.tight_layout(rect=(0, 0.1, 1, 0.97))
        fig.savefig(OUT / f"{name}.png", dpi=150)
        plt.close(fig)
        print(f"wrote {os.path.relpath(OUT / name)}.png")

    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector

    x, w = ParameterVector("x", 4), ParameterVector("w", 4)
    encoding, ansatz = QuantumCircuit(4), QuantumCircuit(4)
    for q in range(4):
        encoding.rx(x[q], q)
    for _ in range(2):
        for q in range(4):
            ansatz.rx(w[q], q)
        for t in range(1, 4):
            ansatz.cx(t - 1, t)
            ansatz.rz(w[t - 1] - w[t], t)
            ansatz.cx(t - 1, t)
        ansatz.barrier()
    circuit_text_figure(
        "FIGURE_2",
        "Figure 2 — quantum neuron: (a) feature map and (b) ansatz with two structural blocks, as implemented",
        [
            (
                "(a) feature map: RX(x_i) per qubit (4 inputs: data lags and other neurons' outputs)",
                encoding,
            ),
            (
                "(b) ansatz: 2 blocks of RX(w_j) and CX-RZ(w_{j-1} - w_j)-CX; output <Z⊗Z⊗Z⊗Z>",
                ansatz,
            ),
        ],
        [
            "Legend: q_0 … q_3 = the 4 qubits of one neuron; Rx(x[i]) = data-encoding rotation of input i; Rx(w[j]), Rz(w[j-1] - w[j]) = rotations with fixed random weights w (reservoir, not trained);",
            "■ and X = CNOT control and target; the shaded bars separate the two structural blocks. The neuron output is the expectation value <Z⊗Z⊗Z⊗Z> measured after the ansatz.",
            "Same gates as lib/quantum_neuron.py; checked against the Qiskit statevector in tests/test_quantum_neuron.py (error < 1e-10). Compare with the paper's Fig. 2.",
        ],
    )
    n, d = 5, 5
    phi = ParameterVector("x", d)
    block = QuantumCircuit(n)
    for layer in range(math.ceil(d / n)):
        for q in range(n):
            block.h(q)
            block.p(2 * phi[layer * n + q], q)
    feature_map = block.compose(block)
    circuit_text_figure(
        "FIGURE_3",
        "Figure 3 — feature map of the quantum-kernel output: (a) two structural blocks, (b) each block = H and P(2x) layers, as implemented (n = 5)",
        [
            (
                "feature map Φ(x) on 5 qubits, d = 5 features, L = 1 layer per block, 2 blocks",
                feature_map,
            )
        ],
        [
            "Legend: q_0 … q_4 = qubits of one kernel instance; H = Hadamard gate; P(2*x[i]) = phase gate encoding feature i (a reservoir output); the second H/P column is the second structural block.",
            "Kernel K(x, x') = |<Φ(x)|Φ(x')>|². No entangling gate, so K = Π_q (1 + b_q(x)·b_q(x'))/2 (closed form, lib/kernels.py; checked against Qiskit in tests/test_kernels.py).",
            "Compare with the paper's Fig. 3.",
        ],
    )


def main() -> int:
    outdir, tex_path = Path(sys.argv[1]), Path(sys.argv[2])
    tex = tex_path.read_text()
    OUT.mkdir(parents=True, exist_ok=True)
    ours, paper = load_ours(outdir), load_paper_cells()
    table_1(tex)
    table_2(tex)
    table_3(tex)
    for arch, number in MAIN_TABLE.items():
        table_main(number, arch, tex, ours, paper)
    table_8(tex, ours, paper)
    table_b(tex)
    for number in C_TABLES:
        table_c(number, ours, paper)
    figures_4_to_7(ours, paper)
    figures_diagrams()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
