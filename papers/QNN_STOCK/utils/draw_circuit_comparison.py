"""Draw the paper HEA layer and the context-preserving circuit side by side.

Two PNGs at the same scale for slides:

- ``fig_circuit_hea.png``: one layer of the paper's QNN ansatz (Eq. 17), RY
  and RZ on every qubit followed by the CNOT ring, repeated 4 times. Every
  gate column touches the context wires.
- ``fig_circuit_context_preserving.png``: the order-1 context-preserving
  circuit (E11), where the context wires carry only control dots and every
  rotation acts on the prediction wire.

Usage::

    python papers/QNN_STOCK/utils/draw_circuit_comparison.py \
        --out results/readout_per_context
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

INK = "#14213D"
MUTED = "#5B6474"
CONTEXT_GATE = "#B4491B"
CONTEXT_FILL = "#FBE9E1"
PREDICTION_GATE = "#1BAF7A"
PREDICTION_FILL = "#E3F5EE"
HIGHLIGHT = "#FBE9E1"

WIRES = [
    (3.0, r"$c_1$"),
    (2.0, r"$c_2$"),
    (1.0, r"$c_3$"),
    (0.0, r"$p$"),
]
PREDICTION_Y = 0.0


def _gate(ax, x, y, label, on_context):
    edge, fill = (
        (CONTEXT_GATE, CONTEXT_FILL)
        if on_context
        else (PREDICTION_GATE, PREDICTION_FILL)
    )
    ax.add_patch(
        FancyBboxPatch(
            (x - 0.36, y - 0.3),
            0.72,
            0.6,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor=fill,
            edgecolor=edge,
            linewidth=1.5,
            zorder=4,
        )
    )
    ax.text(x, y, label, ha="center", va="center", fontsize=12, color=INK, zorder=5)


def _wires(ax, x_end, highlight_context):
    if highlight_context:
        ax.add_patch(
            Rectangle(
                (-0.1, 0.55),
                x_end + 0.2,
                2.9,
                facecolor="#EEF3FA",
                edgecolor="none",
                zorder=0,
            )
        )
    for y, label in WIRES:
        ax.plot([0, x_end], [y, y], color=INK, linewidth=1.2, zorder=1)
        ax.text(-0.2, y, label, ha="right", va="center", fontsize=15, color=INK)


def _cnot(ax, x, control_y, target_y):
    ax.plot([x, x], [control_y, target_y], color=INK, linewidth=1.2, zorder=2)
    ax.scatter([x], [control_y], s=45, color=INK, zorder=3)
    ax.add_patch(
        plt.Circle(
            (x, target_y),
            0.17,
            facecolor="white",
            edgecolor=INK,
            linewidth=1.2,
            zorder=3,
        )
    )
    ax.plot(
        [x - 0.17, x + 0.17], [target_y, target_y], color=INK, linewidth=1.2, zorder=4
    )
    ax.plot(
        [x, x], [target_y - 0.17, target_y + 0.17], color=INK, linewidth=1.2, zorder=4
    )


def draw_hea(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    x_end = 6.2
    _wires(ax, x_end, highlight_context=False)
    for y, _ in WIRES:
        on_context = y != PREDICTION_Y
        _gate(ax, 0.7, y, r"$R_y$", on_context)
        _gate(ax, 1.6, y, r"$R_z$", on_context)
    # Ring of CNOTs, qubit j controls qubit j + 1 (mod 4), top wire is qubit 0.
    ring = [(3.0, 2.0), (2.0, 1.0), (1.0, 0.0), (0.0, 3.0)]
    for index, (control_y, target_y) in enumerate(ring):
        _cnot(ax, 2.6 + 0.75 * index, control_y, target_y)
    ax.plot([0.25, 0.25], [-0.55, 3.55], color=MUTED, linewidth=1.0, linestyle="--")
    ax.plot([5.45, 5.45], [-0.55, 3.55], color=MUTED, linewidth=1.0, linestyle="--")
    ax.text(5.6, 3.55, r"$\times 4$", fontsize=15, color=MUTED, va="center")
    ax.set_xlim(-1.0, 6.6)
    ax.set_ylim(-0.75, 3.8)
    ax.axis("off")
    fig.savefig(
        out / "fig_circuit_hea.png", dpi=220, bbox_inches="tight", transparent=True
    )
    plt.close(fig)


def draw_context_preserving(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    x_end = 6.2
    _wires(ax, x_end, highlight_context=False)
    gates = [
        (0.9, None, r"$R_y(\theta_0)$"),
        (2.3, 3.0, r"$R_y(\theta_1)$"),
        (3.7, 2.0, r"$R_y(\theta_2)$"),
        (5.1, 1.0, r"$R_y(\theta_3)$"),
    ]
    for x, control_y, label in gates:
        if control_y is not None:
            ax.plot([x, x], [control_y, 0.3], color=INK, linewidth=1.2, zorder=2)
            ax.scatter([x], [control_y], s=45, color=INK, zorder=3)
        ax.add_patch(
            FancyBboxPatch(
                (x - 0.55, -0.3),
                1.1,
                0.6,
                boxstyle="round,pad=0.02,rounding_size=0.08",
                facecolor=PREDICTION_FILL,
                edgecolor=PREDICTION_GATE,
                linewidth=1.5,
                zorder=4,
            )
        )
        ax.text(
            x, 0.0, label, ha="center", va="center", fontsize=11, color=INK, zorder=5
        )
    ax.set_xlim(-1.0, 6.6)
    ax.set_ylim(-0.75, 3.8)
    ax.axis("off")
    fig.savefig(
        out / "fig_circuit_context_preserving.png",
        dpi=220,
        bbox_inches="tight",
        transparent=True,
    )
    plt.close(fig)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    plt.rcParams["mathtext.fontset"] = "cm"
    args.out.mkdir(parents=True, exist_ok=True)
    draw_hea(args.out)
    draw_context_preserving(args.out)
    print(f"circuits written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
