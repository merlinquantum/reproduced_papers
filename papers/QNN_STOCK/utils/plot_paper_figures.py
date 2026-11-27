"""Redraw the paper's result figures (Fig. 11-19, 21) in the paper's layout, plus Table 1.

Layout follows the published figures: default Matplotlib look, bars for the
generated distribution, a starred line for the original one, "Qubit States"
on x, and one loss panel per figure. Uses one seed of a completed sweep
(default 42, the paper's seed) and reads only ``distributions.json``,
``losses.csv``, ``params.npz`` and ``metrics.json``. Figures go to
``<sweep_dir>/figures`` and ``--results``; the Table 1 comparison goes to
``--results/table1_kl.csv``.

Usage::

    python papers/QNN_STOCK/utils/plot_paper_figures.py <sweep_dir> --results results/paper_figures
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PAPER_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PAPER_DIR.parents[1]))
sys.path.insert(0, str(PAPER_DIR))

from lib.circuits import apply_ansatz  # noqa: E402
from lib.statevector import basis_state, probabilities  # noqa: E402

PAPER_TABLE1 = {
    ("AAPL", "STL"): 0.1047,
    ("GOOGL", "STL"): 0.1239,
    ("AAPL", "MTL"): 0.0614,
    ("GOOGL", "MTL"): 0.0754,
}
KL_VARIANTS = (
    "kl_joint",
    "kl_joint_reverse",
    "kl_conditional_sum",
    "kl_conditional_sum_reverse",
    "kl_conditional_mean",
    "kl_conditional_weighted",
)
NAMES = {
    "AAPL": "Apple",
    "GOOGL": "Google",
    "MSFT": "Microsoft",
    "AMZN": "Amazon",
    "PEP": "PepsiCo",
    "WDC": "Western Digital",
    "TXN": "Texas Instruments",
    "IBM": "IBM",
}
# Apple and Google colors as in the paper's figures; the others are free choices.
COLORS = {
    "AAPL": "dodgerblue",
    "GOOGL": "gold",
    "MSFT": "mediumseagreen",
    "AMZN": "orangered",
    "PEP": "mediumpurple",
    "WDC": "teal",
    "TXN": "crimson",
    "IBM": "slategray",
}
XLABEL = "Qubit States"
YLABEL = "Normalized Counts after Measurement"


class Sweep:
    def __init__(self, sweep_dir: Path) -> None:
        status = json.loads(
            (sweep_dir / "sweep_status.json").read_text(encoding="utf-8")
        )
        self.runs = {
            (r["candidate"]["config"], r["seed"]): sweep_dir / r["run_dir"]
            for r in status["runs"]
        }
        self.seeds = sorted({seed for _, seed in self.runs})

    def json(self, config: str, seed: int, name: str) -> dict:
        path = self.runs[(config, seed)] / name
        return json.loads(path.read_text(encoding="utf-8"))

    def losses(self, config: str, seed: int) -> list[dict]:
        path = self.runs[(config, seed)] / "losses.csv"
        with path.open(encoding="utf-8") as handle:
            return list(csv.DictReader(handle))


def _paper_order(joint) -> tuple[np.ndarray, list[str]]:
    """Reorder a ``(contexts, 2)`` joint to the paper's 4-bit state labels.

    The paper's Fig. 11 labels the 3-qubit contexts 0000-0111, so the leading
    bit of Fig. 13's labels is read as the prediction qubit followed by the
    three context bits (LOG.md, Reproducibility Assessment, inferred).
    """

    joint = np.asarray(joint)
    n_contexts = joint.shape[0]
    width = int(np.log2(n_contexts))
    labels = [
        f"{outcome}{np.binary_repr(context, width)}"
        for outcome in range(2)
        for context in range(n_contexts)
    ]
    return joint.T.reshape(-1), labels


def _save(fig, name: str, outputs: list[Path]) -> None:
    fig.tight_layout()
    for out in outputs:
        out.mkdir(parents=True, exist_ok=True)
        fig.savefig(out / f"{name}.png", dpi=150)
    plt.close(fig)


def _distribution_panel(ax, asset: str, target, generated, possessive: bool) -> None:
    """One paper-style panel: translucent bars (generated), starred line (original)."""

    target_values, labels = _paper_order(target)
    generated_values, _ = _paper_order(generated)
    prefix = f"{NAMES[asset]}'s " if possessive else ""
    x = np.arange(len(labels))
    ax.plot(
        x,
        target_values,
        color=COLORS[asset],
        marker="*",
        linewidth=2,
        label=f"{prefix}Original Likelihood Distribution",
    )
    ax.bar(
        x,
        generated_values,
        color=COLORS[asset],
        alpha=0.6,
        label=f"{prefix}Generated Likelihood Distribution",
    )
    ax.set_xticks(x, labels)
    ax.set_xlabel(XLABEL)
    ax.set_ylabel(YLABEL)


def plot_distributions(
    sweep: Sweep, seed: int, config: str, model: str, name: str, outputs: list[Path]
) -> None:
    """Fig. 13, 14, 15, 21: one 10x6 panel per asset, in the config's ticker order."""

    tables = sweep.json(config, seed, "distributions.json")
    order = sweep.json(config, seed, "config_snapshot.json")["dataset"]["tickers"]
    rows = (len(order) + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(20, 6 * rows), squeeze=False)
    for ax, asset in zip(axes.ravel(), order):
        table = tables[asset]
        _distribution_panel(
            ax,
            asset,
            table["train_joint"],
            table[f"{model}_output_joint"],
            possessive=len(order) > 2,
        )
        if len(order) <= 2:
            ax.set_title(NAMES[asset])
        ax.legend(loc="upper left")
    _save(fig, name, outputs)


def plot_fig11(sweep: Sweep, seed: int, outputs: list[Path]) -> None:
    """Fig. 11: loaded context distribution, labels 0000-0111 as in the paper."""

    run = sweep.runs[("e2_loader_original", seed)]
    params = np.load(run / "params.npz")["loader_AAPL"]
    loaded = probabilities(apply_ansatz(basis_state(0, 3), params, 3))
    tables = sweep.json("e2_loader_original", seed, "distributions.json")
    target = np.asarray(tables["AAPL"]["train_joint"]).sum(axis=1)
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(8)
    ax.plot(
        x,
        target,
        color="orangered",
        marker="*",
        linewidth=2,
        label="Apple's Original Prior Distribution",
    )
    ax.bar(
        x,
        loaded,
        color="orangered",
        alpha=0.6,
        label="Apple's Generated Prior Distribution",
    )
    ax.set_xticks(x, [f"0{np.binary_repr(c, 3)}" for c in x])
    ax.set_xlabel(XLABEL)
    ax.set_ylabel(YLABEL)
    ax.legend(loc="upper left")
    _save(fig, "paper_fig11_loader_distribution", outputs)


def _loss_axes(ylabel: str):
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.set_xlabel("Number of Epochs")
    ax.set_ylabel(ylabel)
    ax.grid(True)
    return fig, ax


def _curve(rows: list[dict], x_key: str) -> tuple[list[int], list[float]]:
    return [int(r[x_key]) for r in rows], [float(r["loss_estimate"]) for r in rows]


def plot_fig12(sweep: Sweep, seed: int, outputs: list[Path]) -> None:
    """Fig. 12: loader training loss (MSE, 10,000 shots)."""

    history = sweep.losses("e2_loader_original", seed)
    rows = [r for r in history if r["stage"] == "loader_AAPL"]
    fig, ax = _loss_axes("Loss Function (MSE)")
    ax.plot(*_curve(rows, "stage_epoch"), color="red", linewidth=2)
    _save(fig, "paper_fig12_loader_loss", outputs)


def plot_fig16(sweep: Sweep, seed: int, outputs: list[Path]) -> None:
    """Fig. 16: STL (first 500 epochs) and MTL (250 per asset) SWAP-test loss, Eq. 14."""

    fig, ax = _loss_axes("Loss Function (Swap Test)")
    stl = sweep.losses("e3_qstl_original", seed)
    mtl = sweep.losses("e4_qmtl_original", seed)
    series = (
        ("Apple (STL)", stl, "qstl_AAPL", "blue"),
        ("Google (STL)", stl, "qstl_GOOGL", "orange"),
        ("Apple (MTL)", mtl, "qmtl_cycle1_AAPL", "dodgerblue"),
        ("Google (MTL)", mtl, "qmtl_cycle1_GOOGL", "gold"),
    )
    for label, history, stage, color in series:
        rows = [
            r for r in history if r["stage"] == stage and int(r["stage_epoch"]) <= 500
        ]
        ax.plot(*_curve(rows, "stage_epoch"), color=color, linewidth=2, label=label)
    ax.legend(loc="upper right")
    _save(fig, "paper_fig16_stl_mtl_loss", outputs)


def plot_fig18(sweep: Sweep, seed: int, outputs: list[Path]) -> None:
    """Fig. 18: QMTL K=4 loss over the 1000 sequential epochs."""

    rows = sweep.losses("e7_qmtl_k4_original", seed)
    fig, ax = _loss_axes("Loss Function (Swap Test)")
    ax.plot(*_curve(rows, "epoch"), color="red", linewidth=2)
    _save(fig, "paper_fig18_qmtl_k4_loss", outputs)


def plot_fig17(sweep: Sweep, seed: int, outputs: list[Path]) -> None:
    """Fig. 17: MSE-trained output, Apple and Google overlaid in one panel."""

    tables = sweep.json("e6_qstl_mse_original", seed, "distributions.json")
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(16)
    labels: list[str] = []
    for asset in ("AAPL", "GOOGL"):
        values, labels = _paper_order(tables[asset]["train_joint"])
        ax.plot(
            x,
            values,
            color=COLORS[asset],
            marker="*",
            linewidth=2,
            label=f"{NAMES[asset]}'s Original Likelihood Distribution",
        )
    for asset in ("AAPL", "GOOGL"):
        values, _ = _paper_order(tables[asset]["QSTL_output_joint"])
        ax.bar(
            x,
            values,
            color=COLORS[asset],
            alpha=0.6,
            label=f"{NAMES[asset]}'s Generated Likelihood Distribution",
        )
    ax.set_xticks(x, labels)
    ax.set_xlabel(XLABEL)
    ax.set_ylabel(YLABEL)
    ax.legend(loc="upper left")
    _save(fig, "paper_fig17_mse", outputs)


def plot_fig19(sweep: Sweep, seed: int, outputs: list[Path]) -> None:
    """Fig. 19: d=4, T=2; states indexed 0-63, next level most significant."""

    table = sweep.json("e8_qstl_d4", seed, "distributions.json")["AAPL"]
    target = np.asarray(table["train_joint"]).T.reshape(-1)
    generated = np.asarray(table["QSTL_output_joint"]).T.reshape(-1)
    fig, ax = plt.subplots(figsize=(10, 4.6))
    x = np.arange(target.size)
    ax.plot(
        x,
        target,
        color="dodgerblue",
        marker="*",
        linewidth=2,
        label="Original Conditional Distribution",
    )
    ax.bar(
        x,
        generated,
        color="gold",
        alpha=0.6,
        edgecolor="black",
        label="Generated Conditional Distribution",
    )
    ax.grid(axis="y", alpha=0.3)
    ax.set_xlabel(XLABEL)
    ax.set_ylabel(YLABEL)
    ax.legend(loc="upper left")
    _save(fig, "paper_fig19_d4", outputs)


def write_table1(sweep: Sweep, seed: int, out: Path) -> None:
    """Every KL definition for the Table 1 cells: seed value, mean and std over seeds."""

    rows = []
    cells = (("STL", "e3_qstl_original", "QSTL"), ("MTL", "e4_qmtl_original", "QMTL"))
    for asset in ("AAPL", "GOOGL"):
        for label, config, model in cells:
            for readout in ("output", "inference"):
                for variant in KL_VARIANTS:
                    values = {}
                    for s in sweep.seeds:
                        metrics = sweep.json(config, s, "metrics.json")
                        entry = metrics["assets"][asset]["models"][model]
                        values[s] = entry[readout]["train"][variant]
                    rows.append(
                        {
                            "asset": asset,
                            "model": label,
                            "paper_kl": PAPER_TABLE1[(asset, label)],
                            "readout": readout,
                            "kl_definition": variant,
                            f"seed_{seed}": values[seed],
                            "mean": statistics.fmean(values.values()),
                            "std": statistics.stdev(values.values()),
                        }
                    )
    out.mkdir(parents=True, exist_ok=True)
    with (out / "table1_kl.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sweep_dir", type=Path)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    plt.rcdefaults()
    sweep = Sweep(args.sweep_dir)
    outputs = [args.sweep_dir / "figures", args.results]
    seed = args.seed
    plot_fig11(sweep, seed, outputs)
    plot_fig12(sweep, seed, outputs)
    for config, model, name in (
        ("e3_qstl_original", "QSTL", "paper_fig13_qstl"),
        ("e4_qmtl_original", "QMTL", "paper_fig14_qmtl_k2"),
        ("e7_qmtl_k4_original", "QMTL", "paper_fig15_qmtl_k4"),
        ("e7_qmtl_k8_original", "QMTL", "paper_fig21_qmtl_k8"),
    ):
        plot_distributions(sweep, seed, config, model, name, outputs)
    plot_fig16(sweep, seed, outputs)
    plot_fig17(sweep, seed, outputs)
    plot_fig18(sweep, seed, outputs)
    plot_fig19(sweep, seed, outputs)
    write_table1(sweep, seed, args.results)
    print(f"paper figures and table1_kl.csv written to {args.results}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
