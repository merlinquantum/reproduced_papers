"""Plotting for the QARIMA reproduction.

All functions read structured arrays/JSON from a run directory (never hardcode
paths) and save PNGs into that run directory; the runner also copies key figures
into ``results/``.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_C = {
    "classical": "#4C78A8",
    "gate": "#E45756",
    "merlin": "#54A24B",
    "seasonal": "#B279A2",
    "truth": "#222222",
    "paper": "#9D755D",
}


def plot_forecasts(run_dir: Path, dataset: str) -> Path:
    """Predicted time series vs. ground truth for the key models."""
    data = np.load(run_dir / "forecasts.npz", allow_pickle=True)
    meta = json.loads((run_dir / "results.json").read_text())
    y_train_tail = data["y_train_tail"]
    y_true = data["y_true"]
    n_tail = y_train_tail.size
    x_hist = np.arange(-n_tail, 0)
    x_oos = np.arange(len(y_true))

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(x_hist, y_train_tail, color="#999999", lw=1, label="train (history)")
    ax.axvline(-0.5, color="#cccccc", ls="--", lw=1)
    ax.plot(x_oos, y_true, color=_C["truth"], lw=2.2, label="ground truth (OOS)")
    for key, lbl in [
        ("classical_auto", "classical ARIMA (auto)"),
        ("best_gate", "QARIMA gate-VQC (best)"),
        ("best_merlin", "QARIMA MerLin-VQC (best)"),
        ("seasonal", "fair seasonal ARIMA"),
    ]:
        if key in data.files:
            color = (
                _C["classical"]
                if "classical" in key
                else _C["gate"]
                if "gate" in key
                else _C["merlin"]
                if "merlin" in key
                else _C["seasonal"]
            )
            ax.plot(x_oos, data[key], color=color, lw=1.6, ls="-", label=lbl)
    ax.set_title(
        f"{dataset}: OOS forecast vs ground truth  (multi-step, {len(y_true)} points)"
    )
    ax.set_xlabel("time index (0 = first OOS point)")
    ax.set_ylabel(meta.get("unit", "value"))
    ax.legend(loc="best", fontsize=8, ncol=2)
    fig.tight_layout()
    out = run_dir / f"forecast_{dataset}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_order_sweep(run_dir: Path, dataset: str) -> Path | None:
    meta = json.loads((run_dir / "results.json").read_text())
    sweep = meta.get("order_sweep")
    if not sweep:
        return None
    ps = [r["p"] for r in sweep]
    mse = [r["mse"] for r in sweep]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(
        ps, mse, "o-", color=_C["classical"], label="classical OLS AR(p,d,0) multi-step"
    )
    if meta.get("classical_auto"):
        ax.axhline(
            meta["classical_auto"]["mse"],
            color=_C["paper"],
            ls="--",
            label=f"classical auto {tuple(meta['classical_auto']['order'])}",
        )
    best_q = meta.get("paper_best_quantum_mse")
    if best_q:
        ax.axhline(
            best_q, color=_C["gate"], ls=":", label=f"paper best-quantum MSE={best_q}"
        )
    ax.set_xlabel("AR order p")
    ax.set_ylabel("OOS MSE")
    ax.set_yscale("log")
    ax.set_title(
        f"{dataset}: OOS MSE vs AR order (the 'quantum gain' is an order effect)"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = run_dir / f"order_sweep_{dataset}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_refiner_comparison(run_dir: Path, dataset: str) -> Path | None:
    meta = json.loads((run_dir / "results.json").read_text())
    rows = meta.get("candidates", [])
    if not rows:
        return None
    labels = [f"({r['p']},{r['d']},{r['q']})" for r in rows]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.1), 4.5))
    width = 0.26
    for i, ref in enumerate(["classical", "gate", "merlin"]):
        vals = [r["refiners"].get(ref, {}).get("mse_mean", np.nan) for r in rows]
        ax.bar(x + (i - 1) * width, vals, width, label=ref, color=_C[ref])
    if meta.get("classical_auto"):
        ax.axhline(
            meta["classical_auto"]["mse"],
            color=_C["paper"],
            ls="--",
            lw=1,
            label=f"classical auto {tuple(meta['classical_auto']['order'])}",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("OOS MSE")
    ax.set_yscale("log")
    ax.set_title(
        f"{dataset}: refiner comparison at matched (p,d,q) (classical ~ gate ~ MerLin)"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = run_dir / f"refiners_{dataset}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def make_all(run_dir: Path, dataset: str) -> list[Path]:
    outs = [plot_forecasts(run_dir, dataset)]
    for fn in (plot_order_sweep, plot_refiner_comparison):
        p = fn(run_dir, dataset)
        if p:
            outs.append(p)
    return outs
