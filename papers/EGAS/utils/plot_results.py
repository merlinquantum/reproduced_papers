"""Plot EGAS reproduction figures from run metrics.

Usage:
    python utils/plot_results.py --wasserstein outdir/run_*/metrics.json
    python utils/plot_results.py --fig1 outdir/run_*/metrics.json
    python utils/plot_results.py --egas outdir/run_WC/metrics.json outdir/run_WQ/metrics.json

Reads structured metrics.json (never hardcodes run dirs) and writes PNGs to both the run dir
and ``results/``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)

PAPER_W1 = {
    "PW": 5.2380,
    "WDGV1": 5.1570,
    "DB": 13.9108,
    "WC": 10.8562,
    "WQ": 3.0112,
    "MGT": 3.3036,
    "EGSSD": 3.5619,
}


def _load(p):
    return json.loads(Path(p).read_text())


def plot_wasserstein(path):
    m = _load(path)["results"]
    names = [n for n in m if "w1" in m[n]]
    repro = [m[n]["w1"] for n in names]
    paper = [PAPER_W1.get(n, np.nan) for n in names]
    x = np.arange(len(names))
    plt.figure(figsize=(8, 4))
    plt.bar(x - 0.2, repro, 0.4, label="reproduced")
    plt.bar(x + 0.2, paper, 0.4, label="paper (Table I)")
    plt.xticks(x, names)
    plt.ylabel("1-Wasserstein distance")
    plt.legend()
    plt.title("Table I: input-space class-conditional W1")
    plt.tight_layout()
    for d in (Path(path).parent, RESULTS):
        plt.savefig(d / "table1_wasserstein.png", dpi=120)
    plt.close()
    print("wrote table1_wasserstein.png")


def plot_fig1(path):
    res = _load(path)["results"]
    plt.figure(figsize=(6, 4))
    for key, d in res.items():
        plt.plot(d["w1"], d["trace_dist"], "o-", label=f"ZZ {key}")
    plt.xlabel("input W1 distance")
    plt.ylabel("trace distance")
    plt.title("Fig 1: trace distance vs input W1 (saturating)")
    plt.legend()
    plt.tight_layout()
    for d in (Path(path).parent, RESULTS):
        plt.savefig(d / "fig1_tracedist_vs_w1.png", dpi=120)
    plt.close()
    print("wrote fig1_tracedist_vs_w1.png")


def plot_egas(paths):
    data = [_load(p) for p in paths]
    names = [d["dataset"] for d in data]

    # accuracy comparison (best G(bias), NQE, ZZ, classical linear)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    x = np.arange(len(names))
    bestGb = [max(g["mean_acc"] for g in d["G_bias"]) for d in data]
    nqe = [d["baselines"]["NQE"]["mean_acc"] for d in data]
    zz = [d["baselines"]["ZZ"]["mean_acc"] for d in data]
    lin = [d["baselines"]["classical_linear"]["mean_acc"] for d in data]
    w = 0.2
    ax[0].bar(x - 1.5 * w, bestGb, w, label="best G*(bias)")
    ax[0].bar(x - 0.5 * w, nqe, w, label="NQE")
    ax[0].bar(x + 0.5 * w, zz, w, label="ZZ")
    ax[0].bar(x + 1.5 * w, lin, w, label="classical linear")
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(names)
    ax[0].set_ylabel("mean test acc")
    ax[0].set_ylim(0.4, 1.0)
    ax[0].legend()
    ax[0].set_title("QKSVM accuracy (Fig 7)")

    # IQR vs W1 (Fig 6 diagnostic)
    iqr = [d["embedding_sensitivity_IQR"] for d in data]
    w1 = [d["w1"] for d in data]
    ax[1].scatter(w1, iqr)
    for xi, yi, ni in zip(w1, iqr, names):
        ax[1].annotate(ni, (xi, yi))
    ax[1].set_xlabel("input W1 distance")
    ax[1].set_ylabel("embedding-sensitivity IQR")
    ax[1].set_title("Fig 6: IQR vs W1 (larger W1 -> larger IQR)")
    plt.tight_layout()
    for d in (RESULTS,):
        plt.savefig(d / "egas_summary.png", dpi=120)
    for p in paths:
        plt.savefig(Path(p).parent / "egas_summary.png", dpi=120)
    plt.close()
    print("wrote egas_summary.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--wasserstein")
    ap.add_argument("--fig1")
    ap.add_argument("--egas", nargs="+")
    a = ap.parse_args()
    if a.wasserstein:
        plot_wasserstein(a.wasserstein)
    if a.fig1:
        plot_fig1(a.fig1)
    if a.egas:
        plot_egas(a.egas)
