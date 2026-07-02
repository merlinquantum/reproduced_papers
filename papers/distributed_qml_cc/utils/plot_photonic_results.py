"""Plot photonic MerLin results in the same style as the gate-model figures.

Inputs: the four ``metrics.json`` files under ``results/photonic/``
(``merlin_baseline.json``, ``merlin_nc.json``, ``merlin_cc.json``,
``merlin_qc.json``).

Outputs (under ``results/photonic/``):

* ``fig_photonic_training_curves.png`` — analogous to ``fig4c`` of the
  paper: validation accuracy versus training iteration for the four
  schemes, mean ± std over the seeds present in each file.
* ``fig_photonic_acc_bar.png`` — analogous to ``fig4d``: final
  validation accuracy per scheme.
* ``photonic_results_table.md`` — Markdown table of the headline
  numbers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCHEME_ORDER = ["baseline", "nc", "cc", "qc"]
SCHEME_LABELS = {
    "baseline": "non-DQML (m=8, n=3)",
    "nc": "NC-DQML (2 × m=8, n=3)",
    "cc": "CC-DQML (NC + classical FF)",
    "qc": "QC-DQML (m=16, n=6)",
}
SCHEME_COLORS = {
    "baseline": "#888888",
    "nc": "#1f77b4",
    "cc": "#d62728",
    "qc": "#2ca02c",
}


def _load(metrics_dir: Path) -> dict[str, dict]:
    return {
        "baseline": json.loads((metrics_dir / "merlin_baseline.json").read_text()),
        "nc": json.loads((metrics_dir / "merlin_nc.json").read_text()),
        "cc": json.loads((metrics_dir / "merlin_cc.json").read_text()),
        "qc": json.loads((metrics_dir / "merlin_qc.json").read_text()),
    }


def plot_training_curves(data: dict[str, dict], outdir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for scheme in SCHEME_ORDER:
        d = data[scheme]
        seed_hists = [r["history"]["val_acc"] for r in d["runs"]]
        if not seed_hists:
            continue
        iters = np.array(d["runs"][0]["history"]["iteration"], dtype=float)
        accs = np.stack([np.array(h) for h in seed_hists], axis=0)
        mean = accs.mean(0)
        std = accs.std(0)
        ax.plot(iters, mean, label=SCHEME_LABELS[scheme],
                color=SCHEME_COLORS[scheme], lw=2)
        ax.fill_between(iters, mean - std, mean + std,
                        color=SCHEME_COLORS[scheme], alpha=0.18)
    ax.set_xlabel("Training iteration")
    ax.set_ylabel("Validation accuracy")
    ax.set_ylim(0.45, 1.02)
    ax.set_title("Photonic MerLin reproduction — training curves\n"
                 "(analogous to Fig. 4c of arXiv:2408.16327)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    path = outdir / "fig_photonic_training_curves.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_accuracy_bar(data: dict[str, dict], outdir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    means = []
    stds = []
    labels = []
    colors = []
    for scheme in SCHEME_ORDER:
        d = data[scheme]
        accs = np.array([r["final_val_acc"] for r in d["runs"]])
        means.append(accs.mean())
        stds.append(accs.std())
        labels.append(SCHEME_LABELS[scheme])
        colors.append(SCHEME_COLORS[scheme])
    x = np.arange(len(means))
    ax.bar(x, means, yerr=stds, color=colors, edgecolor="black",
           capsize=6, width=0.55)
    for xi, mu, sd in zip(x, means, stds):
        ax.text(xi, mu + sd + 0.012, f"{mu * 100:.1f}%",
                ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=10, ha="right")
    ax.set_ylim(0.45, 1.02)
    ax.set_ylabel("Validation accuracy (mean ± std, 3 seeds)")
    ax.set_title("Photonic MerLin reproduction — scheme comparison\n"
                 "(analogous to Fig. 4d of arXiv:2408.16327)")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    path = outdir / "fig_photonic_acc_bar.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def write_table(data: dict[str, dict], outdir: Path) -> Path:
    lines = [
        "| Scheme | Photonic geometry | Params | Final val acc | Best val acc |",
        "|---|---|---:|---:|---:|",
    ]
    for scheme in SCHEME_ORDER:
        d = data[scheme]
        params = d["runs"][0]["n_params"]
        finals = np.array([r["final_val_acc"] for r in d["runs"]])
        bests = np.array([r["best_val_acc"] for r in d["runs"]])
        if scheme == "baseline":
            geom = f"1 × m={d['n_modes']}, n={d['n_photons']}"
        elif scheme == "qc":
            geom = (f"1 × m={2 * d['n_modes_per_chip']}, "
                    f"n={2 * d['n_photons_per_chip']}")
        else:
            extra = " + classical FF" if scheme == "cc" else ""
            geom = (f"2 × m={d['n_modes_per_chip']}, "
                    f"n={d['n_photons_per_chip']}{extra}")
        lines.append(
            f"| {SCHEME_LABELS[scheme]} | {geom} | {params} | "
            f"{finals.mean() * 100:.2f} ± {finals.std() * 100:.2f} | "
            f"{bests.mean() * 100:.2f} ± {bests.std() * 100:.2f} |"
        )
    path = outdir / "photonic_results_table.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-dir", type=str, default="results/photonic")
    args = parser.parse_args()
    metrics_dir = Path(args.metrics_dir).resolve()
    data = _load(metrics_dir)
    p1 = plot_training_curves(data, metrics_dir)
    p2 = plot_accuracy_bar(data, metrics_dir)
    p3 = write_table(data, metrics_dir)
    print(f"Wrote {p1}")
    print(f"Wrote {p2}")
    print(f"Wrote {p3}")


if __name__ == "__main__":
    main()
