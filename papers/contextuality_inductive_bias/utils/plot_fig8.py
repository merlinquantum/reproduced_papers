"""Fig. 8-style curves: test KL (log scale) and train NLL vs training step.

Usage (from the repository root)::

    python3 papers/contextuality_inductive_bias/utils/plot_fig8.py \\
        --sweep-dir <sweep_dir> [--sweep-dir <sweep_dir> ...] \\
        --protocol paper --variant paper_settings --label "paper-accurate" \\
        --out papers/contextuality_inductive_bias/results/fig8_paper_protocol.png

Each curve is the mean over seeds (bold) with every seed drawn faintly, as in
the paper's figure. ``--variant paper_settings`` draws the four model classes
at the settings stated in the paper (quantum L = 2, generic with the Fig. 10
CNOTs); ``--variant code_settings`` draws the settings the author notebook
plots (quantum L = 1, generic with the code CNOTs, discrepancies D2/D3);
``--variant all`` draws every quantum variant. The PNG goes to ``--out`` and a
copy to the first sweep directory.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sweep_io import describe, load_sweep, matches  # noqa: E402

# Reference palette, categorical slots 1-4 in fixed order (dataviz skill).
SLOTS = {
    "biased_quantum": "#2a78d6",
    "generic_quantum": "#eb6834",
    "surrogate_1": "#1baf7a",
    "surrogate_2": "#eda100",
    "photonic_vertex": "#4a3aa7",
    "photonic_group": "#008300",
    "photonic_unbiased": "#e87ba4",
}
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

VARIANTS = {
    "paper_settings": [
        {"kind": "biased_quantum", "layers": 2},
        {"kind": "generic_quantum", "layers": 2, "cnot_pattern": "paper"},
        {"kind": "surrogate", "spectrum": 1},
        {"kind": "surrogate", "spectrum": 2},
    ],
    "code_settings": [
        {"kind": "biased_quantum", "layers": 1},
        {"kind": "generic_quantum", "layers": 1, "cnot_pattern": "code"},
        {"kind": "surrogate", "spectrum": 1},
        {"kind": "surrogate", "spectrum": 2},
    ],
    "photonic": [
        {"kind": "biased_quantum", "layers": 2},
        {"kind": "biased_quantum", "layers": 1},
        {"kind": "surrogate", "spectrum": 1},
        {
            "kind": "photonic_merlin",
            "readout": "vertex",
            "n_photons": 2,
            "computation_space": "unbunched",
        },
        {
            "kind": "photonic_merlin",
            "readout": "group",
            "n_photons": 2,
            "computation_space": "unbunched",
        },
        {
            "kind": "photonic_merlin",
            "readout": "unbiased",
            "n_photons": 2,
            "computation_space": "unbunched",
        },
        {
            "kind": "photonic_merlin",
            "readout": "vertex",
            "n_photons": 2,
            "computation_space": "fock",
        },
        {
            "kind": "photonic_merlin",
            "readout": "group",
            "n_photons": 2,
            "computation_space": "fock",
        },
    ],
    "all": [
        {"kind": "biased_quantum", "layers": 1},
        {"kind": "biased_quantum", "layers": 2},
        {"kind": "generic_quantum", "layers": 1, "cnot_pattern": "paper"},
        {"kind": "generic_quantum", "layers": 2, "cnot_pattern": "paper"},
        {"kind": "generic_quantum", "layers": 1, "cnot_pattern": "code"},
        {"kind": "generic_quantum", "layers": 2, "cnot_pattern": "code"},
        {"kind": "surrogate", "spectrum": 1},
        {"kind": "surrogate", "spectrum": 2},
    ],
}


def _style(selector: dict) -> dict:
    """Hue encodes the model class; line style encodes L and the CNOT variant."""
    kind = selector["kind"]
    if kind == "photonic_merlin":
        return {
            "color": SLOTS[f"photonic_{selector['readout']}"],
            "linestyle": "--" if selector["computation_space"] == "fock" else "-",
        }
    if kind == "surrogate":
        return {"color": SLOTS[f"surrogate_{selector['spectrum']}"], "linestyle": "-"}
    if kind == "generic_quantum" and selector["cnot_pattern"] == "code":
        return {
            "color": SLOTS[kind],
            "linestyle": ":" if selector["layers"] == 1 else "-.",
        }
    return {"color": SLOTS[kind], "linestyle": "--" if selector["layers"] == 1 else "-"}


def plot(
    sweep_dirs: list[Path], protocol: str, variant: str, label: str, out: Path
) -> Path:
    """Draw the figure and return the written path.

    Raises
    ------
    ValueError
        If a requested model class has no completed run for ``protocol``.
    """
    records = [
        r
        for d in sweep_dirs
        for r in load_sweep(d)
        if r.candidate["protocol"] == protocol and r.candidate["l2"] == 0.0
    ]
    fig, (ax_kl, ax_nll) = plt.subplots(1, 2, figsize=(12, 4.6))
    summary = []
    for selector in VARIANTS[variant]:
        runs = [r for r in records if matches(r.candidate, selector)]
        if not runs:
            raise ValueError(
                f"No completed run for {selector} under protocol {protocol}"
            )
        histories = [r.history() for r in runs]
        steps = np.array(histories[0]["step"])
        kl = np.array([h["test_kl"] for h in histories])
        nll = np.array([h["train_objective"] for h in histories])
        style = _style(selector)
        name = f"{describe(runs[0].candidate)} (n={len(runs)})"
        for row_kl, row_nll in zip(kl, nll):
            ax_kl.plot(steps, row_kl, color=style["color"], alpha=0.12, linewidth=0.8)
            ax_nll.plot(steps, row_nll, color=style["color"], alpha=0.12, linewidth=0.8)
        ax_kl.plot(steps, kl.mean(axis=0), linewidth=2, label=name, **style)
        ax_nll.plot(steps, nll.mean(axis=0), linewidth=2, label=name, **style)
        summary.append((name, kl.mean(axis=0)[-1]))
    for ax, ylabel in (
        (ax_kl, "KL divergence (test)"),
        (ax_nll, "negative log likelihood (train)"),
    ):
        ax.set_xlabel("training step", color=TEXT)
        ax.set_ylabel(ylabel, color=TEXT)
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.tick_params(colors=MUTED)
        for spine in ax.spines.values():
            spine.set_color(GRID)
    ax_kl.set_yscale("log")
    ax_kl.set_ylim(3e-3, 6e-2)
    ax_nll.set_ylim(0.65, 0.73)
    ax_nll.legend(frameon=False, fontsize=8, labelcolor=TEXT)
    fig.suptitle(
        f"RPS zero-sum game, protocol '{protocol}' ({label}); mean over seeds (bold), single seeds (faint)",
        color=TEXT,
        fontsize=10,
    )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    shutil.copy(out, Path(sweep_dirs[0]) / out.name)
    for name, final in summary:
        print(f"{name}: final mean test KL {final:.4e}")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sweep-dir", type=Path, action="append", required=True)
    parser.add_argument("--protocol", choices=["paper", "code", "fork"], required=True)
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="paper_settings")
    parser.add_argument(
        "--label", required=True, help="Result-scope label, e.g. 'paper-accurate'"
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    plot(args.sweep_dir, args.protocol, args.variant, args.label, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
