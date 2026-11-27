"""Per-context view of the readout gap: P(up | context) from each readout.

Two figures for one (asset, seed) on MA5 data:

- ``fig_readout_per_context.png``: paper HEA QSTL. Left, ``q(up | c)`` read
  from the circuit's output on the superposed training input (the state the
  loss sees); right, ``q(up | c)`` from loading one context at a time (the
  paper's inference). Both against the empirical training ``P(up | c)``.
- ``fig_context_preserving_per_context.png``: the context-preserving order-1
  circuit (E11, one context at a time) against logistic regression B2 and the
  empirical ``P(up | c)``.

Contexts are the last three moves, oldest first, 1 = up; the bar width is
not weighted, so each context's training share is printed under its label.

Usage::

    python papers/QNN_STOCK/utils/plot_readout_per_context.py \
        --s1 <S1 sweep dir> --s2 <S2 sweep dir> --out results/readout_per_context
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

PAPER_DIR = Path(__file__).resolve().parents[1]
# lib/__init__ imports the shared runtime_lib from the repository root.
sys.path.insert(0, str(PAPER_DIR.parents[1]))
sys.path.insert(0, str(PAPER_DIR))
sys.path.insert(0, str(PAPER_DIR / "utils"))

from lib.metrics import normalize_rows  # noqa: E402
from plot_phase3_s1 import INK, INK_2, SERIES, _save, _style, plt  # noqa: E402

UP = 1


def _run_dir(sweep_dir: Path, config: str, seed: int) -> Path:
    status = json.loads((sweep_dir / "sweep_status.json").read_text(encoding="utf-8"))
    for run in status["runs"]:
        if run["candidate"]["config"] == config and run["seed"] == seed:
            return (sweep_dir / run["metrics_path"]).parent
    raise KeyError(f"{config} seed {seed} not in {sweep_dir}")


def _tables(sweep_dir: Path, config: str, seed: int, asset: str) -> dict:
    path = _run_dir(sweep_dir, config, seed) / "distributions.json"
    return json.loads(path.read_text(encoding="utf-8"))[asset]


def _context_labels(train_joint: np.ndarray) -> list[str]:
    n_contexts = train_joint.shape[0]
    width = int(np.log2(n_contexts))
    share = train_joint.sum(axis=1) / train_joint.sum()
    return [
        f"{context:0{width}b}\n{100 * share[context]:.0f}%"
        for context in range(n_contexts)
    ]


def _bars(ax, labels, truth, series, title):
    x = np.arange(len(labels))
    width = 0.8 / len(series)
    for index, (name, values, color) in enumerate(series):
        ax.bar(
            x + (index - (len(series) - 1) / 2) * width,
            values,
            width,
            color=color,
            label=name,
        )
    ax.scatter(
        x,
        truth,
        marker="_",
        s=900,
        color=INK,
        linewidths=2.5,
        zorder=3,
        label="Data: P(up | context)",
    )
    ax.axhline(0.5, color=INK_2, linewidth=0.8, linestyle=":")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Context: last 3 moves, oldest first (1 = up) · training share")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--s1", type=Path, required=True)
    parser.add_argument("--s2", type=Path, required=True)
    parser.add_argument(
        "--s8",
        type=Path,
        default=None,
        help="S8 sweep; adds fig_qbgu_vs_per_sample_per_context.png",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--asset", default="AAPL")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    outputs = [args.out]
    _style()

    hea = _tables(args.s1, "e3_qstl_ma5", args.seed, args.asset)
    train_joint = np.asarray(hea["train_joint"])
    truth = normalize_rows(train_joint)[:, UP]
    labels = _context_labels(train_joint)
    joint_readout = normalize_rows(np.asarray(hea["QSTL_output_joint"]))[:, UP]
    inference = np.asarray(hea["QSTL_inference_conditional"])[:, UP]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    _bars(
        axes[0],
        labels,
        truth,
        [("Circuit, superposed input", joint_readout, SERIES[0])],
        "All contexts at once (what the loss sees): matches",
    )
    _bars(
        axes[1],
        labels,
        truth,
        [("Circuit, one context at a time", inference, SERIES[1])],
        "One context at a time (the paper's inference): ≈ coin flip",
    )
    axes[0].set_ylabel("Predicted P(up | context)")
    for ax in axes:
        ax.legend(loc="upper left", fontsize=8, frameon=False)
    fig.text(
        0.01,
        -0.03,
        f"Paper HEA QSTL (3 context + 1 prediction qubit) · {args.asset} · "
        f"5-day moving average · seed {args.seed} · training fidelity "
        f"{_fidelity(args.s1, 'e3_qstl_ma5', args.seed, args.asset):.3f}",
        fontsize=8,
        color=INK_2,
    )
    fig.tight_layout()
    _save(fig, "fig_readout_per_context", outputs)

    preserving = _tables(args.s2, "e11_qstl_cc1_ma5", args.seed, args.asset)
    baselines = _tables(args.s2, "e3_baselines_ma5", args.seed, args.asset)
    if not np.allclose(np.asarray(preserving["train_joint"]), train_joint):
        raise ValueError("S1 and S2 training targets differ")
    cc1 = np.asarray(preserving["QSTL_inference_conditional"])[:, UP]
    logistic = np.asarray(baselines["B2_logistic_conditional"])[:, UP]
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    _bars(
        ax,
        labels,
        truth,
        [
            (
                "Context-preserving circuit, one context at a time (4 params)",
                cc1,
                SERIES[2],
            ),
            ("Logistic regression B2 (4 params)", logistic, SERIES[3]),
        ],
        "Keeping the context fixes inference, and lands on the classical answer",
    )
    ax.set_ylabel("Predicted P(up | context)")
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    fig.text(
        0.01,
        -0.03,
        f"{args.asset} · 5-day moving average · seed {args.seed}",
        fontsize=8,
        color=INK_2,
    )
    fig.tight_layout()
    _save(fig, "fig_context_preserving_per_context", outputs)

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "per_context.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "context",
                "train_share",
                "data_p_up",
                "hea_superposed_readout",
                "hea_one_context",
                "context_preserving_one_context",
                "logistic_b2",
            ]
        )
        share = train_joint.sum(axis=1) / train_joint.sum()
        for context in range(len(truth)):
            writer.writerow(
                [
                    context,
                    share[context],
                    truth[context],
                    joint_readout[context],
                    inference[context],
                    cc1[context],
                    logistic[context],
                ]
            )
    if args.s8 is not None:
        _plot_qbgu_vs_per_sample(args, train_joint, truth, labels, outputs)
    print(f"per-context figures written to {args.out}")
    return 0


def _plot_qbgu_vs_per_sample(args, train_joint, truth, labels, outputs) -> None:
    """S8: the context-preserving circuit trained with QBGU vs per-sample SGD."""

    qbgu = _tables(args.s8, "e11_qstl_cc1_ma5", args.seed, args.asset)
    per_sample = _tables(args.s8, "e11_qstl_cc1_ma5__ps_b10000", args.seed, args.asset)
    for tables in (qbgu, per_sample):
        if not np.allclose(np.asarray(tables["train_joint"]), train_joint):
            raise ValueError("S1 and S8 training targets differ")
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    _bars(
        ax,
        labels,
        truth,
        [
            (
                "QBGU loss (superposed fidelity)",
                np.asarray(qbgu["QSTL_inference_conditional"])[:, UP],
                SERIES[0],
            ),
            (
                "Per-sample loss (one-hot fidelity), batch 10⁴",
                np.asarray(per_sample["QSTL_inference_conditional"])[:, UP],
                SERIES[1],
            ),
        ],
        "Same circuit, same shots: QBGU matches the data, per-sample goes to 0 or 1",
    )
    ax.set_ylabel("Predicted P(up | context)")
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    fig.text(
        0.01,
        -0.03,
        f"Context-preserving order-1 circuit · {args.asset} · 5-day moving average · "
        f"seed {args.seed} · 3000 × 10⁴ shots",
        fontsize=8,
        color=INK_2,
    )
    fig.tight_layout()
    _save(fig, "fig_qbgu_vs_per_sample_per_context", outputs)


def _fidelity(sweep_dir: Path, config: str, seed: int, asset: str) -> float:
    metrics = json.loads(
        (_run_dir(sweep_dir, config, seed) / "metrics.json").read_text(encoding="utf-8")
    )
    return metrics["assets"][asset]["models"]["QSTL"]["final_fidelity"]


if __name__ == "__main__":
    raise SystemExit(main())
