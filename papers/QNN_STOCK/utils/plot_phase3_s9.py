"""S9 figures: exact vs trained context loader for two circuits.

- ``fig_s9_loader.png``: final QBGU fidelity (top) and inference test
  log-loss (bottom) for the paper HEA and the context-preserving order-1
  circuit, each with exact loading, the trained HEA loader without the
  leading Hadamard layer, and the Fig. 4 loader (Hadamard first). Bars are
  seed means, dots single seeds; the dashed line is logistic regression B2.
- ``fig_s9_bound.png``: for the context-preserving circuit with a trained
  loader, the final fidelity against the best fidelity any real per-context
  prediction rotation can reach from that loaded state,
  ``max_phi (sum_c sqrt(P(c)) |Re(exp(-i phi) psi_c)|)^2``.
- ``s9_loader_states.csv``: per run and asset, the loaded state's TVD, its
  fidelity to the exact state, the fidelity with phases removed, the bound,
  and the final QSTL fidelity.

Usage::

    python papers/QNN_STOCK/utils/plot_phase3_s9.py <S9 sweep dir> \
        --out results/phase3_s9
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

from lib.circuits import apply_ansatz  # noqa: E402
from lib.statevector import apply_1q, basis_state  # noqa: E402
from lib.training import HADAMARD  # noqa: E402
from plot_phase3_s1 import INK, INK_2, SERIES, _save, _style, plt  # noqa: E402

CIRCUITS = {
    "paper HEA": {"raw": "e3_qstl_original", "ma5": "e3_qstl_ma5"},
    "context-preserving\norder 1": {"raw": "e11_qstl_cc1", "ma5": "e11_qstl_cc1_ma5"},
}
LOADERS = {
    "": "exact amplitudes",
    "__hea_loader": "trained loader (MSE + SPSA)",
    "__hea_loader_fig4": "trained loader, Fig. 4 (H first)",
}
BASELINES = {"raw": "e3_baselines_classical", "ma5": "e3_baselines_ma5"}
PHASE_GRID = np.linspace(0.0, np.pi, 721)


def _runs(sweep_dir: Path) -> dict[tuple[str, int], Path]:
    status = json.loads((sweep_dir / "sweep_status.json").read_text(encoding="utf-8"))
    return {
        (run["candidate"]["config"], run["seed"]): (
            sweep_dir / run["metrics_path"]
        ).parent
        for run in status["runs"]
    }


def _metrics(run_dir: Path) -> dict:
    return json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))


def loaded_state(run_dir: Path, asset: str) -> np.ndarray:
    """Rebuild the trained loader's state from ``params.npz`` and the config."""

    cfg = json.loads((run_dir / "config_snapshot.json").read_text(encoding="utf-8"))
    params = np.load(run_dir / "params.npz")[f"loader_{asset}"]
    n_qubits = int(cfg["dataset"]["context_length"])
    state = basis_state(0, n_qubits)
    if cfg["model"]["context_loader"]["initial_hadamard"]:
        for qubit in range(n_qubits):
            state = apply_1q(state, HADAMARD, qubit, n_qubits)
    return apply_ansatz(state, params, n_qubits)


def context_preserving_bound(context_prob: np.ndarray, state: np.ndarray) -> float:
    """Best QBGU fidelity of real per-context prediction rotations on ``state``.

    The output is ``sum_c psi_c |c>|phi_c>`` with real ``|phi_c>``, so the
    overlap with the target is ``sum_c sqrt(P(c)) psi_c <t_c|phi_c>`` and
    ``<t_c|phi_c>`` ranges over ``[-1, 1]``.
    """

    projections = np.abs(np.real(np.exp(-1j * PHASE_GRID)[:, None] * state[None, :]))
    return float(np.max(projections @ np.sqrt(context_prob)) ** 2)


def loader_rows(sweep_dir: Path) -> list[dict]:
    rows = []
    for (config, seed), run_dir in sorted(_runs(sweep_dir).items()):
        if "__hea_loader" not in config:
            continue
        metrics = _metrics(run_dir)
        tables = json.loads(
            (run_dir / "distributions.json").read_text(encoding="utf-8")
        )
        for asset, block in metrics["assets"].items():
            context_prob = np.asarray(tables[asset]["train_joint"]).sum(axis=1)
            state = loaded_state(run_dir, asset)
            rows.append(
                {
                    "candidate": config,
                    "seed": seed,
                    "asset": asset,
                    "tvd": block["loader"]["tvd"],
                    "fidelity_to_exact_state": block["loader"][
                        "fidelity_to_exact_state"
                    ],
                    "fidelity_magnitudes_only": block["loader"][
                        "fidelity_magnitudes_only"
                    ],
                    "context_preserving_bound": context_preserving_bound(
                        context_prob, state
                    ),
                    "qstl_final_fidelity": block["models"]["QSTL"]["final_fidelity"],
                }
            )
    return rows


def _value(metrics: dict, asset: str, model: str, path: str) -> float:
    value = metrics["assets"][asset]["models"][model]
    for key in path.split("/"):
        value = value[key]
    return float(value)


def plot_loader(sweep_dir: Path, data: str, outputs: list[Path]) -> None:
    runs = _runs(sweep_dir)
    seeds = sorted({seed for _, seed in runs})
    assets = list(_metrics(next(iter(runs.values())))["assets"])
    panels = [
        ("final_fidelity", "Final QBGU fidelity"),
        ("inference/test/log_loss", "Test log-loss (one context at a time)"),
    ]
    fig, axes = plt.subplots(
        2, len(assets), figsize=(6.2 * len(assets), 7.2), sharex=True
    )
    width = 0.8 / len(LOADERS)
    for column, asset in enumerate(assets):
        for row, (path, label) in enumerate(panels):
            ax = axes[row, column]
            for group, configs in enumerate(CIRCUITS.values()):
                for index, (suffix, name) in enumerate(LOADERS.items()):
                    values = np.array(
                        [
                            _value(
                                _metrics(runs[(configs[data] + suffix, seed)]),
                                asset,
                                "QSTL",
                                path,
                            )
                            for seed in seeds
                        ]
                    )
                    x = group + (index - (len(LOADERS) - 1) / 2) * width
                    ax.bar(
                        x,
                        values.mean(),
                        width * 0.9,
                        color=SERIES[index],
                        label=name if group == 0 else None,
                    )
                    ax.scatter(
                        np.full(len(values), x), values, s=12, color=INK, zorder=3
                    )
            if path.startswith("inference"):
                baseline = _value(
                    _metrics(runs[(BASELINES[data], seeds[0])]),
                    asset,
                    "B2_logistic",
                    path,
                )
                ax.axhline(
                    baseline,
                    color=INK_2,
                    linestyle="--",
                    linewidth=1.0,
                    label="logistic regression B2",
                )
            ax.set_ylabel(label)
            if row == 0:
                ax.set_title(
                    f"{asset} · {'5-day moving average' if data == 'ma5' else 'raw daily moves'}",
                    fontsize=11,
                )
            ax.set_xticks(range(len(CIRCUITS)), list(CIRCUITS))
        axes[1, 0].legend(loc="upper left", fontsize=8, frameon=False)
    axes[0, 0].legend(loc="lower left", fontsize=8, frameon=False)
    fig.text(
        0.01,
        -0.02,
        f"S9 · seeds {seeds[0]}–{seeds[-1]} · bars: mean, dots: seeds",
        fontsize=8,
        color=INK_2,
    )
    fig.tight_layout()
    _save(fig, f"fig_s9_loader_{data}", outputs)


def plot_bound(rows: list[dict], outputs: list[Path]) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    markers = {"": "o", "_ma5": "s"}
    for index, suffix in enumerate(("__hea_loader", "__hea_loader_fig4")):
        for data_suffix, marker in markers.items():
            selected = [
                r
                for r in rows
                if r["candidate"] == f"e11_qstl_cc1{data_suffix}{suffix}"
            ]
            ax.scatter(
                [r["context_preserving_bound"] for r in selected],
                [r["qstl_final_fidelity"] for r in selected],
                color=SERIES[index + 1],
                marker=marker,
                s=28,
                label=f"{LOADERS[suffix]}, {'MA5' if data_suffix else 'raw'}",
            )
    ax.plot([0, 1], [0, 1], color=INK_2, linestyle="--", linewidth=1.0)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Best reachable fidelity from the loaded state")
    ax.set_ylabel("Final QBGU fidelity, context-preserving circuit")
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    fig.tight_layout()
    _save(fig, "fig_s9_bound", outputs)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sweep_dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    outputs = [args.out]
    _style()
    rows = loader_rows(args.sweep_dir)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "s9_loader_states.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for data in ("raw", "ma5"):
        plot_loader(args.sweep_dir, data, outputs)
    plot_bound(rows, outputs)
    print(f"S9 figures written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
