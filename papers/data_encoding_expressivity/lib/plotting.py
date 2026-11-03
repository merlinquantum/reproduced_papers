"""Figures written by the runner (fitting curves, coefficient clouds, universality)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_KIND_COLORS = {
    "quantum": "tab:blue",
    "fourier_classical": "tab:green",
    "photonic": "tab:purple",
}


def plot_fit_cases(
    inputs: np.ndarray,
    cases: Sequence[dict],
    runs: Sequence[dict],
    predictions: dict[str, np.ndarray],
    path: Path,
) -> None:
    """Draw one row per fitting case, in the layout of paper Figs. 3-4.

    Left: target, grid samples, the B1 oracle fit and every seed's final
    prediction. Right: full-grid MSE per optimizer step for every seed, with
    the B1 floor as a dashed line.

    Parameters
    ----------
    inputs : np.ndarray
        Grid of shape ``(N,)``.
    cases : Sequence[dict]
        Per-case records with ``name``, ``targets``, ``b1_predictions`` and
        ``b1_mse``.
    runs : Sequence[dict]
        Per-run records with ``case``, ``model_kind``, ``seed`` and
        ``mse_history``.
    predictions : dict
        Final predictions keyed ``"<case>/<kind>/<seed>"``.
    path : Path
        Output PNG path.
    """
    figure, axes = plt.subplots(
        len(cases), 2, figsize=(11, 3.2 * len(cases)), squeeze=False
    )
    for row, case in enumerate(cases):
        fit_axis, loss_axis = axes[row]
        fit_axis.plot(inputs, case["targets"], color="black", lw=1.5, label="target")
        fit_axis.scatter(
            inputs,
            case["targets"],
            s=12,
            facecolor="white",
            edgecolor="black",
            zorder=3,
        )
        fit_axis.plot(
            inputs,
            case["b1_predictions"],
            color="tab:red",
            ls="--",
            lw=1.2,
            label="B1 oracle",
        )
        labelled_kinds = set()
        for run in runs:
            if run["case"] != case["name"]:
                continue
            kind = run["model_kind"]
            color = _KIND_COLORS[kind]
            key = f"{case['name']}/{kind}/{run['seed']}"
            label = kind if kind not in labelled_kinds else None
            labelled_kinds.add(kind)
            fit_axis.plot(
                inputs, predictions[key], color=color, lw=0.8, alpha=0.6, label=label
            )
            loss_axis.semilogy(run["mse_history"], color=color, lw=0.8, alpha=0.6)
        loss_axis.axhline(max(case["b1_mse"], 1e-32), color="tab:red", ls="--", lw=1.2)
        fit_axis.set_title(case["name"], fontsize=10)
        fit_axis.set_xlabel("x")
        fit_axis.set_ylim(-1, 1)
        fit_axis.legend(fontsize=7, loc="upper right")
        loss_axis.set_xlabel("optimizer step")
        loss_axis.set_ylabel("full-grid MSE")
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)


def plot_coefficient_clouds(samples: dict[str, np.ndarray], path: Path) -> None:
    """Scatter ``Re c_n`` against ``Im c_n`` per setting, in the layout of paper Fig. 5.

    Parameters
    ----------
    samples : dict
        Setting label to complex array of shape ``(n_samples, K + 1)``.
    path : Path
        Output PNG path.
    """
    labels = list(samples)
    n_coefficients = samples[labels[0]].shape[1]
    figure, axes = plt.subplots(
        len(labels),
        n_coefficients,
        figsize=(2.2 * n_coefficients, 2.3 * len(labels)),
        squeeze=False,
    )
    for row, label in enumerate(labels):
        for index in range(n_coefficients):
            axis = axes[row, index]
            values = samples[label][:, index]
            axis.scatter(
                values.real, values.imag, s=10, facecolor="white", edgecolor="tab:red"
            )
            # Same axis range as the reference notebook's Fig. 5 cell.
            axis.set_xlim(-1, 1)
            axis.set_ylim(-1, 1)
            axis.set_aspect("equal")
            axis.tick_params(labelsize=6)
            if row == 0:
                axis.set_title(f"$c_{index}$", fontsize=9)
            if index == 0:
                axis.set_ylabel(label, fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)


def plot_universality(rows: Sequence[dict], path: Path) -> None:
    """Plot the square-wave RMS error of the constructed model against ``m``.

    Parameters
    ----------
    rows : Sequence[dict]
        Records with ``n_qubits``, ``rms_to_target`` and ``parseval_tail_rms``.
    path : Path
        Output PNG path.
    """
    qubits = [row["n_qubits"] for row in rows]
    figure, axis = plt.subplots(figsize=(5, 3.5))
    axis.loglog(
        qubits,
        [row["rms_to_target"] for row in rows],
        "o-",
        label="constructed model $f_m$",
    )
    axis.loglog(
        qubits,
        [row["parseval_tail_rms"] for row in rows],
        "k--",
        label="Parseval truncation error",
    )
    axis.set_xlabel("subsystems m (degree K = m)")
    axis.set_ylabel("RMS error to sign(sin x)")
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)
