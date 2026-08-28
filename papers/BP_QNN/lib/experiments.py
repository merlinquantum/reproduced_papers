"""Gate-based and photonic gradient-variance experiments.
From https://github.com/easonoob/Eason-2026-preasymptotic-trainability-pvqc/tree/main
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import numpy as np


def _int_list(value: object, name: str) -> list[int]:
    if isinstance(value, str):
        values = [int(item.strip()) for item in value.split(",") if item.strip()]
    elif isinstance(value, list):
        values = [int(item) for item in value]
    else:
        raise TypeError(f"{name} must be a list or comma-separated string")
    if not values or any(item <= 0 for item in values):
        raise ValueError(f"{name} must contain positive integers")
    return values


def _write_rows(run_dir: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("Experiment produced no rows")
    with (run_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _fit_exponential(
    qubits: np.ndarray, variance: np.ndarray
) -> tuple[float, float, float]:
    slope, intercept = np.polyfit(qubits, np.log10(variance), 1)
    fitted = intercept + slope * qubits
    residual = np.sum((np.log10(variance) - fitted) ** 2)
    total = np.sum((np.log10(variance) - np.mean(np.log10(variance))) ** 2)
    return float(slope), float(intercept), float(1.0 - residual / total)


def _resolve_fig3_layers(number_of_qubits: int, cfg: dict) -> int:
    """Resolve the linear-in-system-size depth used for gate-based Fig. 3."""
    layers_per_qubit = cfg.get("layers_per_qubit")
    if layers_per_qubit is None:
        raise ValueError(
            "fig3_gb requires 'layers_per_qubit'; fixed shallow depth does not reproduce the paper's qubit-scaling experiment"
        )
    layers_per_qubit = int(layers_per_qubit)
    if layers_per_qubit <= 0:
        raise ValueError("layers_per_qubit must be positive")
    return layers_per_qubit * number_of_qubits


def _run_gate_based(cfg: dict, run_dir: Path) -> None:
    from .gate_based import sample_gradient_variance

    experiment = str(cfg["experiment"])
    print(f"[Gate-based] Starting {experiment} experiment")
    qubits = _int_list(cfg["qubits"], "qubits")
    layers = _int_list(cfg["layers"], "layers")
    if experiment == "fig3_gb":
        layer_values = {
            number_of_qubits: _resolve_fig3_layers(number_of_qubits, cfg)
            for number_of_qubits in qubits
        }
    else:
        layer_values = {
            number_of_qubits: number_of_layers
            for number_of_qubits in qubits
            for number_of_layers in layers
        }
    rows: list[dict[str, object]] = []
    for number_of_qubits in qubits:
        current_layers = (
            [layer_values[number_of_qubits]] if experiment == "fig3_gb" else layers
        )
        for number_of_layers in current_layers:
            print(
                f"[Gate-based] Running circuit: qubits={number_of_qubits}, "
                f"layers={number_of_layers}, samples={int(cfg['samples'])}"
            )
            variance = sample_gradient_variance(
                number_of_qubits, number_of_layers, int(cfg["samples"]), cfg
            )
            rows.append(
                {
                    "qubits": number_of_qubits,
                    "layers": number_of_layers,
                    "gradient_variance": variance,
                }
            )
    _write_rows(run_dir, rows)
    if experiment == "fig3_gb":
        x = np.array(qubits, dtype=float)
        y = np.array([float(row["gradient_variance"]) for row in rows], dtype=float)
        slope, intercept, r_squared = _fit_exponential(x, y)
        (run_dir / "fit.json").write_text(
            json.dumps(
                {
                    "model": "log10(variance) = slope * qubits + intercept",
                    "slope": slope,
                    "intercept": intercept,
                    "r_squared": r_squared,
                },
                indent=2,
            )
        )
        if cfg.get("plot", False):
            _plot_fig3(run_dir, x, y, slope, intercept, "Gate-based Fig. 3")
    elif cfg.get("plot", False):
        _plot_fig4(run_dir, rows)


def _run_merlin(cfg: dict, run_dir: Path) -> None:
    from .photonic import sample_photonic_variance

    print("[Photonic/MerLin] Starting fig3_merlin experiment")
    rows: list[dict[str, object]] = []
    for initialization in cfg["theta_initializations"]:
        for computation_space in cfg["computation_spaces"]:
            for number_of_qubits in _int_list(cfg["qubits"], "qubits"):
                print(
                    f"[Photonic/MerLin] Running circuit: qubits={number_of_qubits}, "
                    f"computation_space={computation_space}, "
                    f"theta_initialization={initialization}, samples={int(cfg['samples'])}"
                )
                variance = sample_photonic_variance(
                    number_of_qubits,
                    computation_space,
                    initialization,
                    int(cfg["samples"]),
                    cfg,
                )
                rows.append(
                    {
                        "qubits": number_of_qubits,
                        "layers": 1,
                        "computation_space": computation_space,
                        "theta_initialization": initialization,
                        "gradient_variance": variance,
                    }
                )
    _write_rows(run_dir, rows)
    if cfg.get("plot", False):
        _plot_photonic(run_dir, rows)


def _plot_fig3(
    run_dir: Path,
    x: np.ndarray,
    y: np.ndarray,
    slope: float,
    intercept: float,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    fit_x = np.linspace(x.min(), x.max(), 200)
    plt.semilogy(x, y, "o", label="sample variance")
    plt.semilogy(
        fit_x, 10 ** (slope * fit_x + intercept), "--", label=f"slope = {slope:.3f}"
    )
    plt.xlabel("Number of qubits")
    plt.ylabel(r"$\mathrm{Var}[\partial_{\theta_{1,1}} E]$")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "figure3.png", dpi=300)
    plt.close()


def _plot_fig4(run_dir: Path, rows: list[dict[str, object]]) -> None:
    import matplotlib.pyplot as plt

    for qubit in sorted({int(row["qubits"]) for row in rows}):
        selected = [row for row in rows if int(row["qubits"]) == qubit]
        plt.semilogy(
            [row["layers"] for row in selected],
            [row["gradient_variance"] for row in selected],
            label=f"{qubit} qubits",
        )
    plt.xlabel("Number of layers, L")
    plt.ylabel(r"$\mathrm{Var}[\partial_{\theta_{1,1}} E]$")
    plt.title("Gate-based Fig. 4")
    plt.legend(ncol=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(run_dir / "figure4.png", dpi=300)
    plt.close()


def _plot_photonic(run_dir: Path, rows: list[dict[str, object]]) -> None:
    import matplotlib.pyplot as plt

    colors = {
        "fock": "#435BEC",
        "unbunched": "#FF6F61",
        "dual_rail": "#6A4C93",
    }
    labels = {
        "fock": "Fock",
        "unbunched": "Unbunched",
        "dual_rail": "Dual rail",
    }
    initializations = list(
        dict.fromkeys(str(row["theta_initialization"]) for row in rows)
    )
    computation_spaces = list(
        dict.fromkeys(str(row["computation_space"]) for row in rows)
    )
    fit_results: dict[str, dict[str, dict[str, float]]] = {}

    figure, axes = plt.subplots(
        1,
        len(initializations),
        figsize=(7 * len(initializations), 5.5),
        sharey=True,
        squeeze=False,
    )
    axes = axes[0]

    for axis, initialization in zip(axes, initializations):
        fit_results[initialization] = {}
        for computation_space in computation_spaces:
            selected = [
                row
                for row in rows
                if row["theta_initialization"] == initialization
                and row["computation_space"] == computation_space
            ]
            selected.sort(key=lambda row: int(row["qubits"]))
            qubits = np.asarray([int(row["qubits"]) for row in selected], dtype=float)
            variance = np.asarray(
                [float(row["gradient_variance"]) for row in selected], dtype=float
            )
            if qubits.size < 2:
                raise ValueError(
                    f"Need at least two qubit values to fit {initialization}/{computation_space}"
                )
            if not np.all(np.isfinite(variance)) or np.any(variance <= 0):
                raise ValueError(
                    f"Gradient variance must be finite and positive to fit {initialization}/{computation_space}"
                )

            slope, intercept = np.polyfit(qubits, np.log10(variance), 1)
            fitted_log10_variance = intercept + slope * qubits
            residual = np.sum((np.log10(variance) - fitted_log10_variance) ** 2)
            total = np.sum((np.log10(variance) - np.mean(np.log10(variance))) ** 2)
            if total <= 0:
                raise ValueError(
                    f"Gradient variance has no variation to fit for {initialization}/{computation_space}"
                )
            r_squared = 1.0 - residual / total
            fit_results[initialization][computation_space] = {
                "slope": float(slope),
                "intercept": float(intercept),
                "r_squared": float(r_squared),
            }

            color = colors[computation_space]
            axis.semilogy(
                qubits,
                variance,
                "o",
                color=color,
                label=f"{labels[computation_space]} data",
                zorder=3,
            )
            fit_qubits = np.linspace(qubits.min(), qubits.max(), 250)
            axis.semilogy(
                fit_qubits,
                10 ** (slope * fit_qubits + intercept),
                "--",
                color=color,
                linewidth=2.2,
                zorder=2,
                label=f"{labels[computation_space]} fit, slope={slope:.3f}",
            )

        axis.set_xlabel("Number of qubits")
        axis.set_title(f"{initialization} initialization")
        axis.grid(True, which="both", alpha=0.25)
        axis.legend(fontsize=8)

    axes[0].set_ylabel("Mean gradient variance")
    figure.suptitle("Photonic Fig. 3 analogue")
    figure.tight_layout()
    figure.savefig(run_dir / "figure3_merlin.png", dpi=300, bbox_inches="tight")
    plt.close(figure)
    (run_dir / "fit_merlin.json").write_text(
        json.dumps(fit_results, indent=2), encoding="utf-8"
    )


def run_experiment(cfg: dict, run_dir: Path) -> None:
    """Dispatch one configured experiment.

    Parameters
    ----------
    cfg : dict
        Resolved experiment configuration.
    run_dir : pathlib.Path
        Output directory for CSV, fit, and figure artifacts.

    Returns
    -------
    None
        The selected experiment writes its artifacts to ``run_dir``.
    """
    random.seed(int(cfg["seed"]))
    np.random.seed(int(cfg["seed"]))
    backend = str(cfg["backend"])
    if backend == "gate_based":
        _run_gate_based(cfg, run_dir)
    elif backend == "merlin":
        _run_merlin(cfg, run_dir)
    else:
        raise ValueError(f"Unsupported backend: {backend}")
