"""Phase 04 MerLin adaptation: variance ratio under realistic photonic hardware.

For each mode count ``M`` (input ``|1,0,1,0,...>``, ``n = M / 2``), Haar-random
interferometers from the shared loader are decomposed into the Clements mesh of
``lib/clements.py`` and evaluated in one batched MerLin ``QuantumLayer`` per
hardware condition (``lib/merlin_hardware.py``). From the per-interferometer
expectation ``<O>_U`` and second moment ``<O^2>_U`` the experiment estimates
``Var_samp``, ``Var_circ`` and their ratio with batch-means errors
(``lib/haar_mc.py``).

The ``ideal`` condition is the paper's model; its estimates are compared with
the exact Haar moments of ``lib/moments.py`` (z-scores, predeclared |z| < 4).
Every other condition is compared with ``ideal`` through the ratio factor
``ratio(condition) / ratio(ideal)`` at each M and the growth factor
``[ratio(M_max) / ratio(M_min)]_condition / [ratio(M_max) / ratio(M_min)]_ideal``.
With threshold detection the observable is evaluated on click patterns.

Artifact: ``merlin_variance_table.csv`` (one row per M, observable, condition).
"""

from __future__ import annotations

import csv
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from . import clements
from .data import build_dataloader
from .fock import input_occupation
from .haar_mc import moment_estimates
from .merlin_hardware import validate_condition
from .specs import build_observable
from .training import PhotonicExpectation

logger = logging.getLogger(__name__)

TABLE_NAME = "merlin_variance_table.csv"
COMPARED = ("mu1", "mu2", "sample_variance", "circuit_variance")


def describe_dataset(cfg: dict[str, Any]) -> dict[str, Any]:
    """Descriptor for ``run_status.json``."""
    return {
        "name": "haar_random_interferometers via Clements decomposition (synthetic, from seed)",
        "splits": {
            "monte_carlo": int(cfg["monte_carlo"]["unitaries"]),
            "modes": list(cfg["modes"]),
        },
        "preprocessing": "none",
        "subset": [condition["name"] for condition in cfg["conditions"]],
    }


def _mesh_parameters(unitaries: np.ndarray) -> tuple[clements.ClementsMesh, np.ndarray]:
    meshes = [clements.decompose(unitary) for unitary in unitaries]
    layout = meshes[0].block_modes
    if any(not np.array_equal(mesh.block_modes, layout) for mesh in meshes):
        raise RuntimeError("Clements layouts differ between draws; cannot batch them")
    return meshes[0], np.stack([mesh.parameters() for mesh in meshes])


def run(cfg: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    """Estimate variances per (M, observable, condition); see the module docstring.

    Parameters
    ----------
    cfg : dict
        Needs ``seed`` (Haar draws), ``observable_seed`` (random observables),
        ``modes``, ``input``, ``observables`` (``label`` and an observable spec), ``conditions``, ``monte_carlo`` (``unitaries``,
        ``batch_size``, ``batches``), ``z_threshold``, ``device``.
    run_dir : pathlib.Path
        Destination of ``merlin_variance_table.csv``.

    Returns
    -------
    dict
        Metrics record with per-row estimates, the ideal-versus-exact check and
        the hardware ratio factors.
    """
    for condition in cfg["conditions"]:
        validate_condition(condition)
    names = [condition["name"] for condition in cfg["conditions"]]
    if "ideal" not in names:
        raise KeyError("The conditions must include 'ideal' as the reference")
    batches = int(cfg["monte_carlo"]["batches"])
    z_threshold = float(cfg["z_threshold"])
    observable_rng = np.random.default_rng(int(cfg["observable_seed"]))
    logger.info(
        "DATASET_READY | ensemble=haar_unitaries_clements | modes=%s | unitaries=%d | conditions=%s",
        cfg["modes"],
        int(cfg["monte_carlo"]["unitaries"]),
        names,
    )
    rows = []
    for case_index, modes in enumerate(int(m) for m in cfg["modes"]):
        occupation = input_occupation(modes, cfg["input"])
        loader = build_dataloader(
            {**cfg, "modes": modes, "case_index": case_index}, split="merlin_variance"
        )
        unitaries = np.concatenate(list(loader))
        mesh, parameters = _mesh_parameters(unitaries)
        circuit = clements.perceval_circuit(mesh)
        observables = [
            (spec["label"], build_observable(spec["observable"], modes, observable_rng))
            for spec in cfg["observables"]
        ]
        for condition in cfg["conditions"]:
            logger.info(
                "EVALUATION_STARTED | M=%d | condition=%s", modes, condition["name"]
            )
            started = time.perf_counter()
            for label, observable in observables:
                expectation = PhotonicExpectation(
                    circuit,
                    occupation,
                    observable.evaluate,
                    device=str(cfg["device"]),
                    hardware=condition,
                )
                probabilities = expectation.probabilities(parameters)
                means = probabilities @ expectation.values
                second = probabilities @ expectation.values**2
                estimates = moment_estimates(means, second, batches)
                row = {
                    "modes": modes,
                    "photons": sum(occupation),
                    "observable": label,
                    "condition": condition["name"],
                    "detector": condition["detector"],
                    "postselect": condition["postselect"],
                    "noise": dict(condition["noise"]),
                    "outcomes": len(expectation.patterns),
                    "unitaries": len(parameters),
                    **{f"{name}": estimates[name]["estimate"] for name in COMPARED},
                    **{
                        f"{name}_se": estimates[name]["standard_error"]
                        for name in COMPARED
                    },
                    "variance_ratio": estimates["sample_variance"]["estimate"]
                    / estimates["circuit_variance"]["estimate"],
                }
                if condition["name"] == "ideal":
                    exact = observable.exact_moments(occupation)
                    for name in COMPARED:
                        row[f"{name}_exact"] = float(exact[name])
                        row[f"{name}_z"] = (row[name] - float(exact[name])) / row[
                            f"{name}_se"
                        ]
                rows.append(row)
            logger.info(
                "EVALUATION_COMPLETED | M=%d | condition=%s | seconds=%.1f",
                modes,
                condition["name"],
                time.perf_counter() - started,
            )
    ideal = {
        (r["modes"], r["observable"]): r for r in rows if r["condition"] == "ideal"
    }
    for row in rows:
        reference = ideal[(row["modes"], row["observable"])]
        row["ratio_factor_vs_ideal"] = (
            row["variance_ratio"] / reference["variance_ratio"]
        )
    mode_values = sorted({row["modes"] for row in rows})
    low, high = mode_values[0], mode_values[-1]
    comparisons = []
    for condition in names:
        for label in [spec["label"] for spec in cfg["observables"]]:
            pick = {
                r["modes"]: r
                for r in rows
                if r["condition"] == condition and r["observable"] == label
            }
            ideal_growth = (
                ideal[(high, label)]["variance_ratio"]
                / ideal[(low, label)]["variance_ratio"]
            )
            growth = pick[high]["variance_ratio"] / pick[low]["variance_ratio"]
            factors = [pick[m]["ratio_factor_vs_ideal"] for m in mode_values]
            comparisons.append(
                {
                    "condition": condition,
                    "observable": label,
                    "ratio_factors_by_modes": dict(zip(mode_values, factors)),
                    "growth": growth,
                    "growth_factor_vs_ideal": growth / ideal_growth,
                    "preserves_conclusion": all(1 / 3 <= f <= 3 for f in factors)
                    and 0.5 <= growth / ideal_growth <= 2,
                }
            )
    ideal_rows = [r for r in rows if r["condition"] == "ideal"]
    max_z = max(abs(r[f"{name}_z"]) for r in ideal_rows for name in COMPARED)
    with (run_dir / TABLE_NAME).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(dict.fromkeys(k for r in rows for k in r))
        )
        writer.writeheader()
        writer.writerows(rows)
    logger.info("ARTIFACT_WRITTEN | path=%s | rows=%d", TABLE_NAME, len(rows))
    logger.info(
        "EVALUATION_COMPLETED | summary | ideal_max_abs_z=%.3f | ideal_valid=%s",
        max_z,
        max_z < z_threshold,
    )
    return {
        "experiment": "merlin_variance",
        "ideal_matches_exact": max_z < z_threshold,
        "ideal_max_abs_z": max_z,
        "comparisons": comparisons,
        "rows": rows,
        "artifacts": {"table": TABLE_NAME},
    }
