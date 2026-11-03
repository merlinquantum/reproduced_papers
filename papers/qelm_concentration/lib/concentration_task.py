"""Concentration check (E1b): output variability versus qubit count and depth.

``Var_{u ~ Unif(-1,1)}[<R_i>]`` (paper Eq. for variability) is computed for weight-1 and
nearest-neighbour weight-2 Paulis at each ``(N, layers)`` and for each named parameter
setting (Table I versus a strongly scrambling kicked-Ising point given in the config).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .qelm import (
    architecture_from_config,
    build_reservoir,
    real_dtype_from_config,
    ring_features,
)
from .reservoir import ReservoirHyperparameters

logger = logging.getLogger(__name__)


def run_concentration(cfg: dict, run_dir: Path) -> dict:
    """Variability grid over qubit counts, layer counts and parameter settings."""
    exp = cfg["concentration"]
    real_dtype = real_dtype_from_config(cfg)
    chunk = int(cfg["reservoir"]["chunk_size"])
    records = []
    logger.info(
        "DATASET_READY | dataset=uniform_inputs | n_samples=%s | distribution=Unif(-1,1)^(N/2)",
        exp["n_samples"],
    )
    for setting_name, setting in exp["parameter_settings"].items():
        hyper = ReservoirHyperparameters.from_config(setting)
        for n_qubits in exp["n_qubits"]:
            for n_layers in exp["n_layers"]:
                arch = architecture_from_config(
                    cfg["reservoir"], n_qubits=int(n_qubits), n_layers=int(n_layers)
                )
                for realisation in exp["reservoir_seeds"]:
                    reservoir = build_reservoir(
                        arch, hyper, int(realisation), cfg["device"], real_dtype
                    )
                    inputs = np.random.default_rng(
                        [int(realisation), int(n_qubits), 5]
                    ).uniform(-1.0, 1.0, size=(int(exp["n_samples"]), arch.n_blocks))
                    features, weights = ring_features(
                        reservoir, inputs, "nearest_neighbour", chunk
                    )
                    variance = features.var(axis=0)
                    record = {
                        "setting": setting_name,
                        "n_qubits": int(n_qubits),
                        "n_layers": int(n_layers),
                        "reservoir_seed": int(realisation),
                    }
                    for weight in (1, 2):
                        selected = variance[weights == weight]
                        record[f"w{weight}_variability_mean"] = float(selected.mean())
                        record[f"w{weight}_variability_median"] = float(
                            np.median(selected)
                        )
                    records.append(record)
                    logger.info(
                        "EVALUATION_COMPLETED | setting=%s | n_qubits=%d | layers=%d | seed=%s | w1_var=%.3e | w2_var=%.3e",
                        setting_name,
                        n_qubits,
                        n_layers,
                        realisation,
                        record["w1_variability_mean"],
                        record["w2_variability_mean"],
                    )
    return {"records": records, "summary": _summarise(records)}


def _summarise(records: list[dict]) -> dict:
    groups: dict[str, list[dict]] = {}
    for record in records:
        key = f"{record['setting']}|n_qubits={record['n_qubits']}|n_layers={record['n_layers']}"
        groups.setdefault(key, []).append(record)
    summary = {}
    for key, items in groups.items():
        summary[key] = {"n": len(items)}
        for metric in ("w1_variability_mean", "w2_variability_mean"):
            values = np.array([item[metric] for item in items])
            summary[key][f"{metric}_avg"] = float(values.mean())
            summary[key][f"{metric}_std"] = float(values.std())
    return summary
