"""Read completed runs of one or more sweeps from their structured artifacts.

Every helper starts from ``sweep_status.json`` and the per-run
``run_status.json`` / ``metrics.json`` / ``history.csv`` written by
``lib/runner.py``; nothing is parsed from logs.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple


@dataclass
class RunRecord:
    """One COMPLETED run of a sweep.

    Attributes
    ----------
    sweep_id : str
        Identifier of the sweep the run belongs to.
    run_dir : pathlib.Path
        Absolute run directory.
    candidate : dict
        ``run_status.json["candidate"]`` (model, protocol, optimiser, l2, ...).
    seed : int
        Initialisation seed.
    metrics : dict
        Parsed ``metrics.json``.
    """

    sweep_id: str
    run_dir: Path
    candidate: dict
    seed: int
    metrics: dict

    def history(self) -> dict[str, list[float]]:
        """Per-step columns of ``history.csv`` as float lists (empty cells skipped)."""
        columns: dict[str, list[float]] = {}
        with (self.run_dir / self.metrics["artifacts"]["history"]).open(
            encoding="utf-8"
        ) as handle:
            for row in csv.DictReader(handle):
                for key, value in row.items():
                    if value != "":
                        columns.setdefault(key, []).append(float(value))
        return columns


def load_sweep(sweep_dir: Path) -> list[RunRecord]:
    """Return every COMPLETED run of a sweep.

    Raises
    ------
    ValueError
        If the sweep is still running or a completed run's status disagrees.
    """
    sweep_dir = Path(sweep_dir).resolve()
    status = json.loads((sweep_dir / "sweep_status.json").read_text(encoding="utf-8"))
    if status["status"] not in ("COMPLETED", "PARTIAL"):
        raise ValueError(
            f"{sweep_dir.name}: sweep status is {status['status']}, not finished"
        )
    records = []
    for run in status["runs"]:
        if run["status"] != "COMPLETED":
            continue
        run_dir = sweep_dir / run["run_dir"]
        run_status = json.loads(
            (run_dir / "run_status.json").read_text(encoding="utf-8")
        )
        if run_status["status"] != "COMPLETED":
            raise ValueError(
                f"{run['run_id']}: run_status {run_status['status']} disagrees with the sweep"
            )
        metrics = json.loads(
            (run_dir / run_status["metrics_path"]).read_text(encoding="utf-8")
        )
        candidate = run_status["candidate"]
        model = candidate.get("model", {})
        if model.get("kind") == "photonic_merlin" and "computation_space" not in model:
            # Runs made before the PNR control (sweep fig8_photonic_merlin) had
            # no computation_space option; UNBUNCHED was the only behaviour.
            candidate = {
                **candidate,
                "model": {**model, "computation_space": "unbunched"},
            }
        records.append(
            RunRecord(
                status["sweep_id"],
                run_dir,
                candidate,
                int(run_status["seed"]),
                metrics,
            )
        )
    return records


class CandidateKey(NamedTuple):
    """Identity of a model configuration, excluding the initialisation seed."""

    kind: str
    layers: int | None
    cnot_pattern: str | None
    zz_repetitions: int | None
    hidden: int | None
    spectrum: int | None
    readout: str | None
    n_photons: int | None
    computation_space: str | None
    protocol: str
    l2: float
    steps: int
    train_source: str
    train_seed: int | None


def candidate_key(candidate: dict) -> CandidateKey:
    """Hashable identity of a model configuration, excluding the seed."""
    model = candidate["model"]
    return CandidateKey(
        model["kind"],
        model.get("layers"),
        model.get("cnot_pattern"),
        model.get("zz_repetitions"),
        model.get("hidden"),
        model.get("spectrum"),
        model.get("readout"),
        model.get("n_photons"),
        model.get("computation_space"),
        candidate["protocol"],
        candidate["l2"],
        candidate["steps"],
        candidate["train_source"],
        candidate["train_seed"],
    )


def describe(candidate: dict) -> str:
    """Short human-readable model label."""
    model = candidate["model"]
    if model["kind"] == "biased_quantum":
        return f"quantum (biased) L={model['layers']}"
    if model["kind"] == "generic_quantum":
        pattern = "Fig. 10 CNOTs" if model["cnot_pattern"] == "paper" else "code CNOTs"
        return f"quantum (no bias, {pattern}) L={model['layers']}"
    if model["kind"] == "bias_fourier":
        return f"B2 bias-encoded Fourier L={model['spectrum']}"
    if model["kind"] == "mlp_biased":
        return f"B3 bias-encoded MLP h={model['hidden']}"
    if model["kind"] == "mlp_unbiased":
        return f"B4 unbiased MLP h={model['hidden']}"
    if model["kind"] == "quadratic":
        return "B5 quadratic (ceiling)"
    if model["kind"] == "photonic_merlin":
        detection = "PNR" if model.get("computation_space") == "fock" else "threshold"
        return f"photonic MerLin ({model['readout']} readout, {detection}) N={model['n_photons']}"
    return f"surrogate L={model['spectrum']}"


def matches(candidate: dict, selector: dict) -> bool:
    """True if every ``selector`` key equals the candidate's model or run field."""
    flat = {
        **candidate["model"],
        **{k: v for k, v in candidate.items() if k != "model"},
    }
    return all(flat.get(key) == value for key, value in selector.items())
