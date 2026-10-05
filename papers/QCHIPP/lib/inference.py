"""Checkpoint inference combining the binding and immunogenicity heads."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import torch

from .baselines import BASELINE_MODELS, predict_baseline
from .data import ALPHABET, encode_peptide
from .models import build_trainable_model
from .training import predict_probabilities

PROJECT_DIR = Path(__file__).resolve().parents[1]


class _Head:
    """One trained task head restored from a completed run directory."""

    def __init__(self, run_dir, expected_task: str):
        path = Path(run_dir).resolve()
        # model.pkl is a pickle; restrict loading to this paper's own outputs.
        if not path.is_relative_to(PROJECT_DIR / "outdir"):
            raise ValueError("Load only trusted local QCHIPP outdir checkpoints")
        self.cfg = json.loads((path / "config_snapshot.json").read_text())
        status = json.loads((path / "run_status.json").read_text())
        if (
            status["status"] != "COMPLETED"
            or self.cfg["dataset"]["task"] != expected_task
        ):
            raise ValueError(
                "Expected a completed checkpoint for the correct biological head"
            )
        self.category_codes = status["dataset"]["category_codes"]
        self.source_sha256 = status["dataset"]["source_sha256"]
        self.name = self.cfg["model"]["name"]
        if self.name in BASELINE_MODELS:
            with (path / "model.pkl").open("rb") as handle:
                self.model = pickle.load(handle)  # noqa: S301 - trusted outdir only
            return
        self.model = build_trainable_model(self.cfg)
        self.model.load_state_dict(
            torch.load(path / "model.pt", map_location="cpu", weights_only=True)
        )
        self.model.eval()

    def predict(self, peptides: list[str], shots: int = 0) -> np.ndarray:
        rows = []
        for peptide in peptides:
            if len(peptide) != 9 or set(peptide) - set(ALPHABET):
                raise ValueError("Expected canonical nine-residue peptides")
            rows.append(
                encode_peptide(
                    peptide, self.cfg["dataset"]["positions"], self.category_codes
                )
            )
        bits = torch.tensor(rows, dtype=torch.float32)
        if self.name in BASELINE_MODELS:
            if shots:
                raise ValueError("Classical checkpoints do not support optical shots")
            return predict_baseline(self.model, self.name, bits)
        return predict_probabilities(self.model, self.cfg, bits, shots=shots).numpy()


class QCHIPP:
    """Combine completed binding and binder-only immunogenicity heads by AND.

    Parameters
    ----------
    binding_run : str | Path
        Completed binding run directory under this paper's ``outdir/``.
    immunogenic_run : str | Path
        Completed immunogenicity run directory under this paper's ``outdir/``.

    Notes
    -----
    This is a reproducibility interface, not a clinically validated predictor.
    The AND is a binary decision, not a calibrated joint probability.
    """

    def __init__(self, binding_run, immunogenic_run):
        torch.set_num_threads(1)
        self.binding = _Head(binding_run, "binding")
        self.immunogenic = _Head(immunogenic_run, "immunogenic")
        if (
            self.binding.name != self.immunogenic.name
            or self.binding.source_sha256 != self.immunogenic.source_sha256
        ):
            raise ValueError("Combine the same model family and source dataset version")
        if self.binding.cfg["model"].get("source_labels") != self.immunogenic.cfg[
            "model"
        ].get("source_labels"):
            raise ValueError("Do not combine inconsistent source-label conventions")

    def predict(self, peptides, shots: int = 0) -> dict:
        """Return both head probabilities and their fixed-threshold AND decision.

        Parameters
        ----------
        peptides : sequence of str
            Canonical nine-residue peptides.
        shots : int
            Shots per optical cell for MerLin heads; 0 is analytic.
            Default value is 0.

        Returns
        -------
        dict
            ``binding_probability``, ``immunogenic_probability`` and the
            boolean ``positive`` array.
        """
        if isinstance(peptides, str):
            raise ValueError("Pass a nonempty sequence of peptide strings")
        peptides = list(peptides)
        if not peptides:
            raise ValueError("Pass a nonempty sequence of peptide strings")
        binding = self.binding.predict(peptides, shots)
        immunogenic = self.immunogenic.predict(peptides, shots)
        return {
            "binding_probability": binding,
            "immunogenic_probability": immunogenic,
            "positive": np.logical_and(binding >= 0.5, immunogenic >= 0.5),
        }
