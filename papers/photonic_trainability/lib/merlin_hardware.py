"""Hardware conditions for the MerLin adaptation (Phase 04).

A condition is an explicit JSON object; every key is required so no hardware
effect is switched on or off by default:

```json
{"name": "loss_0.7_postselected",
 "noise": {"transmittance": 0.7},
 "detector": "pnr",
 "postselect": true}
```

* ``noise`` - keyword arguments of ``perceval.NoiseModel`` (``{}`` for none);
  ``indistinguishability`` models partially distinguishable photons and
  ``transmittance`` uniform loss.
* ``detector`` - ``"pnr"`` (photon-number resolving, the paper's model) or
  ``"threshold"`` (click / no-click on every mode). With threshold detectors
  MerLin returns click patterns, and photon-number observables are evaluated on
  the clicks, i.e. ``n_i`` is replaced by ``1[n_i >= 1]``.
* ``postselect`` - keep only outcomes with the input photon number and
  renormalise (the paper's Fig. 9 protocol); without it, lossy outcomes with
  fewer photons are kept and the observable is evaluated on what is detected.

The computation space is always ``ComputationSpace.FOCK``: the observables are
photon-number polynomials, so the full occupation pattern is needed (threshold
detection is applied by MerLin's detector transform on top of the Fock
distribution).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import merlin as ML
import numpy as np
import perceval as pcvl
import torch

REQUIRED_KEYS = ("name", "noise", "detector", "postselect")
DETECTORS = ("pnr", "threshold")


def validate_condition(condition: Mapping[str, Any]) -> None:
    """Fail clearly on a missing key or an unknown detector."""
    missing = [key for key in REQUIRED_KEYS if key not in condition]
    if missing:
        raise KeyError(f"Hardware condition {condition.get('name')!r} misses {missing}")
    if condition["detector"] not in DETECTORS:
        raise ValueError(
            f"Unknown detector {condition['detector']!r}; use one of {DETECTORS}"
        )
    unknown = set(condition["noise"]) - {"indistinguishability", "transmittance"}
    if unknown:
        raise ValueError(f"Unsupported noise parameters {sorted(unknown)}")


def build_layer(
    circuit: pcvl.Circuit,
    occupation: tuple[int, ...],
    condition: Mapping[str, Any] | None,
    device: torch.device,
) -> ML.QuantumLayer:
    """MerLin layer for a circuit under a hardware condition (``None`` = ideal PNR).

    Parameters
    ----------
    circuit : perceval.Circuit
        Circuit with phases named ``phi_00000, ...``.
    occupation : tuple of int
        Fock input state.
    condition : Mapping[str, Any] | None
        Hardware condition (module docstring) or ``None`` for the Phase 03 model.
    device : torch.device
        Torch device.

    Returns
    -------
    merlin.QuantumLayer
        Layer returning output probabilities over ``layer.output_keys``.
    """
    common = {
        "input_parameters": ["phi"],
        "input_state": pcvl.BasicState(list(occupation)),
        "measurement_strategy": ML.MeasurementStrategy.probs(ML.ComputationSpace.FOCK),
        "dtype": torch.float64,
        "device": device,
    }
    if condition is None:
        return ML.QuantumLayer(circuit=circuit, **common)
    validate_condition(condition)
    noise = pcvl.NoiseModel(**condition["noise"]) if condition["noise"] else None
    if condition["detector"] == "threshold":
        experiment = pcvl.Experiment(circuit)
        for mode in range(circuit.m):
            experiment.detectors[mode] = pcvl.Detector.threshold()
        return ML.QuantumLayer(experiment=experiment, noise=noise, **common)
    return ML.QuantumLayer(circuit=circuit, noise=noise, **common)


def outcome_mask(
    patterns: np.ndarray,
    occupation: tuple[int, ...],
    condition: Mapping[str, Any] | None,
) -> np.ndarray | None:
    """Boolean mask of kept outcomes under post-selection, or ``None`` if none applies."""
    if condition is None or not condition["postselect"]:
        return None
    if condition["detector"] == "threshold":
        raise ValueError(
            "Post-selection on photon number is undefined with threshold detectors"
        )
    return patterns.sum(axis=1) == sum(occupation)
