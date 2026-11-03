"""Quantum and classical reservoirs (paper Sec. II, Figs. 1 and A1).

Quantum reservoir (the paper leaves the wiring and update unstated; our choices are in `docs/reproduction_details.md`):

* each of the N neurons is a 4-qubit circuit (:mod:`lib.quantum_neuron`) with
  4 inputs; ``k_i`` in {0, 1, 2} of them (drawn uniformly per neuron) are outputs
  of other reservoir neurons, the remaining ``4 - k_i`` are input lags;
* source neurons are drawn uniformly without replacement among all N neurons
  (self-loops allowed, Fig. 1 draws one); data lags are drawn without
  replacement among the 4 lags; data angles go to the first qubits, recurrent
  angles to the last ones;
* every neuron output starts at 0 (``RX(0) = I``) and the data is propagated
  ``passes = 3`` times; the reservoir features are the outputs after the last pass;
* each time step is processed from its own 4-lag window: no state is carried
  between windows.

Classical reservoir: ReservoirPy ``Reservoir`` (the framework named by the
paper) driven by the 4-lag window, with its state carried through time.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from .noise import IDEAL, DeviceNoise
from .quantum_neuron import N_QUBITS, neuron_expectations, neuron_functional

logger = logging.getLogger(__name__)

MAX_RECURRENT_INPUTS = 2


@dataclass
class QuantumReservoirSpec:
    """Random structure of one quantum reservoir, shared by its ideal and noisy versions."""

    weights: np.ndarray  # (N, 4) ansatz parameters, uniform in [0, 2 pi)
    data_lags: list[np.ndarray]  # per neuron: lag indices feeding the first qubits
    sources: list[np.ndarray]  # per neuron: neuron indices feeding the last qubits

    @property
    def n_neurons(self) -> int:
        return self.weights.shape[0]


def sample_quantum_reservoir(
    n_neurons: int, n_lags: int, rng: np.random.Generator
) -> QuantumReservoirSpec:
    """Draw the ansatz parameters and the wiring of an N-neuron reservoir."""

    if n_lags != N_QUBITS:
        raise ValueError(
            f"Each neuron has {N_QUBITS} inputs; the paper uses 4 lags, got {n_lags}"
        )
    weights = rng.uniform(0.0, 2 * np.pi, size=(n_neurons, N_QUBITS))
    data_lags, sources = [], []
    for _ in range(n_neurons):
        k = int(rng.integers(0, MAX_RECURRENT_INPUTS + 1))
        k = min(k, n_neurons)
        sources.append(np.sort(rng.choice(n_neurons, size=k, replace=False)))
        data_lags.append(np.sort(rng.choice(n_lags, size=N_QUBITS - k, replace=False)))
    return QuantumReservoirSpec(weights=weights, data_lags=data_lags, sources=sources)


class QuantumReservoir:
    """Exactly simulated quantum reservoir, ideal or with a device noise model.

    Parameters
    ----------
    spec : QuantumReservoirSpec
        Random structure from :func:`sample_quantum_reservoir`.
    noise : DeviceNoise
        Device noise model; ``IDEAL`` for the ideal simulator.
    observable : str
        ``"z_all"`` or ``"z0"``.
    passes : int
        Number of propagations through the reservoir (3 in the paper).
    update : str
        ``"synchronous"`` (all neurons read the previous pass) or ``"sequential"``
        (neurons read the latest outputs, in index order).
    shots : int or None
        If set, each expectation is replaced by a finite-shot estimate.
    rng : numpy.random.Generator or None
        Generator for shot sampling; required when ``shots`` is set.
    """

    def __init__(
        self,
        spec: QuantumReservoirSpec,
        noise: DeviceNoise,
        observable: str,
        passes: int,
        update: str,
        shots: int | None = None,
        rng: np.random.Generator | None = None,
    ):
        if update not in ("synchronous", "sequential"):
            raise ValueError(
                f"update must be 'synchronous' or 'sequential', got {update!r}"
            )
        if shots is not None and rng is None:
            raise ValueError("A generator is required for finite-shot sampling")
        self.spec, self.noise, self.passes, self.update = spec, noise, passes, update
        self.shots, self.rng = shots, rng
        self.functionals = np.stack(
            [neuron_functional(w, noise, observable) for w in spec.weights]
        )

    @property
    def n_neurons(self) -> int:
        return self.spec.n_neurons

    def _angles(
        self, windows: np.ndarray, outputs: np.ndarray, neurons: np.ndarray
    ) -> np.ndarray:
        angles = np.empty((windows.shape[0], neurons.size, N_QUBITS))
        for slot, i in enumerate(neurons):
            lags, src = self.spec.data_lags[i], self.spec.sources[i]
            angles[:, slot, : lags.size] = windows[:, lags]
            angles[:, slot, lags.size :] = outputs[:, src]
        return angles

    def _evaluate(self, angles: np.ndarray, neurons: np.ndarray) -> np.ndarray:
        values = neuron_expectations(self.functionals[neurons], angles, self.noise)
        if self.shots is not None:
            plus = self.rng.binomial(self.shots, np.clip((1 + values) / 2, 0.0, 1.0))
            values = 2 * plus / self.shots - 1
        return values

    def transform(self, windows: np.ndarray) -> np.ndarray:
        """Map ``(B, 4)`` normalized windows to ``(B, N)`` reservoir features."""

        outputs = np.zeros((windows.shape[0], self.n_neurons))
        everyone = np.arange(self.n_neurons)
        for _ in range(self.passes):
            if self.update == "synchronous":
                outputs = self._evaluate(
                    self._angles(windows, outputs, everyone), everyone
                )
            else:
                for i in everyone:
                    one = np.array([i])
                    outputs[:, i] = self._evaluate(
                        self._angles(windows, outputs, one), one
                    )[:, 0]
        return outputs


class ClassicalReservoir:
    """ReservoirPy echo-state reservoir driven by the 4-lag window.

    ``params`` holds ``lr``, ``sr``, ``input_scaling``, ``input_connectivity`` and
    ``rc_connectivity``; the paper does not report them, so the paper
    baseline uses the ReservoirPy 0.4.2 defaults (1.0, 1.0, 1.0, 0.1, 0.1).
    """

    def __init__(self, n_neurons: int, params: dict[str, Any], seed: int):
        from reservoirpy.nodes import Reservoir

        self.n_neurons = n_neurons
        self.node = Reservoir(
            units=n_neurons,
            lr=params["lr"],
            sr=params["sr"],
            input_scaling=params["input_scaling"],
            input_connectivity=params["input_connectivity"],
            rc_connectivity=params["rc_connectivity"],
            seed=seed,
        )

    def transform_sequence(self, splits: list[np.ndarray]) -> list[np.ndarray]:
        """Run the splits in chronological order, carrying the state across them."""

        lengths = [s.shape[0] for s in splits]
        states = self.node.run(np.concatenate(splits, axis=0))
        return np.split(np.asarray(states), np.cumsum(lengths)[:-1])


def noise_for(reservoir_type: str, device: DeviceNoise) -> DeviceNoise:
    """Device noise for a reservoir type: ``quantum_ideal`` ignores the device."""

    return IDEAL if reservoir_type == "quantum_ideal" else device
