"""MerLin photonic adaptation of the paper's reservoir and kernel readout.

This is an **adaptation**, not the paper's method. See the README, *Photonic Assessment and
MerLin Extension*, for the scope (`PARTIAL_MERLIN_TRANSLATION`).

* Photonic neuron: each reservoir neuron is a small frozen
  linear-optical chip: 4 modes, 2 photons in ``[1, 0, 1, 0]``, a fixed random
  interferometer, angle encoding of the neuron's 4 inputs (``scale * x`` phase
  on each mode), a second fixed random interferometer, threshold detection
  (``UNBUNCHED`` space). The output is ``<n_0> - 1`` in [-1, 1], playing the role
  of the qubit neuron's ``<Z (x) Z (x) Z (x) Z>``. The reservoir wiring (which
  lags and which neurons feed each chip) and the 3 synchronous passes are those of
  :mod:`lib.reservoirs`.
* Photonic fidelity kernel: ``n_modes`` modes (the paper's kernel
  qubit count, 5 or 10), 2 photons at evenly spaced modes, a fixed random
  interferometer, then ``B = 2`` blocks of ``L = ceil(d / n_modes)`` layers of
  [angle encoding of ``n_modes`` features, fixed random interferometer]. The
  kernel ``|<s|U(x')^dagger U(x)|s>|^2 = |<psi(x')|psi(x)>|^2`` is computed
  exactly from the Fock-space amplitudes. Physically it is the
  compute-uncompute probability of detecting the (unbunched) input pattern with
  threshold detectors. Unlike the paper's qubit feature map, two-photon
  interference makes this kernel non-factorizable over modes.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from .metrics import rmse
from .readouts import FittedReadout, _cholesky_solve, select_lambda
from .reservoirs import QuantumReservoirSpec

NEURON_MODES = 4
NEURON_INPUT_STATE = [1, 0, 1, 0]


def _seed(sequence: np.random.SeedSequence) -> int:
    return int(sequence.generate_state(1)[0] % (2**31 - 1))


def evenly_spaced_input_state(n_modes: int, n_photons: int) -> list[int]:
    """One photon per mode at evenly spaced positions (e.g. 10 modes, 2 photons -> modes 0 and 5)."""

    if n_photons > n_modes:
        raise ValueError(
            f"{n_photons} photons do not fit in {n_modes} modes without bunching"
        )
    state = [0] * n_modes
    for i in range(n_photons):
        state[(i * n_modes) // n_photons] = 1
    return state


def build_neuron_layer(scale: float, sequence: np.random.SeedSequence):
    """Frozen 4-mode, 2-photon neuron chip returning per-mode photon expectations."""

    import merlin as ml

    first, second = sequence.spawn(2)
    builder = ml.CircuitBuilder(n_modes=NEURON_MODES)
    builder.add_entangling_layer(trainable=False, seed=_seed(first))
    builder.add_angle_encoding(modes=list(range(NEURON_MODES)), scale=float(scale))
    builder.add_entangling_layer(trainable=False, seed=_seed(second))
    return ml.QuantumLayer(
        input_size=NEURON_MODES,
        builder=builder,
        input_state=NEURON_INPUT_STATE,
        n_photons=sum(NEURON_INPUT_STATE),
        measurement_strategy=ml.MeasurementStrategy.mode_expectations(
            computation_space=ml.ComputationSpace.UNBUNCHED
        ),
        dtype=torch.float64,
    )


class PhotonicReservoir:
    """Reservoir of frozen photonic neuron chips with the paper's wiring.

    Parameters
    ----------
    spec : QuantumReservoirSpec
        Wiring (data lags and source neurons per neuron) shared with the gate
        reservoir of the same seed; its ansatz angles are not used.
    scale : float
        Angle-encoding scale of the neuron chips.
    passes : int
        Number of synchronous propagations (3 in the paper).
    sequence : numpy.random.SeedSequence
        Seed of the fixed random interferometers.
    """

    def __init__(
        self,
        spec: QuantumReservoirSpec,
        scale: float,
        passes: int,
        sequence: np.random.SeedSequence,
    ):
        self.spec, self.passes = spec, passes
        self.layers = [
            build_neuron_layer(scale, s) for s in sequence.spawn(spec.n_neurons)
        ]

    @property
    def n_neurons(self) -> int:
        return self.spec.n_neurons

    def transform(self, windows: np.ndarray) -> np.ndarray:
        """Map ``(B, 4)`` normalized windows to ``(B, N)`` neuron outputs in [-1, 1]."""

        outputs = np.zeros((windows.shape[0], self.n_neurons))
        for _ in range(self.passes):
            new = np.empty_like(outputs)
            for i, layer in enumerate(self.layers):
                lags, src = self.spec.data_lags[i], self.spec.sources[i]
                angles = np.concatenate([windows[:, lags], outputs[:, src]], axis=1)
                with torch.no_grad():
                    expectations = layer(torch.from_numpy(angles))
                new[:, i] = expectations[:, 0].numpy() - 1.0
            outputs = new
        return outputs


class PhotonicKernelMap:
    """Fock-space state preparation of the photonic fidelity kernel."""

    def __init__(
        self,
        n_features: int,
        n_modes: int,
        n_photons: int,
        scale: float,
        blocks: int,
        sequence: np.random.SeedSequence,
    ):
        import merlin as ml

        self.n_features, self.n_modes, self.blocks = n_features, n_modes, blocks
        self.layers_per_block = math.ceil(n_features / n_modes)
        n_encodings = blocks * self.layers_per_block
        seeds = sequence.spawn(n_encodings + 1)
        builder = ml.CircuitBuilder(n_modes=n_modes)
        builder.add_entangling_layer(trainable=False, seed=_seed(seeds[0]))
        for k in range(n_encodings):
            builder.add_angle_encoding(modes=list(range(n_modes)), scale=float(scale))
            builder.add_entangling_layer(trainable=False, seed=_seed(seeds[k + 1]))
        self.input_state = evenly_spaced_input_state(n_modes, n_photons)
        self.layer = ml.QuantumLayer(
            input_size=n_encodings * n_modes,
            builder=builder,
            input_state=self.input_state,
            n_photons=n_photons,
            measurement_strategy=ml.MeasurementStrategy.amplitudes(
                computation_space=ml.ComputationSpace.FOCK
            ),
            dtype=torch.float64,
        )

    def encode(self, features: np.ndarray) -> np.ndarray:
        """Circuit inputs: features zero-padded to ``L * n_modes`` and repeated for each block."""

        padded = np.zeros((features.shape[0], self.layers_per_block * self.n_modes))
        padded[:, : self.n_features] = features
        return np.tile(padded, (1, self.blocks))

    def states(self, features: np.ndarray, chunk_size: int = 8192) -> torch.Tensor:
        """``(B, dim)`` complex Fock-space amplitudes ``|psi(x)>``."""

        inputs = self.encode(features)
        parts = []
        with torch.no_grad():
            for start in range(0, inputs.shape[0], chunk_size):
                parts.append(
                    self.layer(torch.from_numpy(inputs[start : start + chunk_size]))
                )
        return torch.cat(parts).to(torch.complex128)


def state_kernel_block(rows: torch.Tensor, cols: torch.Tensor) -> torch.Tensor:
    """``|<rows|cols>|^2`` between two sets of normalized state vectors."""

    return (rows.conj() @ cols.T).abs() ** 2


def state_gram(states: torch.Tensor, chunk_size: int = 4096) -> torch.Tensor:
    n = states.shape[0]
    gram = torch.empty(n, n, dtype=torch.float64, device=states.device)
    for start in range(0, n, chunk_size):
        gram[start : start + chunk_size] = state_kernel_block(
            states[start : start + chunk_size], states
        )
    return gram


def state_kernel_times_vector(
    rows: torch.Tensor, cols: torch.Tensor, vector: torch.Tensor, chunk_size: int = 4096
) -> torch.Tensor:
    out = torch.empty(rows.shape[0], dtype=torch.float64, device=rows.device)
    for start in range(0, rows.shape[0], chunk_size):
        out[start : start + chunk_size] = (
            state_kernel_block(rows[start : start + chunk_size], cols) @ vector
        )
    return out


def fit_photonic_kernel_ridge(
    features: dict[str, np.ndarray],
    targets: dict[str, np.ndarray],
    lambda_grid: list[float],
    kernel_map: PhotonicKernelMap,
    device: str,
    selection_subsample: int | None,
    fit_subsample: int | None,
) -> FittedReadout:
    """Kernel ridge with the photonic fidelity kernel.

    Same protocol as :func:`lib.readouts.fit_kernel_ridge`: centred target,
    n-normalized lambda selected on eval RMSE from an evenly spaced training
    subsample (ties within 1 % -> larger lambda), then one refit.
    """

    torch_device = torch.device(device)

    def evenly_spaced(limit: int | None) -> np.ndarray:
        n = features["train"].shape[0]
        if limit is None or limit >= n:
            return np.arange(n)
        return np.unique(np.linspace(0, n - 1, limit).round().astype(int))

    states = {
        split: kernel_map.states(values).to(torch_device)
        for split, values in features.items()
    }
    y_train = targets["train"]
    y_mean = float(y_train.mean())

    sel_index = evenly_spaced(selection_subsample)
    sel_states = states["train"][torch.from_numpy(sel_index).to(torch_device)]
    sel_rhs = torch.from_numpy(y_train[sel_index] - y_mean).to(
        torch_device, torch.float64
    )
    sel_gram = state_gram(sel_states)
    scores: dict[float, float] = {}
    for lam in lambda_grid:
        alpha = _cholesky_solve(sel_gram.clone(), sel_rhs, sel_index.size * lam)
        pred = (
            state_kernel_times_vector(states["eval"], sel_states, alpha).cpu().numpy()
            + y_mean
        )
        scores[lam] = rmse(targets["eval"], pred)
    del sel_gram
    chosen = select_lambda(scores)

    fit_index = evenly_spaced(fit_subsample)
    fit_states = states["train"][torch.from_numpy(fit_index).to(torch_device)]
    fit_rhs = torch.from_numpy(y_train[fit_index] - y_mean).to(
        torch_device, torch.float64
    )
    gram = state_gram(fit_states)
    alpha = _cholesky_solve(gram, fit_rhs, fit_index.size * chosen)
    del gram
    predictions = {
        split: state_kernel_times_vector(states[split], fit_states, alpha).cpu().numpy()
        + y_mean
        for split in features
    }
    details: dict[str, Any] = {
        "kernel": "photonic_fidelity",
        "n_modes": kernel_map.n_modes,
        "n_photons": sum(kernel_map.input_state),
        "input_state": kernel_map.input_state,
        "layers": kernel_map.layers_per_block,
        "blocks": kernel_map.blocks,
        "fock_dim": int(states["train"].shape[1]),
        "selection_points": int(sel_index.size),
        "fit_points": int(fit_index.size),
    }
    return FittedReadout(
        predictions=predictions,
        lambda_selected=chosen,
        selection_scores=scores,
        details=details,
    )
