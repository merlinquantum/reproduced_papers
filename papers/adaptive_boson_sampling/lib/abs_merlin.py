"""MerLin adaptation of the ABS feature map (Phase 4; not the Phase 3 reproduction).

Two simulation paths for the same circuits as ``lib/circuits.py``:

- ``merlin`` (exact): each programmed mesh becomes a ``pcvl.Circuit`` in a
  ``merlin.QuantumLayer`` with ``MeasurementStrategy.partial`` on the adaptive
  detectors in ``ComputationSpace.FOCK``. The branch whose outcome is the
  observed detection string carries the conditional residual state |psi_p>.
  FOCK is required because B3 post-selects bunched outcomes (pseudo-PNR).
- ``merlin_noisy``: the imperfect-source model of SI Note 6, with
  ``pcvl.NoiseModel(indistinguishability, g2)`` on the source,
  ``MeasurementStrategy.probs(FOCK)`` (MerLin requires FOCK for source noise)
  and a simulated tomography station on the residual modes. Each residual
  density matrix is reconstructed by linear inversion of the post-selected
  probabilities, then projected to the PSD cone (the paper uses maximum
  likelihood). Losses are not modelled: balanced losses commute with the
  circuit and are removed by post-selection (SI Note 6).

Mode conventions follow ``lib/circuits.py`` (0-indexed). Perceval's default
``BS()`` is ``[[1, i], [i, 1]] / sqrt(2)``, the splitter inside our MZI, so
``MZI(theta) = BS // PS(theta) on the top mode // BS``.
"""

from __future__ import annotations

import itertools
import logging
import math
from typing import Any

import merlin as ml
import numpy as np
import perceval as pcvl
import torch

from .circuits import AbsPlatform, state_table
from .fock import fock_basis

logger = logging.getLogger(__name__)

SIMULATION_BACKENDS = ("numpy", "merlin", "merlin_noisy")
DETECTION_MODELS = ("threshold", "pnr")


def platform_circuit(
    platform: AbsPlatform, outcome: tuple[int, ...], u0: np.ndarray | None
) -> pcvl.Circuit:
    """Return the Perceval circuit programmed for ``outcome`` (Figs. 3-4 meshes)."""
    circuit = pcvl.Circuit(platform.n_modes)
    if u0 is not None:
        circuit.add(0, pcvl.Unitary(pcvl.Matrix(u0)))
    for kind, mode, value in platform.operations_for(outcome):
        if kind == "ps":
            circuit.add(mode, pcvl.PS(value))
        elif kind == "mzi":
            circuit.add(mode, pcvl.BS())
            circuit.add(mode, pcvl.PS(value))
            circuit.add(mode, pcvl.BS())
        else:
            raise ValueError(f"unknown operation kind {kind!r}")
    return circuit


def _input_state(platform: AbsPlatform) -> list[int]:
    return [
        1 if mode in platform.input_modes else 0 for mode in range(platform.n_modes)
    ]


def merlin_state_table(
    platform: AbsPlatform, u0: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(states, probabilities)`` like ``circuits.state_table``, via MerLin.

    ``states[c, k]`` is the normalised residual state for observed outcome ``c``
    under the circuit programmed for outcome ``k``; NaN where the post-selection
    probability is zero.
    """
    n_outcomes = len(platform.outcomes)
    states = np.full(
        (n_outcomes, n_outcomes, platform.output_dimension), np.nan, dtype=np.complex128
    )
    probabilities = np.zeros((n_outcomes, n_outcomes))
    index_of = {outcome: index for index, outcome in enumerate(platform.outcomes)}
    for circuit_index, programmed in enumerate(platform.outcomes):
        layer = ml.QuantumLayer(
            circuit=platform_circuit(platform, programmed, u0),
            input_state=_input_state(platform),
            n_photons=len(platform.input_modes),
            measurement_strategy=ml.MeasurementStrategy.partial(
                modes=list(platform.detected_modes),
                computation_space=ml.ComputationSpace.FOCK,
            ),
            return_object=True,
            dtype=torch.float64,
        )
        with torch.no_grad():
            result = layer()
        if tuple(result.unmeasured_modes) != tuple(platform.residual_modes):
            raise ValueError("MerLin residual modes differ from the platform's")
        for branch in result.branches:
            observed = tuple(int(count) for count in branch.outcome)
            if observed not in index_of:
                continue
            probability = float(branch.probability.reshape(-1)[0])
            if probability <= 1e-14:
                continue
            amplitudes = branch.amplitudes
            basis = [tuple(int(n) for n in state) for state in amplitudes.basis]
            expected_basis = _residual_basis(platform)
            if basis != expected_basis:
                raise ValueError(f"unexpected residual basis order {basis}")
            vector = amplitudes.tensor.detach().numpy().reshape(-1)
            observed_index = index_of[observed]
            states[observed_index, circuit_index] = vector / np.linalg.norm(vector)
            probabilities[observed_index, circuit_index] = probability
    return states, probabilities


def _residual_basis(platform: AbsPlatform) -> list[tuple[int, ...]]:
    n_residual = len(platform.input_modes) - sum(platform.outcomes[0])
    return fock_basis(len(platform.residual_modes), n_residual)


def tomography_settings(dimension: int) -> list[np.ndarray]:
    """Return the measurement unitaries of the tomography station on d modes.

    The identity (computational basis), then for every mode pair (i, j) a 50:50
    splitter (X-like) and a pi/2 phase on i followed by the splitter (Y-like).
    For d = 2 this is the Pauli set of SI Note 3; for d = 3 the 7 settings are
    informationally complete for a qutrit.
    """
    splitter = np.array([[1, 1j], [1j, 1]]) / math.sqrt(2)
    settings = [np.eye(dimension, dtype=np.complex128)]
    for i, j in itertools.combinations(range(dimension), 2):
        for phase in (0.0, math.pi / 2):
            unitary = np.eye(dimension, dtype=np.complex128)
            unitary[i, i] = np.exp(1j * phase)
            mix = np.eye(dimension, dtype=np.complex128)
            mix[np.ix_([i, j], [i, j])] = splitter
            settings.append(mix @ unitary)
    return settings


def reconstruct_density_matrix(
    settings: list[np.ndarray], frequencies: np.ndarray
) -> np.ndarray:
    """Return the PSD, unit-trace rho best fitting ``<j|V rho V^dag|j> = f[s, j]``.

    Linear inversion by least squares over Hermitian matrices, then negative
    eigenvalues are clipped and the trace renormalised.
    """
    dimension = settings[0].shape[0]
    basis = []
    for row in range(dimension):
        for column in range(dimension):
            element = np.zeros((dimension, dimension), dtype=np.complex128)
            if row == column:
                element[row, row] = 1
            elif row < column:
                element[row, column] = element[column, row] = 1
            else:
                element[row, column], element[column, row] = 1j, -1j
            basis.append(element)
    design, target = [], []
    for setting, row_frequencies in zip(settings, frequencies):
        for outcome in range(dimension):
            projector = setting.conj().T[:, [outcome]] @ setting[[outcome], :]
            design.append([np.trace(projector @ element).real for element in basis])
            target.append(row_frequencies[outcome])
    coefficients, *_ = np.linalg.lstsq(np.array(design), np.array(target), rcond=None)
    rho = sum(c * element for c, element in zip(coefficients, basis))
    rho = (rho + rho.conj().T) / 2
    values, vectors = np.linalg.eigh(rho)
    values = np.clip(values, 0, None)
    rho = (vectors * values) @ vectors.conj().T
    return rho / np.trace(rho).real


def _accepts_detection(
    observed: tuple[int, ...], target: tuple[int, ...], detection: str
) -> bool:
    if detection == "pnr":
        return observed == target
    # Threshold clicks; a bunched target (count 2) needs the pseudo-PNR fiber
    # splitter of B3, so it is matched by exact count.
    return all(
        (seen == want) if want >= 2 else ((seen > 0) == (want > 0))
        for seen, want in zip(observed, target)
    )


def noisy_density_table(
    platform: AbsPlatform, u0: np.ndarray | None, noise_cfg: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    """Return residual density matrices under the imperfect-source model.

    Parameters
    ----------
    platform : AbsPlatform
        Platform with one residual photon (A, B1, B2, B3).
    u0 : numpy.ndarray | None
        Fixed ``U_0`` for B2/B3; None for A/B1.
    noise_cfg : dict
        ``indistinguishability`` and ``g2`` (Perceval source model) and
        ``detection`` (``"threshold"`` or ``"pnr"``) for post-selection.

    Returns
    -------
    density : numpy.ndarray
        ``(D, D, d, d)``; ``density[c, k]`` for observed ``c`` under circuit ``k``,
        NaN when nothing is post-selected.
    probabilities : numpy.ndarray
        ``(D, D)`` post-selection probability in the computational-basis setting.
    """
    detection = noise_cfg["detection"]
    if detection not in DETECTION_MODELS:
        raise ValueError(f"detection must be one of {DETECTION_MODELS}")
    n_residual = len(platform.input_modes) - sum(platform.outcomes[0])
    if n_residual != 1:
        raise ValueError("the tomography model supports one residual photon")
    residual = list(platform.residual_modes)
    if residual != list(range(residual[0], residual[0] + len(residual))):
        raise ValueError("residual modes must be contiguous for the tomography block")
    dimension = len(residual)
    settings = tomography_settings(dimension)
    noise = pcvl.NoiseModel(
        indistinguishability=noise_cfg["indistinguishability"], g2=noise_cfg["g2"]
    )
    n_outcomes = len(platform.outcomes)
    counts = np.zeros((n_outcomes, n_outcomes, len(settings), dimension))
    for circuit_index, programmed in enumerate(platform.outcomes):
        for setting_index, setting in enumerate(settings):
            circuit = platform_circuit(platform, programmed, u0)
            circuit.add(residual[0], pcvl.Unitary(pcvl.Matrix(setting)))
            layer = ml.QuantumLayer(
                circuit=circuit,
                input_state=_input_state(platform),
                n_photons=len(platform.input_modes),
                noise=noise,
                measurement_strategy=ml.MeasurementStrategy.probs(
                    computation_space=ml.ComputationSpace.FOCK
                ),
                dtype=torch.float64,
            )
            with torch.no_grad():
                distribution = layer().detach().numpy().reshape(-1)
            # With g2 > 0 the photon number varies: read outcomes by key.
            for key, probability in zip(layer.output_keys, distribution):
                key = tuple(int(n) for n in key)
                residual_counts = [key[mode] for mode in residual]
                clicks = [mode for mode, n in enumerate(residual_counts) if n > 0]
                if len(clicks) != 1:
                    continue
                if detection == "pnr" and residual_counts[clicks[0]] != 1:
                    continue
                detected = tuple(key[mode] for mode in platform.detected_modes)
                for observed_index, target in enumerate(platform.outcomes):
                    if _accepts_detection(detected, target, detection):
                        counts[
                            observed_index, circuit_index, setting_index, clicks[0]
                        ] += probability
        logger.debug("noisy tomography | circuit %d/%d", circuit_index + 1, n_outcomes)

    density = np.full(
        (n_outcomes, n_outcomes, dimension, dimension), np.nan, dtype=np.complex128
    )
    probabilities = counts[:, :, 0, :].sum(axis=-1)
    for observed_index, circuit_index in np.ndindex(n_outcomes, n_outcomes):
        per_setting = counts[observed_index, circuit_index]
        totals = per_setting.sum(axis=-1)
        if np.any(totals <= 1e-14):
            continue
        density[observed_index, circuit_index] = reconstruct_density_matrix(
            settings, per_setting / totals[:, None]
        )
    return density, probabilities


def simulated_state_table(
    platform: AbsPlatform, u0: np.ndarray | None, simulation: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    """Dispatch to the configured simulation backend.

    ``simulation["backend"]`` is ``"numpy"`` (Phase 3 reference, ``lib/fock.py``),
    ``"merlin"`` (exact) or ``"merlin_noisy"`` (needs ``simulation["noise"]``).
    States are ``(D, D, d)`` vectors, or ``(D, D, d, d)`` density matrices for
    ``merlin_noisy``.
    """
    backend = simulation["backend"]
    if backend == "numpy":
        return state_table(platform, u0)
    if backend == "merlin":
        return merlin_state_table(platform, u0)
    if backend == "merlin_noisy":
        return noisy_density_table(platform, u0, simulation["noise"])
    raise ValueError(f"unknown backend {backend!r}; expected {SIMULATION_BACKENDS}")
