"""Photonic circuit construction and execution for the ObliQ solvers.

Every model is a MerLin :class:`~merlin.QuantumLayer` emitting Fock-space
probabilities, so it is an ordinary ``torch.nn.Module``: differentiable on the
local simulator, and callable through a remote Quandela processor.

Three builders, matching the paper's three variants:

* :func:`static_model` -- anchor layers only, no trainable parameters.
* :func:`vqc_model`    -- trainable mesh only, no anchor.
* :func:`obliq_model`  -- anchor layers followed by the trainable mesh (hybrid).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import perceval as pcvl
import torch
from merlin import MeasurementStrategy, QuantumLayer
from merlin.core import ComputationSpace
from models.backend import read_quandela_api_key
from perceval.components.unitary_components import BS, PERM, PS


@dataclass
class SimulationConfig:
    """How to execute a model.

    Attributes:
        nsamples: shots per forward pass. Only meaningful on a remote backend --
            the local simulator returns exact probabilities.
        backend: ``None`` for the local differentiable simulator, or a Quandela
            platform name (``"sim:slos"``, ``"sim:ascella"``, ``"qpu:ascella"``).
        token: explicit Quandela API token; falls back to env/file lookup.
    """

    nsamples: int = 1_024
    backend: str | None = None
    token: str | None = None


# ---------------------------------------------------------------------------
# Execution backend (local simulator vs remote processor)
# ---------------------------------------------------------------------------
def _resolve_processor(config: SimulationConfig):
    """Build the inference backend for ``config``.

    ``backend=None`` -> local differentiable simulator (default; no token, no
    cloud). A backend name is wrapped in a ``MerlinProcessor`` around a
    ``RemoteProcessor``, authenticated with the resolved Quandela token.
    """
    if config.backend is None:
        return None

    from merlin.core.merlin_processor import MerlinProcessor

    token = read_quandela_api_key(config.token)
    remote = (
        pcvl.RemoteProcessor(config.backend, token=token)
        if token
        else pcvl.RemoteProcessor(config.backend)
    )
    return MerlinProcessor(remote)


def run_distribution(model, config: SimulationConfig) -> torch.Tensor:
    """Forward pass returning the output distribution as a ``(1, D)`` tensor.

    Local: exact probabilities from the differentiable simulator, so the result
    is a deterministic function of the model parameters. Remote: ``nsamples``
    shots on the platform, so the distribution is sampled and (on hardware)
    irreducibly stochastic -- no seed can make a QPU repeat itself.
    """
    processor = _resolve_processor(config)
    if processor is None:
        with torch.no_grad():
            return model()
    return processor.forward(model.eval(), torch.zeros(1, 0), nsample=config.nsamples)


# ---------------------------------------------------------------------------
# Circuit layers
# ---------------------------------------------------------------------------
def _add_anchor_layers(
    circ: pcvl.Circuit, size: int, num_rep: int, theta_values: Sequence[float]
) -> None:
    """Add ``num_rep`` copies of the QUBO-encoding beam-splitter layer.

    One beam splitter per mode pair ``(i, j)``, with the angle taken from
    ``theta_values`` in strict-upper-triangular row-major order -- the same order
    :func:`models.obliq.qubo_norm_to_theta` emits. Non-adjacent pairs are brought
    together by a permutation, split, then permuted back.

    Splitting each angle across ``num_rep`` repetitions (rather than applying it
    once) is what makes the anchor a gentle, distributed encoding.
    """
    pairs = [(i, j) for i in range(size) for j in range(i + 1, size)]
    for _ in range(num_rep):
        for idx, (i, j) in enumerate(pairs):
            angle = float(theta_values[idx]) / num_rep
            if j == i + 1:
                circ.add((i, i + 1), BS(angle, 0, 0, 0, 0))
            else:
                perm = list(range(size))
                perm[i + 1], perm[j] = perm[j], perm[i + 1]
                circ.add(tuple(range(size)), PERM(perm))
                circ.add((i, i + 1), BS(angle, 0, 0, 0, 0))
                circ.add(tuple(range(size)), PERM(perm))


def _add_mixing_layer(circ: pcvl.Circuit, size: int) -> None:
    """Fixed 50:50 beam splitters between adjacent modes (theta = pi/2).

    Only :func:`vqc_model` needs this, immediately before :func:`_add_vqc_layers`.
    Independent per-mode phases applied to a single definite Fock basis state
    are provably just a global phase which cancels out of every measurement 
    probability, so that layer's ``size`` parameters would be inert regardless 
    of training. This layer creates real interference first, hence
    the phases downstream of it are physically meaningful. :func:`obliq_model`
    does not need it: its anchor layer already mixes modes before the same mesh
    code runs.
    """
    for i in range(size - 1):
        circ.add((i, i + 1), BS(np.pi / 2))


def _add_vqc_layers(circ: pcvl.Circuit, size: int) -> None:
    """Add the trainable mesh: two blocks of phase shifters and beam splitters.

    Every gate parameter is named ``coeffs<k>`` so MerLin groups them into a
    single trainable tensor (see :func:`get_coeff_parameter`).
    """
    k = 0
    for _ in range(2):
        for i in range(size):
            circ.add(i, PS(pcvl.P(f"coeffs{k}")))
            k += 1
        for i in range(size - 1):
            circ.add((i, i + 1), BS(pcvl.P(f"coeffs{k}")))
            k += 1
        for i in range(size):
            circ.add(i, PS(pcvl.P(f"coeffs{k}")))
            k += 1
        for i in reversed(range(1, size - 1)):
            circ.add((i - 1, i), BS(pcvl.P(f"coeffs{k}")))
            k += 1


# ---------------------------------------------------------------------------
# Trainable coefficients
# ---------------------------------------------------------------------------
def expected_coeff_count(size: int) -> int:
    """Number of trainable coefficients in the VQC mesh for ``size`` modes."""
    return max(2, 8 * size - 6)


def prepare_coeff_vector(coeffs: Sequence[float] | None, size: int) -> np.ndarray:
    """Validate a coefficient vector, defaulting to all zeros.

    All-zero coefficients make the VQC mesh the identity, which is what lets the
    untrained hybrid reproduce its anchor seed exactly.
    """
    expected = expected_coeff_count(size)
    if coeffs is None:
        return np.zeros(expected, dtype=float)
    coeff_array = np.asarray(coeffs, dtype=float)
    if coeff_array.size < expected:
        raise ValueError(
            f"Expected at least {expected} coefficients, received {coeff_array.size}"
        )
    return coeff_array


def get_coeff_parameter(model) -> torch.Tensor:
    """Return the trainable ``coeffs`` tensor registered inside ``model``."""
    matches = [param for name, param in model.named_parameters() if "coeffs" in name]
    if not matches:
        raise RuntimeError("Model has no trainable 'coeffs' parameter to optimize.")
    if len(matches) > 1:
        raise RuntimeError(
            "Expected a single grouped 'coeffs' parameter; found multiple."
        )
    return matches[0]


def set_coeff_parameter(model, coeffs) -> None:
    """Copy a flat coefficient vector into the model's trainable ``coeffs``.

    Called on every model before use, which is also why MerLin's own random
    parameter initialization never leaks into a result: it is overwritten with
    either zeros or a seeded draw (see :func:`models.obliq.train_obliq_vqc_coeffs`).
    """
    coeff_param = get_coeff_parameter(model)
    with torch.no_grad():
        values = torch.as_tensor(
            np.asarray(coeffs, dtype=float), dtype=coeff_param.dtype
        )
        coeff_param.copy_(values.reshape(coeff_param.shape))


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------
def _quantum_layer(circ: pcvl.Circuit, input_state, trainable: bool) -> QuantumLayer:
    """Wrap a circuit as a MerLin layer emitting Fock-space probabilities."""
    kwargs = {"trainable_parameters": ["coeffs"]} if trainable else {}
    return QuantumLayer(
        input_size=0,
        circuit=circ,
        input_state=list(input_state),
        dtype=torch.float32,
        measurement_strategy=MeasurementStrategy.probs(
            computation_space=ComputationSpace.FOCK
        ),
        **kwargs,
    )


def static_model(
    size: int, num_rep: int, theta_values: Sequence[float]
) -> QuantumLayer:
    """Anchor circuit only: the QUBO encoded into fixed beam-splitter angles.

    No trainable parameters -- a single forward pass *is* the zero-shot solution.
    """
    circ = pcvl.Circuit(size)
    _add_anchor_layers(circ, size, num_rep, theta_values)
    return _quantum_layer(circ, [1] * size, trainable=False)


def vqc_model(size: int) -> QuantumLayer:
    """Trainable mesh with no anchor -- the plain photonic VQC baseline.

    Prefixed with a fixed mixing layer (see :func:`_add_mixing_layer`) so the
    mesh's first phase-shifter layer acts on genuine interference instead of the
    raw, definite ``[1, 1, ..., 1]`` input.
    """
    circ = pcvl.Circuit(size)
    _add_mixing_layer(circ, size)
    _add_vqc_layers(circ, size)
    return _quantum_layer(circ, [1] * size, trainable=True)


def obliq_model(
    size: int, num_rep: int, input_state: Sequence[int], theta_values: Sequence[float]
) -> QuantumLayer:
    """Hybrid: the fixed anchor circuit followed by the trainable mesh.

    ``input_state`` is normally the static model's bitstring, so the search starts
    from ObliQ's zero-shot solution instead of a blank circuit.
    """
    circ = pcvl.Circuit(size)
    _add_anchor_layers(circ, size, num_rep, theta_values)
    _add_vqc_layers(circ, size)
    return _quantum_layer(circ, input_state, trainable=True)
