"""Layered RY-RZ-ring-CNOT circuits (paper Eq. 17, Fig. 2, Fig. 4).

One layer applies RY(alpha_j) then RZ(beta_j) on every qubit, then CNOT from
qubit j to qubit (j + 1) mod n. The paper gives the gate set and layer count
but not the full QNN decomposition; 4 layers on 4 qubits give the 32
parameters of Table 1 (STL), which is how the QNN ansatz is inferred here
(LOG.md, Reproducibility Assessment).
"""

from __future__ import annotations

import itertools
import math

import numpy as np

from .statevector import apply_1q, apply_cnot, ry, rz


def params_per_layer(n_qubits: int) -> int:
    return 2 * n_qubits


def n_params(n_qubits: int, layers: int) -> int:
    return params_per_layer(n_qubits) * layers


def apply_rotations(
    state: np.ndarray, layer_params: np.ndarray, n_qubits: int
) -> np.ndarray:
    """Apply RY(alpha_j) then RZ(beta_j) on every qubit, with no entangling gate."""

    alphas, betas = layer_params[:n_qubits], layer_params[n_qubits:]
    for qubit in range(n_qubits):
        state = apply_1q(state, ry(alphas[qubit]), qubit, n_qubits)
    for qubit in range(n_qubits):
        state = apply_1q(state, rz(betas[qubit]), qubit, n_qubits)
    return state


def apply_layer(
    state: np.ndarray, layer_params: np.ndarray, n_qubits: int
) -> np.ndarray:
    """Apply one RY-RZ-ring-CNOT layer; ``layer_params`` holds alphas then betas."""

    state = apply_rotations(state, layer_params, n_qubits)
    if n_qubits > 1:
        for qubit in range(n_qubits):
            state = apply_cnot(state, qubit, (qubit + 1) % n_qubits, n_qubits)
    return state


def apply_ansatz(state: np.ndarray, params: np.ndarray, n_qubits: int) -> np.ndarray:
    """Apply ``len(params) / (2 n)`` layers in order."""

    layers = np.asarray(params).reshape(-1, params_per_layer(n_qubits))
    for layer_params in layers:
        state = apply_layer(state, layer_params, n_qubits)
    return state


def apply_share_and_specify(
    state: np.ndarray,
    shared_params: np.ndarray,
    specific_params: np.ndarray,
    n_qubits: int,
    asset_block: str,
) -> np.ndarray:
    """Share-and-specify ansatz with the asset label fixed (paper Eq. 24, 30).

    With the label register in a basis state ``|k>``, every label-controlled
    gate of asset ``k`` acts unconditionally and the other assets' gates act
    as identity, so ``U(theta) = prod_l U_k^l U_s^l`` on the data register
    (Eq. 30). Layer ``l`` applies the shared layer (RY, RZ, ring CNOT), then
    asset ``k``'s block.

    ``asset_block``: ``"rotations"`` is Fig. 7: the asset block is
    label-controlled RY and RZ on each data qubit, with no CNOT.
    ``"layer"`` reproduces the implementation used before 2026-11-19, whose
    asset block also carried a ring CNOT; since every asset block holds the
    same parameter-free ring, that ring acts for every label.
    """

    shared = np.asarray(shared_params).reshape(-1, params_per_layer(n_qubits))
    specific = np.asarray(specific_params).reshape(-1, params_per_layer(n_qubits))
    if len(shared) != len(specific):
        raise ValueError("shared and specific blocks need the same number of layers")
    if asset_block == "rotations":
        apply_asset = apply_rotations
    elif asset_block == "layer":
        apply_asset = apply_layer
    else:
        raise ValueError(f"Unknown model.qmtl_asset_block '{asset_block}'")
    for shared_layer, specific_layer in zip(shared, specific):
        state = apply_layer(state, shared_layer, n_qubits)
        state = apply_asset(state, specific_layer, n_qubits)
    return state


def context_subsets(context_length: int, order: int) -> list[tuple[int, ...]]:
    """Context-qubit subsets that control one rotation each, sizes 0..``order``.

    The empty subset is the uncontrolled RY on the prediction qubit.
    """

    if not 0 <= order <= context_length:
        raise ValueError(f"interaction order {order} outside 0..{context_length}")
    return [
        subset
        for size in range(order + 1)
        for subset in itertools.combinations(range(context_length), size)
    ]


def n_context_controlled_params(context_length: int, order: int) -> int:
    return sum(math.comb(context_length, size) for size in range(order + 1))


def apply_context_controlled(
    state: np.ndarray, params: np.ndarray, context_length: int, order: int
) -> np.ndarray:
    """Context-preserving ansatz: RY rotations on the prediction qubit only.

    Not in the paper (LOG.md, E11). One RY(theta_S) on the prediction qubit per
    subset ``S`` of context qubits with ``|S| <= order``, controlled on all
    qubits of ``S``. The context register is never rotated, so
    ``U |c>|0> = |c> (cos(a_c/2)|0> + sin(a_c/2)|1>)`` with
    ``a_c = sum of theta_S over subsets S whose qubits are all 1 in c``:
    Eq. 5 holds by construction and ``q(1 | c) = sin^2(a_c / 2)``.

    Basis order: index ``2 c + o`` with the oldest context step as the most
    significant bit of ``c`` (as in ``lib.data.context_indices``).
    """

    subsets = context_subsets(context_length, order)
    if len(params) != len(subsets):
        raise ValueError(f"expected {len(subsets)} parameters, got {len(params)}")
    n_contexts = 2**context_length
    if state.shape != (2 * n_contexts,):
        raise ValueError("context-controlled ansatz needs one binary prediction qubit")
    amplitudes = np.asarray(state, dtype=complex).reshape(n_contexts, 2)
    output = np.empty_like(amplitudes)
    for context in range(n_contexts):
        bits = [
            (context >> (context_length - 1 - q)) & 1 for q in range(context_length)
        ]
        angle = sum(
            theta
            for theta, subset in zip(params, subsets)
            if all(bits[q] for q in subset)
        )
        cos, sin = math.cos(angle / 2.0), math.sin(angle / 2.0)
        output[context, 0] = cos * amplitudes[context, 0] - sin * amplitudes[context, 1]
        output[context, 1] = sin * amplitudes[context, 0] + cos * amplitudes[context, 1]
    return output.reshape(-1)


def _context_bits(context_length: int) -> np.ndarray:
    """Bits of every context index, oldest step first; shape ``(2**T, T)``."""

    contexts = np.arange(2**context_length)
    shifts = np.arange(context_length - 1, -1, -1)
    return ((contexts[:, None] >> shifts) & 1).astype(float)


def _rotate_axis(amplitudes: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """RY on the axis of size 2 at position -2; ``angles`` broadcasts over the rest."""

    cos, sin = np.cos(angles / 2.0), np.sin(angles / 2.0)
    lower, upper = amplitudes[..., 0, :], amplitudes[..., 1, :]
    return np.stack([cos * lower - sin * upper, sin * lower + cos * upper], axis=-2)


def n_context_sequential_params(context_length: int, horizon: int) -> int:
    return sum(1 + context_length + step for step in range(horizon))


def apply_context_sequential(
    state: np.ndarray, params: np.ndarray, context_length: int, horizon: int
) -> np.ndarray:
    """Multi-step context-preserving ansatz without interference (E14, circuit A).

    Not in the paper. Prediction qubit ``k`` (step ``k`` after the context)
    gets one RY whose angle is additive over its controls: a bias, one term
    per context qubit that is 1, and one term per earlier prediction qubit
    that is 1. Steps are applied in order, so the model is the chain rule
    ``q(path | c) = prod_k q(step k | c, earlier steps)`` with
    ``q(step k up | ...) = sin^2(angle / 2)``, and its amplitudes on
    ``|c>|0..0>`` are real and non-negative for angles in ``[0, pi]``. The
    context register is never rotated.

    ``params`` holds, step by step, ``[bias, T context terms, k earlier-step
    terms]``. Basis order: index ``2**tau * c + o`` with the earliest step as
    the most significant bit of ``o`` (``lib.data.outcome_windows``).
    """

    expected = n_context_sequential_params(context_length, horizon)
    if len(params) != expected:
        raise ValueError(f"expected {expected} parameters, got {len(params)}")
    n_contexts = 2**context_length
    if state.shape != (n_contexts * 2**horizon,):
        raise ValueError("state does not match context_length and horizon")
    context_bits = _context_bits(context_length)
    amplitudes = np.asarray(state, dtype=complex)
    offset = 0
    for step in range(horizon):
        bias = params[offset]
        context_terms = np.asarray(params[offset + 1 : offset + 1 + context_length])
        earlier_terms = np.asarray(
            params[offset + 1 + context_length : offset + 1 + context_length + step]
        )
        offset += 1 + context_length + step
        earlier_bits = (
            (np.arange(2**step)[:, None] >> np.arange(step - 1, -1, -1)) & 1
        ).astype(float)
        angles = (
            bias
            + (context_bits @ context_terms)[:, None]
            + (earlier_bits @ earlier_terms)[None, :]
        )
        shaped = amplitudes.reshape(n_contexts, 2**step, 2, 2 ** (horizon - step - 1))
        amplitudes = _rotate_axis(shaped, angles[:, :, None]).reshape(-1)
    return amplitudes


def n_context_entangling_params(context_length: int, horizon: int, layers: int) -> int:
    return layers * horizon * (1 + context_length)


def apply_context_entangling(
    state: np.ndarray,
    params: np.ndarray,
    context_length: int,
    horizon: int,
    layers: int,
) -> np.ndarray:
    """Multi-step context-preserving ansatz with interference (E14, circuit B).

    Not in the paper. ``layers`` rotation layers on the prediction register;
    in each, every prediction qubit gets an RY whose angle is a bias plus one
    term per context qubit that is 1 (context-controlled RY). Between
    consecutive rotation layers a chain of CZ gates entangles neighbouring
    prediction qubits. The amplitudes stay real but can be negative, so paths
    interfere; the context register is never rotated.

    ``params`` has shape ``(layers, horizon, 1 + T)`` flattened: bias first,
    then the context terms, oldest step first.
    """

    expected = n_context_entangling_params(context_length, horizon, layers)
    if len(params) != expected:
        raise ValueError(f"expected {expected} parameters, got {len(params)}")
    if layers < 1:
        raise ValueError("model.prediction_layers must be at least 1")
    n_contexts = 2**context_length
    n_paths = 2**horizon
    if state.shape != (n_contexts * n_paths,):
        raise ValueError("state does not match context_length and horizon")
    context_bits = _context_bits(context_length)
    weights = np.asarray(params, dtype=float).reshape(
        layers, horizon, 1 + context_length
    )
    path_bits = (np.arange(n_paths)[:, None] >> np.arange(horizon - 1, -1, -1)) & 1
    # CZ on each neighbouring pair (k, k + 1): sign -1 when both bits are 1.
    cz_sign = np.where(
        (path_bits[:, :-1] & path_bits[:, 1:]).sum(axis=1) % 2 == 1, -1.0, 1.0
    )
    amplitudes = np.asarray(state, dtype=complex)
    for layer in range(layers):
        for qubit in range(horizon):
            angles = weights[layer, qubit, 0] + context_bits @ weights[layer, qubit, 1:]
            shaped = amplitudes.reshape(
                n_contexts, 2**qubit, 2, 2 ** (horizon - qubit - 1)
            )
            amplitudes = _rotate_axis(shaped, angles[:, None, None]).reshape(-1)
        if layer < layers - 1:
            amplitudes = (amplitudes.reshape(n_contexts, n_paths) * cz_sign).reshape(-1)
    return amplitudes
