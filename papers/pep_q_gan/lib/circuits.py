"""Fixed MerLin entangling-layer circuits for the selected Fock experiment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .artifacts import write_json

SCHEMA = "merlin_fixed_entangling_layer_v1"


def generate_spec(modes: int = 32, layers: int = 1, seed: int = 42) -> dict:
    """Describe the fixed MerLin MZI mesh used by this release.

    Parameters
    ----------
    modes : int
        Mode count; smaller counts support bounded tests. Default is 32.
    layers : int
        Number of entangling layers. Only the selected one-layer case is accepted.
    seed : int
        MerLin circuit seed, independent of event and GAN RNGs. Default is 42.

    Returns
    -------
    dict
        Public construction recipe, without a privately supplied transfer matrix.
    """
    if type(modes) is not int or modes < 2:
        raise ValueError("At least two modes are required")
    if layers != 1:
        raise ValueError("The selected release uses one entangling layer")
    if type(seed) is not int or seed < 0:
        raise ValueError("Circuit seed must be a nonnegative integer")
    return {
        "schema": SCHEMA,
        "modes": modes,
        "layers": layers,
        "circuit_seed": seed,
        "model": "mzi",
        "trainable": False,
        "constructor": "merlin.CircuitBuilder.add_entangling_layer",
        "phase_rule": "Independent uniform phases in [0, 2*pi); not Haar sampling",
    }


def circuit_from_spec(spec: dict):
    """Compile the selected public recipe with MerLin and return its unitary.

    Parameters
    ----------
    spec : dict
        Recipe created by generate_spec; older circuit schemas are rejected.

    Returns
    -------
    tuple
        Perceval circuit and complex128 matrix, with output rows/input columns.
    """
    import merlin

    expected = generate_spec(spec["modes"], spec["layers"], spec["circuit_seed"])
    if spec != expected:
        raise ValueError(
            "Circuit specification differs from the selected MerLin recipe"
        )
    builder = merlin.CircuitBuilder(spec["modes"])
    builder.add_entangling_layer(
        trainable=False, model="mzi", seed=spec["circuit_seed"]
    )
    circuit = builder.to_pcvl_circuit()
    unitary = np.asarray(circuit.compute_unitary(), dtype=np.complex128)
    if not np.allclose(
        unitary.conj().T @ unitary, np.eye(spec["modes"]), atol=1e-10, rtol=0
    ):
        raise ValueError("Compiled circuit is not unitary")
    return circuit, unitary


def circuit_id(spec: dict) -> str:
    """Return the SHA256 of the complete construction recipe.

    Parameters
    ----------
    spec : dict
        Selected circuit recipe.

    Returns
    -------
    str
        Recipe identity; numerical unitary identity is recorded separately.
    """
    return hashlib.sha256(
        json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def save_circuit(spec: dict, folder: Path) -> tuple:
    """Compile and save the recipe, serialized circuit, and computed unitary.

    Parameters
    ----------
    spec : dict
        Selected MerLin construction recipe.
    folder : Path
        Destination within the current run.

    Returns
    -------
    tuple
        Compiled circuit and complex128 unitary.
    """
    from perceval.serialization import serialize

    folder.mkdir(parents=True, exist_ok=True)
    circuit, unitary = circuit_from_spec(spec)
    write_json(folder / "circuit_spec.json", spec)
    write_json(folder / "perceval_circuit.json", serialize(circuit))
    np.savez(folder / "unitary.npz", unitary=unitary)
    return circuit, unitary


def input_state(photons: int, modes: int = 32) -> list[int]:
    """Place single photons evenly; the production input is [1, 0] * 16.

    Parameters
    ----------
    photons : int
        Total photon number, no greater than the mode count.
    modes : int
        Number of modes. Default is 32.

    Returns
    -------
    list[int]
        Binary input occupations; output collisions remain allowed.
    """
    if type(photons) is not int or not 1 <= photons <= modes:
        raise ValueError("Photon number must lie in [1, modes]")
    state = np.zeros(modes, dtype=int)
    state[np.arange(photons) * modes // photons] = 1
    return state.tolist()


def fock_moments(
    unitary: np.ndarray, state: list[int], condition: str = "indist"
) -> tuple:
    """Compute ideal Dist or Indist moments for binary Fock inputs.

    Parameters
    ----------
    unitary : np.ndarray
        Output-row/input-column transfer matrix.
    state : list[int]
        Binary input occupations.
    condition : str
        ``dist`` or ``indist``. Default is ``indist``.

    Returns
    -------
    tuple
        Per-mode mean and covariance; no distribution certification is implied.
    """
    if condition not in {"dist", "indist"}:
        raise ValueError("Photon condition must be dist or indist")
    state = np.asarray(state)
    if state.shape != (len(unitary),) or not np.isin(state, (0, 1)).all():
        raise ValueError("This moment formula requires binary input occupations")
    occupied = unitary[:, state == 1]
    probabilities = abs(occupied) ** 2
    mean = probabilities.sum(axis=1)
    projection = occupied @ occupied.conj().T
    if condition == "dist":
        covariance = np.diag(mean) - probabilities @ probabilities.T
    else:
        covariance = (
            np.diag(mean) + abs(projection) ** 2 - 2 * probabilities @ probabilities.T
        )
    return mean, covariance
