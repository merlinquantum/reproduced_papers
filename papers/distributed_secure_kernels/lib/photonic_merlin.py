"""MerLin adaptation (not the paper's method): a photonic swap test for the overlap kernel.

The server of the paper estimates |<psi(x)|psi(y)>|^2 with a qubit swap test.
Its photonic counterpart is register-wise Hong-Ou-Mandel interference:

- each sample is amplitude-encoded into one m-mode register holding 2 photons,
  over the C(m, 2) collision-free Fock basis states (normalised, zero-padded);
- the two registers enter one ``QuantumLayer`` on 2m modes with 4 photons; a
  ``pcvl.PERM`` interleaves them (register mode i -> 2i, partner mode i -> 2i + 1)
  and one 50:50 ``pcvl.BS`` mixes each pair (2i, 2i + 1);
- photon-number-resolving detection (FOCK space) gives the parity of the photon
  number in the odd output ports, and P(even) = 1/2 + 1/2 |<psi(x)|psi(y)>|^2,
  the swap-test probability (Garcia-Escartin and Chamorro-Posada, 2013).

The teleportation stage has no deterministic linear-optics counterpart and is not
modelled (see the README, "Photonic Assessment and MerLin Extension"). Threshold detectors cannot
implement the parity readout, hence the FOCK computation space.
"""

from __future__ import annotations

import math
from functools import cache

import merlin as ml
import numpy as np
import perceval as pcvl
import torch
from merlin.core import EncodingSpace, StateVector

PHOTONS_PER_REGISTER = 2


def register_modes(n_features: int, photons: int = PHOTONS_PER_REGISTER) -> int:
    """Smallest m with C(m, photons) >= n_features."""

    modes = photons
    while math.comb(modes, photons) < n_features:
        modes += 1
    return modes


def encode_registers(
    features: np.ndarray, modes: int, photons: int = PHOTONS_PER_REGISTER
) -> np.ndarray:
    """Normalised feature vectors zero-padded to the register basis size C(modes, photons)."""

    features = np.atleast_2d(np.asarray(features, dtype=float))
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Zero-norm sample cannot be amplitude-encoded")
    size = math.comb(modes, photons)
    if features.shape[1] > size:
        raise ValueError(
            f"{features.shape[1]} features do not fit C({modes}, {photons}) = {size}"
        )
    states = np.zeros((features.shape[0], size))
    states[:, : features.shape[1]] = features / norms
    return states


class PhotonicSwapTest:
    """Two-register Hong-Ou-Mandel swap test simulated with MerLin.

    Parameters
    ----------
    modes : int
        Modes per register (2 photons each).
    device : str
        Torch device for the MerLin simulation. Default value is "cpu".
    """

    def __init__(self, modes: int, device: str = "cpu") -> None:
        self.modes = modes
        self.photons = PHOTONS_PER_REGISTER
        self.device = torch.device(device)
        circuit = pcvl.Circuit(2 * modes)
        # Perceval components need consecutive ports: interleave the registers first.
        circuit.add(
            0,
            pcvl.PERM(
                [2 * i for i in range(modes)] + [2 * i + 1 for i in range(modes)]
            ),
        )
        for i in range(modes):
            circuit.add(2 * i, pcvl.BS())
        self.layer = ml.QuantumLayer(
            circuit=circuit,
            n_photons=2 * self.photons,
            measurement_strategy=ml.MeasurementStrategy.probs(
                computation_space=ml.ComputationSpace.FOCK
            ),
            device=self.device,
            dtype=torch.float64,
        )
        odd_ports = [2 * i + 1 for i in range(modes)]
        self.even_mask = torch.tensor(
            [sum(key[j] for j in odd_ports) % 2 == 0 for key in self.layer.output_keys],
            dtype=torch.float64,
            device=self.device,
        )
        self.pair_index, self.joint_size = _joint_basis_index(modes, self.photons)

    def _joint_states(self, left: np.ndarray, right: np.ndarray) -> StateVector:
        # Product state |left> (x) |right> written on the joint Fock basis (2m modes, 4 photons).
        batch = left.shape[0]
        joint = np.zeros((batch, self.joint_size))
        joint[:, self.pair_index.reshape(-1)] = np.einsum(
            "bi,bj->bij", left, right
        ).reshape(batch, -1)
        return StateVector.from_tensor(
            torch.tensor(joint, device=self.device),
            n_modes=2 * self.modes,
            n_photons=2 * self.photons,
            encoding=EncodingSpace.FOCK,
        )

    def even_probability(
        self, left: np.ndarray, right: np.ndarray, batch: int = 4096
    ) -> np.ndarray:
        """P(even parity) for paired rows ``left[k]``, ``right[k]`` (register amplitudes)."""

        out = np.empty(left.shape[0])
        with torch.no_grad():
            for start in range(0, left.shape[0], batch):
                stop = min(start + batch, left.shape[0])
                probs = self.layer(
                    self._joint_states(left[start:stop], right[start:stop])
                )
                out[start:stop] = (probs * self.even_mask).sum(dim=-1).cpu().numpy()
        return out

    def zero_probability_matrix(
        self, left: np.ndarray, right: np.ndarray, symmetric: bool
    ) -> np.ndarray:
        """P(even) for every (row, column) pair; only the lower triangle when ``symmetric``."""

        if symmetric:
            rows, cols = np.tril_indices(left.shape[0])
        else:
            rows, cols = (
                idx.reshape(-1) for idx in np.indices((left.shape[0], right.shape[0]))
            )
        values = self.even_probability(left[rows], right[cols])
        matrix = np.zeros((left.shape[0], right.shape[0]))
        matrix[rows, cols] = values
        if symmetric:
            matrix[cols, rows] = values
        return matrix


@cache
def _joint_basis_index(modes: int, photons: int) -> tuple[np.ndarray, int]:
    """Joint Fock-basis index of e_i (x) e_j for every pair of register basis states.

    ``StateVector.tensor_product`` only accepts single states, so the map is built
    once from basis vectors and reused for batched product states.
    """

    size = math.comb(modes, photons)
    eye = torch.eye(size, dtype=torch.float64)

    def register(vector):
        return StateVector.from_tensor(
            vector, n_modes=modes, n_photons=photons, encoding=EncodingSpace.UNBUNCHED
        )

    index = np.zeros((size, size), dtype=int)
    joint_size = None
    for i in range(size):
        for j in range(size):
            dense = (
                register(eye[i]).tensor_product(register(eye[j])).to_dense().reshape(-1)
            )
            index[i, j] = int(torch.argmax(dense.abs()))
            joint_size = dense.numel()
    return index, joint_size
