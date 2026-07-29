"""Interpreting a photonic measurement: Fock outcome -> bitstring -> energy.

The two pieces both photonic solvers need. Each solver owns its own encoding and
decoding (see :mod:`models.obliq`, :mod:`models.cvar_vqe`); what they share is how
a Fock outcome becomes a bitstring, and how that bitstring's QUBO energy is looked
up during training.

This module deliberately depends on neither solver, so the baseline never has to
import from the method it is a baseline for.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import torch
from utils.qubo import qubo_objective


def number_mapping(
    occupation: Sequence[int], size: int, invert: bool = False
) -> np.ndarray:
    """Number mapping: a mode holding at least one photon is a selected variable.

    ``|3,0,2,1> -> |1,0,1,1>``. The occupation is sliced to ``size``, which drops a
    homogenization ancilla mode when the circuit was augmented.

    ``invert`` complements every bit, so an *empty* mode becomes the selected one.
    That is what the CVaR-VQE reference readout switches between: despite being
    called a parity there (``_parify_samples_threshold``), its formula
    ``(int(n == 0) + j) % 2`` is a threshold on zero followed by a global
    complement -- ``j = 1`` is this mapping and ``j = 0`` its inverse. It is *not*
    photon-number parity (``n % 2``, which would give ``|3,0,2,1> -> |1,0,0,1>``).
    """
    bits = np.where(np.asarray(occupation)[:size] >= 1, 1, 0)
    return 1 - bits if invert else bits


class EnergyTable:
    """QUBO energy of each measurement outcome, materialized lazily.

    An outcome's energy is constant w.r.t. the trainable parameters, so it is
    computed the first time that outcome is requested and memoized thereafter,
    keyed by its index into the model's ``output_keys``.

    Why lazy: the two optimizers touch the outcome space very differently.

    * Autograd (Adam/SGD) runs on the dense local simulator, which emits a
      probability for *every* outcome, so it needs the full basis -- use
      :meth:`full`. Gradients flow through all entries.
    * COBYLA trains on finite shots, so only the handful of outcomes actually
      sampled in a forward pass carry probability mass. Building energies for the
      whole Fock space (which grows combinatorially with modes and photons) would
      be wasted work -- :meth:`for_indices` computes only what was observed.

    ``mapping`` selects the readout, so ObliQ and the CVaR-VQE baseline (which
    inverts the mapping for one of its two passes) share this machinery instead of
    each carrying a copy.
    """

    def __init__(
        self,
        Q: np.ndarray,
        output_keys: Sequence,
        mapping: Callable[[Sequence[int], int], np.ndarray] = number_mapping,
    ):
        self._Q = np.asarray(Q, dtype=float)
        self._size = self._Q.shape[0]
        self._output_keys = output_keys
        self._mapping = mapping
        self._cache: dict[int, float] = {}

    def _energy(self, index: int) -> float:
        energy = self._cache.get(index)
        if energy is None:
            bits = self._mapping(self._output_keys[index], self._size)
            energy = qubo_objective(self._Q, bits)
            self._cache[index] = energy
        return energy

    def for_indices(self, indices) -> torch.Tensor:
        """Energies for the given outcome indices, order preserved."""
        return torch.tensor(
            [self._energy(int(i)) for i in indices], dtype=torch.float32
        )

    def full(self) -> torch.Tensor:
        """Materialize the energy of every outcome (dense autograd path)."""
        return self.for_indices(range(len(self._output_keys)))
