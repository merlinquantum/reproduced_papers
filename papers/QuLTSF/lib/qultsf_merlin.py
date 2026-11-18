"""MerLin photonic adaptation of QuLTSF (Phase 4; an adaptation, not the paper).

QuLTSF's hidden layer amplitude-embeds a ``Linear(L -> 2**N)`` output on N
qubits, applies a trainable unitary and reads N quadratic forms ``<Z_i>``.
The photonic counterpart keeps that role in linear optics:

```text
x[L] -> Linear(L -> C(m, n)) -> L2 normalise
     -> amplitudes over the collision-free basis of n photons in m modes
     -> trainable universal interferometer (one MZI mesh, m*(m-1) phases)
     -> per-mode click probability under threshold detection (m values)
     -> Linear(m -> T)
```

With m = 14 modes and n = 4 photons the embedding has C(14, 4) = 1001
amplitudes, close to QuLTSF's 2**10 = 1024. A per-mode click probability is a
quadratic form of the state, like ``<Z_i>``; ``<Z_i> = 1 - 2 P(bit_i = 1)``
is affine in a qubit marginal, and the linear head absorbs that affine map.

Basis convention: the amplitude vector is laid out in MerLin's UNBUNCHED
basis order, which is ``itertools.combinations(range(m), n)`` order (checked
in ``tests/test_qultsf_merlin.py`` against an empty MerLin circuit). The
identity mode therefore equals the trainable mode with the interferometer set
to the identity.
"""

from __future__ import annotations

import itertools
import math

import merlin as ml
import torch
import torch.nn as nn
from merlin.core.computation_space import ComputationSpace
from merlin.measurement import MeasurementStrategy

INTERFEROMETERS = ("trainable", "identity")


def unbunched_occupancy(n_modes: int, n_photons: int) -> torch.Tensor:
    """Occupancy matrix of the collision-free basis.

    Parameters
    ----------
    n_modes : int
        Number of optical modes m.
    n_photons : int
        Number of photons n.

    Returns
    -------
    torch.Tensor
        ``[C(m, n), m]`` float64 matrix; row ``b`` is 1 on the modes occupied
        by basis state ``b`` (``itertools.combinations`` order).
    """
    states = list(itertools.combinations(range(n_modes), n_photons))
    occupancy = torch.zeros(len(states), n_modes, dtype=torch.float64)
    for row, modes in enumerate(states):
        occupancy[row, list(modes)] = 1.0
    return occupancy


class PhotonicAmplitudeLayer(nn.Module):
    """Amplitude-encoded n-photon state, interferometer, threshold readout.

    Parameters
    ----------
    n_modes : int
        Number of modes m.
    n_photons : int
        Number of photons n; at least 2 for a meaningful photonic model.
    interferometer : str
        ``"trainable"``: one universal MZI mesh simulated by MerLin.
        ``"identity"``: no interferometer; the click probabilities are read
        from the encoded state directly (the photonic counterpart of the
        Phase 3 identity-circuit control, computed classically).
    dtype : torch.dtype
        Real dtype of the classical tensors; the MerLin layer uses the
        matching complex dtype.

    Raises
    ------
    ValueError
        If ``interferometer`` is unknown or the photon/mode counts are invalid.
    """

    def __init__(
        self, n_modes: int, n_photons: int, interferometer: str, dtype: torch.dtype
    ) -> None:
        super().__init__()
        if interferometer not in INTERFEROMETERS:
            raise ValueError(
                f"interferometer must be one of {INTERFEROMETERS}, got {interferometer!r}"
            )
        if not 2 <= n_photons < n_modes:
            raise ValueError(
                f"Need 2 <= n_photons < n_modes, got n_photons={n_photons}, n_modes={n_modes}"
            )
        self.n_modes = n_modes
        self.n_photons = n_photons
        self.interferometer = interferometer
        self.basis_size = math.comb(n_modes, n_photons)
        self.complex_dtype = (
            torch.complex128 if dtype == torch.float64 else torch.complex64
        )
        if interferometer == "trainable":
            builder = ml.CircuitBuilder(n_modes=n_modes)
            builder.add_entangling_layer(trainable=True, model="mzi", name="U")
            self.layer = ml.QuantumLayer(
                builder=builder,
                n_photons=n_photons,
                measurement_strategy=MeasurementStrategy.mode_expectations(
                    ComputationSpace.UNBUNCHED
                ),
                dtype=dtype,
            )
            if self.layer.output_size != n_modes:
                raise ValueError(
                    f"Expected {n_modes} mode expectations, got {self.layer.output_size}"
                )
        else:
            self.layer = None
            self.register_buffer(
                "occupancy", unbunched_occupancy(n_modes, n_photons).to(dtype)
            )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return each mode's click probability for each row of ``features``.

        Parameters
        ----------
        features : torch.Tensor
            Real ``[batch, C(m, n)]``; each row is L2-normalised into a state.

        Returns
        -------
        torch.Tensor
            ``[batch, m]`` click probabilities; each row sums to n.
        """
        amplitudes = features / torch.linalg.vector_norm(features, dim=-1, keepdim=True)
        if self.layer is None:
            return (amplitudes * amplitudes) @ self.occupancy
        # Propagate all basis states in one chunk: the same result as MerLin's
        # default chunking (to float32 rounding) with fewer, larger kernels;
        # MerLin's per-step cost is set by the basis size, not the batch.
        return self.layer(
            amplitudes.to(self.complex_dtype), simultaneous_processes=self.basis_size
        ).to(features.dtype)


class QuLTSFMerlin(nn.Module):
    """QuLTSF with the qubit hidden layer replaced by :class:`PhotonicAmplitudeLayer`.

    Parameters
    ----------
    seq_len : int
        Lookback length L.
    pred_len : int
        Forecast horizon T.
    n_modes : int
        Number of modes m.
    n_photons : int
        Number of photons n.
    interferometer : str
        ``"trainable"`` or ``"identity"``.
    dtype : torch.dtype
        Parameter dtype.
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        n_modes: int,
        n_photons: int,
        interferometer: str,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        basis_size = math.comb(n_modes, n_photons)
        self.input_classical_layer = nn.Linear(seq_len, basis_size, dtype=dtype)
        self.hidden_photonic_layer = PhotonicAmplitudeLayer(
            n_modes, n_photons, interferometer, dtype
        )
        self.output_classical_layer = nn.Linear(n_modes, pred_len, dtype=dtype)

    def _rows(self, rows: torch.Tensor) -> torch.Tensor:
        hidden = self.input_classical_layer(rows)
        hidden = self.hidden_photonic_layer(hidden)
        return self.output_classical_layer(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, channels = x.shape
        rows = x.permute(0, 2, 1).reshape(batch * channels, seq_len)
        return self._rows(rows).reshape(batch, channels, -1).permute(0, 2, 1)


def qultsf_merlin_parameter_count(
    seq_len: int, pred_len: int, n_modes: int, n_photons: int, interferometer: str
) -> int:
    """Trainable parameters of :class:`QuLTSFMerlin` (mesh: m*(m-1) phases)."""
    basis_size = math.comb(n_modes, n_photons)
    mesh = n_modes * (n_modes - 1) if interferometer == "trainable" else 0
    return seq_len * basis_size + basis_size + mesh + n_modes * pred_len + pred_len
