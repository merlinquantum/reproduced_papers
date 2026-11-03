"""MerLin photonic adaptation of the zero-sum-biased model (Phase 4).

This is an **adaptation**, not the paper's model. The paper enforces
sum_k E[Y_k | x] = 0 with qubit generators that commute with Z1 + Z2 + Z3
(Fig. 7). Here the same conservation law comes from photon-number
conservation: passive linear optics preserves the total photon number N for
every interferometer and phase encoding, and every collision-free detection
outcome holds exactly N photons. A readout that is linear in the detected
occupations n = (n_0, ..., n_5) therefore encodes the bias exactly for every
input and parameter value.

Circuit (6 modes, MerLin ``CircuitBuilder``)::

    U0 -> phases z(x) -> U1 -> phases z(x) -> U2      (L = 2 re-uploading)

with trainable MZI meshes U_l (30 phases each, 90 in total) and z(x) = (x1,
x2, x3, x4 x5, x5 x6, x4 x6), the Fourier variables of the qubit model.
Detection (``computation_space``):
- ``"unbunched"`` (default): threshold detectors with collision-free
  postselection (``ComputationSpace.UNBUNCHED``, which MerLin renormalises;
  see LOG.md);
- ``"fock"``: photon-number-resolving detectors, with no postselection
  (``ComputationSpace.FOCK``). This is the Phase 4 control for postselection
  acting as a hidden nonlinearity. The readouts use photon counts, so the
  bias stays exact.

Each outcome n has a response vector r(n) in [0, 1]^3 (the probability that a
sampled label of player k is +1), and P_k(+1 | x) = sum_n P(n | x) r_k(n).

- ``vertex``: r(n) = sum_m n_m v_m / N, with v_m the six hexagon vertices
  (``rps_game.HEXAGON_VERTICES``); every r(n) lies in the zero-sum plane.
  The behaviour region reaches V_{0.75} for N = 2.
- ``group``: player k owns modes (2k, 2k + 1), and
  r_k(n) = 1/4 + (3/4) (n_{2k} + n_{2k+1}) / N. This is the photonic analogue
  of the measurement-based approach. Behaviours lie in the triangle
  {P_k >= 1/4}, and the model is noncontextual for every parameter value.
- ``unbiased``: r(n) = sum_m n_m sigmoid(w_m) / N with 18 trainable w_m in
  R^3, not constrained to the plane (the unbiased counterpart).
"""

from __future__ import annotations

import warnings

import merlin as ml
import torch
from torch import nn

from .rps_game import HEXAGON_VERTICES
from .surrogate import fourier_variables

READOUTS = ("vertex", "group", "unbiased")
COMPUTATION_SPACES = {
    "unbunched": ml.ComputationSpace.UNBUNCHED,
    "fock": ml.ComputationSpace.FOCK,
}
N_MODES = 6
N_FEATURES = 6


def validate_input_state(
    input_state: list[int], n_modes: int, n_photons: int
) -> list[int]:
    """Check an explicit Fock input state; never repair it.

    Raises
    ------
    ValueError
        If the length, occupations or photon number do not match.
    """
    state = [int(v) for v in input_state]
    if len(state) != n_modes or any(v < 0 for v in state) or sum(state) != n_photons:
        raise ValueError(
            f"input_state {input_state} must have {n_modes} non-negative entries summing to n_photons={n_photons}"
        )
    if any(v > 1 for v in state):
        raise ValueError(
            f"input_state {input_state} must be collision-free for threshold detection"
        )
    return state


def _build_circuit(n_modes: int, layers: int) -> ml.CircuitBuilder:
    builder = ml.CircuitBuilder(n_modes=n_modes)
    for layer in range(layers):
        builder.add_entangling_layer(name=f"U{layer}")
        builder.add_angle_encoding(modes=list(range(n_modes)), name=f"x{layer}")
    builder.add_entangling_layer(name=f"U{layers}")
    return builder


class PhotonicZeroSumModel(nn.Module):
    """Photonic multi-task model returning E[Y_k | x] for the three players.

    Parameters
    ----------
    readout : str
        ``"vertex"``, ``"group"`` or ``"unbiased"``.
    n_photons : int
        Number of photons N (>= 2 for a scientific result).
    input_state : list of int
        Explicit collision-free Fock input state on 6 modes.
    layers : int
        Number of re-uploading encodings L (paper qubit model: 2).
    dtype : torch.dtype
        Simulation dtype (``torch.float64`` for paper runs).
    computation_space : str
        ``"unbunched"`` (threshold detection, collision-free postselection)
        or ``"fock"`` (photon-number-resolving detection, no postselection).
        Default value is ``"unbunched"``.

    Notes
    -----
    Initialisation uses the global torch RNG (MerLin draws the mesh phases at
    construction), which the runner seeds with ``cfg["seed"]`` beforehand.
    """

    def __init__(
        self,
        readout: str,
        n_photons: int,
        input_state: list[int],
        layers: int,
        dtype: torch.dtype,
        computation_space: str = "unbunched",
    ) -> None:
        super().__init__()
        if computation_space not in COMPUTATION_SPACES:
            raise ValueError(
                f"computation_space must be one of {sorted(COMPUTATION_SPACES)}, got {computation_space!r}"
            )
        if readout not in READOUTS:
            raise ValueError(f"readout must be one of {READOUTS}, got {readout!r}")
        if n_photons < 1 or layers < 1:
            raise ValueError(
                f"n_photons and layers must be >= 1, got {n_photons}, {layers}"
            )
        self.readout = readout
        self.n_photons = n_photons
        self.layers = layers
        self.input_state = validate_input_state(input_state, N_MODES, n_photons)
        self.real_dtype = dtype
        self.computation_space = computation_space
        self.quantum = ml.QuantumLayer(
            input_size=N_FEATURES * layers,
            builder=_build_circuit(N_MODES, layers),
            input_state=self.input_state,
            n_photons=n_photons,
            measurement_strategy=ml.MeasurementStrategy.probs(
                computation_space=COMPUTATION_SPACES[computation_space]
            ),
            dtype=dtype,
        )
        occupations = torch.tensor(self.quantum.output_keys, dtype=dtype)
        if occupations.shape[1] != N_MODES or not torch.all(
            occupations.sum(dim=1) == n_photons
        ):
            raise ValueError(
                "Unexpected output keys: every outcome must hold exactly n_photons photons"
            )
        self.register_buffer("occupations", occupations)
        self.register_buffer("vertices", torch.as_tensor(HEXAGON_VERTICES, dtype=dtype))
        if readout == "unbiased":
            self.readout_logits = nn.Parameter(
                0.1 * torch.randn(N_MODES, 3, dtype=dtype)
            )

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        """Repeat z(x) once per re-uploading layer: shape ``(batch, 6 L)``."""
        if features.ndim != 2 or features.shape[1] != N_FEATURES:
            raise ValueError(
                f"Expected features of shape (batch, 6), got {tuple(features.shape)}"
            )
        z = fourier_variables(features)
        return z.repeat(1, self.layers)

    def outcome_responses(self) -> torch.Tensor:
        """Response vectors r(n) for every outcome, shape ``(n_outcomes, 3)``."""
        fractions = self.occupations / self.n_photons
        if self.readout == "vertex":
            return fractions @ self.vertices
        if self.readout == "group":
            group_fractions = fractions.reshape(-1, 3, 2).sum(dim=2)
            return 0.25 + 0.75 * group_fractions
        return fractions @ torch.sigmoid(self.readout_logits)

    def outcome_probabilities(self, features: torch.Tensor) -> torch.Tensor:
        """Outcome distribution P(n | x) (postselected for ``unbunched``), ``(batch, n_outcomes)``."""
        return self.quantum(self.encode(features))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """E[Y_k | x] = 2 P_k(+1 | x) - 1, shape ``(batch, 3)``."""
        win = self.outcome_probabilities(features) @ self.outcome_responses()
        return 2.0 * win - 1.0

    @torch.no_grad()
    def postselection_success(self, features: torch.Tensor) -> torch.Tensor:
        """Probability that an ideal shot is kept, per sample.

        For ``unbunched`` this is the collision-free probability, computed
        with a FOCK-space twin that shares the circuit parameters. For
        ``fock`` no outcome is discarded, so the success rate is 1.
        """
        if self.computation_space == "fock":
            return torch.ones(
                features.shape[0], dtype=self.real_dtype, device=features.device
            )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            twin = ml.QuantumLayer(
                input_size=N_FEATURES * self.layers,
                builder=_build_circuit(N_MODES, self.layers),
                input_state=self.input_state,
                n_photons=self.n_photons,
                measurement_strategy=ml.MeasurementStrategy.probs(
                    computation_space=ml.ComputationSpace.FOCK
                ),
                dtype=self.real_dtype,
            )
        twin.load_state_dict(self.quantum.state_dict())
        probabilities = twin(self.encode(features))
        collision_free = torch.tensor(
            [max(key) <= 1 for key in twin.output_keys], device=probabilities.device
        )
        return probabilities[:, collision_free].sum(dim=1)
