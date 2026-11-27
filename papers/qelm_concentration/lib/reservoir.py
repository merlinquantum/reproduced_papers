"""Kicked-Ising brickwall QELM reservoir simulated as a batched statevector.

Paper mapping (arXiv:2603.13005, Sec. "Model design" and Table I):

- ``N`` qubits on a ring, initialised as Bell pairs ``(|00> + |11>)/sqrt(2)``. Paper Fig. 1
  draws them as cups on ``(1, 2), (3, 4), ..., (N-1, 0)``, offset from the first layer's
  blocks ``(0, 1), (2, 3), ...`` (``bell_pairs = "odd"``); ``"even"`` puts them on the
  first layer's pairs (ablation).
- Each layer holds ``N/2`` two-qubit blocks in a brickwall alternating between the pairs
  ``(0, 1), (2, 3), ...`` and ``(1, 2), ..., (N-1, 0)``; layer ``l`` uses the first set when
  ``(l + brickwall_offset)`` is even.
- Every block has its own frozen ``(J, h, b)`` drawn from normal distributions (Table I).
  Default ``block_order = "figure1"`` follows paper Fig. 1 gate by gate (time order):
  ``RZ(h)`` on the block's second qubit, ``RZZ(J)``, ``RX(b)`` on both qubits, ``RZZ(J)``,
  ``RZ(h)`` on the second qubit, with Qiskit conventions ``RZ(t) = exp(-i t Z / 2)``,
  ``RZZ(t) = exp(-i t ZZ / 2)``, ``RX(t) = exp(-i t X / 2)``. The block is therefore
  ``D X D`` with ``D = exp(-i [J Z_a Z_b + h Z_b] / 2)`` and ``X = exp(-i b (X_a + X_b) / 2)``.
  The orders ``ising_then_kick`` / ``kick_then_ising`` (``exp(-i theta P)`` angles, field on
  both qubits) are earlier inferred layouts kept for ablation.
- Layers follow a 1 encoding : 3 dynamics pattern. In encoding blocks the kick is
  data-dependent, ``b = a_in * u[k]``; ``J`` and ``h`` keep their frozen values.

Basis ordering: qubit 0 is the most significant bit of the computational-basis index,
and ``Z|0> = +|0>``. For a ring pair ``(N-1, 0)`` the second qubit is 0. ``angle_scale``
multiplies every angle. The configs use 2.0: each Fig. 1 gate then implements
``exp(-i theta P)`` with ``theta`` the labelled coefficient (unit-time kicked-Ising
evolution), the reading that reproduces the Fig. 2 Table I stars (LOG.md).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch

logger = logging.getLogger(__name__)

REQUIRED_HYPERPARAMETER_KEYS = (
    "input_strength",
    "b_mean",
    "b_std",
    "h_mean",
    "h_std",
    "J_mean",
    "J_std",
    "h_min",
)
REQUIRED_ARCHITECTURE_KEYS = (
    "n_qubits",
    "n_layers",
    "encoding_period",
    "input_shift",
    "angle_scale",
    "brickwall_offset",
    "block_order",
    "bell_pairs",
)
BELL_PAIRINGS = ("odd", "even")
BLOCK_ORDERS = ("figure1", "ising_then_kick", "kick_then_ising")


@dataclass(frozen=True)
class ReservoirHyperparameters:
    """Distribution parameters of the frozen block couplings (paper Table I).

    Parameters
    ----------
    input_strength : float
        ``a_in`` in ``b = a_in * u`` for encoding blocks.
    b_mean, b_std : float
        Normal distribution of the transverse kick of dynamics blocks.
    h_mean, h_std : float
        Normal distribution of the longitudinal field.
    J_mean, J_std : float
        Normal distribution of the ZZ coupling.
    h_min : float
        Lower bound enforced on ``h`` to stay away from the integrable ``h = 0`` limit.
    """

    input_strength: float
    b_mean: float
    b_std: float
    h_mean: float
    h_std: float
    J_mean: float
    J_std: float
    h_min: float

    @classmethod
    def from_config(cls, hyper_cfg: dict) -> ReservoirHyperparameters:
        """Build hyperparameters from a config mapping, requiring every key.

        Raises
        ------
        KeyError
            If a required key is missing.
        """
        missing = [key for key in REQUIRED_HYPERPARAMETER_KEYS if key not in hyper_cfg]
        if missing:
            raise KeyError(f"Missing reservoir hyperparameter keys: {missing}")
        return cls(
            **{key: float(hyper_cfg[key]) for key in REQUIRED_HYPERPARAMETER_KEYS}
        )


@dataclass(frozen=True)
class ReservoirArchitecture:
    """Circuit layout of the QELM.

    Parameters
    ----------
    n_qubits : int
        Even number of qubits on the ring.
    n_layers : int
        Total number of brickwall layers (encoding and dynamics).
    encoding_period : int
        Layer ``l`` encodes data when ``l % encoding_period == 0``; 4 gives the paper's
        1 encoding : 3 dynamics pattern.
    input_shift : int
        Cyclic shift of the input vector between consecutive encoding layers. Block ``m``
        of encoding layer ``k`` receives ``u[(k * input_shift + m) % len(u)]``.
    angle_scale : float
        Multiplier of every gate angle (1.0: the conventions of ``block_order``).
    brickwall_offset : int
        0: layer 0 acts on the Bell pairs ``(0, 1), (2, 3), ...``; 1: on ``(1, 2), ...``.
        Default value is 0.
    block_order : str
        ``"figure1"``: the paper's Fig. 1 block (symmetric, Qiskit half angles, field on
        the second qubit). ``"kick_then_ising"`` / ``"ising_then_kick"``: one
        ``exp(-i b X)`` kick and one ``exp(-i [J ZZ + h (Z + Z)])`` phase per layer in that
        order (ablations). Default value is "figure1".
    bell_pairs : str
        ``"odd"``: initial Bell pairs on ``(1, 2), ..., (N-1, 0)`` as drawn in Fig. 1;
        ``"even"``: on ``(0, 1), (2, 3), ...``. Default value is "odd".
    """

    n_qubits: int
    n_layers: int
    encoding_period: int
    input_shift: int
    angle_scale: float
    brickwall_offset: int = 0
    block_order: str = "figure1"
    bell_pairs: str = "odd"

    def __post_init__(self) -> None:
        if self.n_qubits < 4 or self.n_qubits % 2:
            raise ValueError(f"n_qubits must be even and >= 4, got {self.n_qubits}")
        if self.n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {self.n_layers}")
        if self.encoding_period < 1:
            raise ValueError(
                f"encoding_period must be >= 1, got {self.encoding_period}"
            )
        if self.brickwall_offset not in (0, 1):
            raise ValueError(
                f"brickwall_offset must be 0 or 1, got {self.brickwall_offset}"
            )
        if self.bell_pairs not in BELL_PAIRINGS:
            raise ValueError(
                f"bell_pairs must be one of {BELL_PAIRINGS}, got {self.bell_pairs!r}"
            )
        if self.block_order not in BLOCK_ORDERS:
            raise ValueError(
                f"block_order must be one of {BLOCK_ORDERS}, got {self.block_order!r}"
            )

    @classmethod
    def from_config(cls, arch_cfg: dict) -> ReservoirArchitecture:
        """Build the architecture from a config mapping, requiring every key."""
        missing = [key for key in REQUIRED_ARCHITECTURE_KEYS if key not in arch_cfg]
        if missing:
            raise KeyError(f"Missing reservoir architecture keys: {missing}")
        return cls(
            n_qubits=int(arch_cfg["n_qubits"]),
            n_layers=int(arch_cfg["n_layers"]),
            encoding_period=int(arch_cfg["encoding_period"]),
            input_shift=int(arch_cfg["input_shift"]),
            angle_scale=float(arch_cfg["angle_scale"]),
            brickwall_offset=int(arch_cfg["brickwall_offset"]),
            block_order=str(arch_cfg["block_order"]),
            bell_pairs=str(arch_cfg["bell_pairs"]),
        )

    @property
    def n_blocks(self) -> int:
        """Number of two-qubit blocks per layer (``N/2``)."""
        return self.n_qubits // 2

    @property
    def n_encoding_layers(self) -> int:
        """Number of data-encoding layers."""
        return len(range(0, self.n_layers, self.encoding_period))

    def is_encoding_layer(self, layer: int) -> bool:
        """Return whether ``layer`` injects input data."""
        return layer % self.encoding_period == 0

    def layer_pairs(self, layer: int) -> list[tuple[int, int]]:
        """Return the qubit pairs of the blocks of ``layer`` (brickwall on a ring)."""
        n = self.n_qubits
        offset = (layer + self.brickwall_offset) % 2
        return [((2 * m + offset) % n, (2 * m + 1 + offset) % n) for m in range(n // 2)]

    def input_indices(self, encoding_layer: int, input_dim: int) -> np.ndarray:
        """Indices of the input vector fed to the blocks of one encoding layer."""
        blocks = np.arange(self.n_blocks)
        return (encoding_layer * self.input_shift + blocks) % input_dim

    def ring_neighbour_pairs(self) -> list[tuple[int, int]]:
        """All nearest-neighbour pairs ``(i, i+1 mod N)`` of the ring."""
        n = self.n_qubits
        return [(i, (i + 1) % n) for i in range(n)]

    def disjoint_pairs(self) -> list[tuple[int, int]]:
        """Non-overlapping pairs ``(0, 1), (2, 3), ...`` (the subsets S2 of the paper)."""
        return [(2 * m, 2 * m + 1) for m in range(self.n_blocks)]


@dataclass(frozen=True)
class BlockParameters:
    """Frozen ``(J, h, b)`` of every block, each of shape ``(n_layers, n_blocks)``.

    ``b`` of encoding layers is unused: those kicks are ``a_in * u``.
    """

    J: np.ndarray
    h: np.ndarray
    b: np.ndarray


def sample_block_parameters(
    hyper: ReservoirHyperparameters,
    arch: ReservoirArchitecture,
    rng: np.random.Generator,
) -> BlockParameters:
    """Draw one reservoir realisation.

    Standard normals for ``J``, ``h`` and ``b`` are drawn first in a fixed order, so two
    hyperparameter settings evaluated with the same seed share the same underlying
    draws (common random numbers for the Pareto search). Fields with ``h <= h_min`` are
    redrawn from a generator derived from ``rng`` until valid.

    Parameters
    ----------
    hyper : ReservoirHyperparameters
        Distribution parameters.
    arch : ReservoirArchitecture
        Circuit layout (gives the array shapes).
    rng : numpy.random.Generator
        Source of randomness.

    Returns
    -------
    BlockParameters
        Frozen block parameters.
    """
    shape = (arch.n_layers, arch.n_blocks)
    z_J = rng.standard_normal(shape)
    z_h = rng.standard_normal(shape)
    z_b = rng.standard_normal(shape)
    J = hyper.J_mean + hyper.J_std * z_J
    h = hyper.h_mean + hyper.h_std * z_h
    b = hyper.b_mean + hyper.b_std * z_b

    if hyper.h_mean <= hyper.h_min and (
        hyper.h_std == 0 or (hyper.h_min - hyper.h_mean) / hyper.h_std > 3
    ):
        raise ValueError(
            f"h ~ N({hyper.h_mean}, {hyper.h_std}) rarely satisfies h > h_min = {hyper.h_min}; "
            "keep h_mean above h_min in the config or search space"
        )
    redraw_rng = np.random.default_rng(rng.integers(2**63))
    invalid = h <= hyper.h_min
    while invalid.any():
        h[invalid] = hyper.h_mean + hyper.h_std * redraw_rng.standard_normal(
            invalid.sum()
        )
        invalid = h <= hyper.h_min
    return BlockParameters(J=J, h=h, b=b)


def complex_dtype_for(real_dtype: torch.dtype) -> torch.dtype:
    """Map the configured real dtype to the statevector complex dtype."""
    if real_dtype == torch.float64:
        return torch.complex128
    if real_dtype == torch.float32:
        return torch.complex64
    raise ValueError(f"Unsupported dtype for statevector simulation: {real_dtype}")


class KickedIsingReservoir:
    """Batched statevector simulator of one frozen QELM realisation.

    Parameters
    ----------
    arch : ReservoirArchitecture
        Circuit layout.
    hyper : ReservoirHyperparameters
        Distribution parameters (only ``input_strength`` is used after sampling).
    params : BlockParameters
        Frozen block parameters.
    device : str or torch.device
        Torch device for the statevector.
    real_dtype : torch.dtype
        ``torch.float32`` (complex64 states) or ``torch.float64`` (complex128 states).
    """

    def __init__(
        self,
        arch: ReservoirArchitecture,
        hyper: ReservoirHyperparameters,
        params: BlockParameters,
        device: str | torch.device,
        real_dtype: torch.dtype,
    ) -> None:
        self.arch = arch
        self.hyper = hyper
        self.params = params
        self.device = torch.device(device)
        self.real_dtype = real_dtype
        self.complex_dtype = complex_dtype_for(real_dtype)
        self._spins = self._spin_values()

    def _spin_values(self) -> torch.Tensor:
        """``z_q(x) = +1/-1`` for every basis index ``x``; shape ``(N, 2**N)``, int8."""
        n = self.arch.n_qubits
        index = torch.arange(2**n, device=self.device)
        shifts = torch.arange(n - 1, -1, -1, device=self.device)
        bits = (index.unsqueeze(0) >> shifts.unsqueeze(1)) & 1
        return (1 - 2 * bits).to(torch.int8)

    def initial_state(self, batch_size: int) -> torch.Tensor:
        """Bell-pair product state broadcast to ``batch_size`` copies."""
        n = self.arch.n_qubits
        bell = torch.tensor(
            [1.0, 0.0, 0.0, 1.0], dtype=self.complex_dtype, device=self.device
        ) / np.sqrt(2.0)
        state = bell
        for _ in range(self.arch.n_blocks - 1):
            state = torch.kron(state, bell)
        if self.arch.bell_pairs == "odd":
            # Relabel qubit q -> q + 1 (mod N): pairs (0,1),(2,3),... become (1,2),...,(N-1,0).
            source_axis_of_new_qubit = [(q - 1) % n for q in range(n)]
            state = (
                state.reshape((2,) * n).permute(source_axis_of_new_qubit).reshape(-1)
            )
        return state.unsqueeze(0).repeat(batch_size, 1)

    def _layer_phase(self, layer: int) -> torch.Tensor:
        """Diagonal Ising factor of one layer.

        ``figure1``: ``exp(-i sum_blocks [J z_a z_b + h z_b] / 2)`` (``RZ(h) RZZ(J)``, applied
        before and after the kicks). Other orders: ``exp(-i sum_blocks [J z_a z_b + h (z_a + z_b)])``.
        """
        scale = self.arch.angle_scale
        figure1 = self.arch.block_order == "figure1"
        phase = torch.zeros(
            2**self.arch.n_qubits, dtype=self.real_dtype, device=self.device
        )
        for block, (qa, qb) in enumerate(self.arch.layer_pairs(layer)):
            z_a = self._spins[qa].to(self.real_dtype)
            z_b = self._spins[qb].to(self.real_dtype)
            J = float(self.params.J[layer, block])
            h = float(self.params.h[layer, block])
            if figure1:
                phase = phase + 0.5 * (J * z_a * z_b + h * z_b)
            else:
                phase = phase + J * z_a * z_b + h * (z_a + z_b)
        return torch.exp(-1j * scale * phase.to(self.complex_dtype))

    def _apply_x_rotation(
        self, state: torch.Tensor, qubit: int, angle: torch.Tensor
    ) -> torch.Tensor:
        """Apply ``exp(-i angle X)`` on ``qubit``; ``angle`` has shape ``(batch,)`` or ``()``."""
        n = self.arch.n_qubits
        batch = state.shape[0]
        view = state.view(batch, 2**qubit, 2, 2 ** (n - qubit - 1))
        cos = torch.cos(angle).to(self.complex_dtype).reshape(-1, 1, 1)
        minus_i_sin = (-1j * torch.sin(angle).to(self.complex_dtype)).reshape(-1, 1, 1)
        amp0 = view[:, :, 0, :]
        amp1 = view[:, :, 1, :]
        rotated = torch.stack(
            (cos * amp0 + minus_i_sin * amp1, minus_i_sin * amp0 + cos * amp1), dim=2
        )
        return rotated.reshape(batch, 2**n)

    def evolve(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return the output statevectors for a batch of input vectors.

        Parameters
        ----------
        inputs : torch.Tensor
            Shape ``(batch, input_dim)``, real, already scaled to the encoding range
            (``[-1, 1]`` in all paper tasks).

        Returns
        -------
        torch.Tensor
            Complex statevectors of shape ``(batch, 2**N)``.
        """
        if inputs.ndim != 2:
            raise ValueError(
                f"inputs must have shape (batch, input_dim), got {tuple(inputs.shape)}"
            )
        inputs = inputs.to(device=self.device, dtype=self.real_dtype)
        batch, input_dim = inputs.shape
        scale = self.arch.angle_scale
        state = self.initial_state(batch)
        encoding_layer = 0
        figure1 = self.arch.block_order == "figure1"
        phase_before = self.arch.block_order in ("figure1", "ising_then_kick")
        phase_after = self.arch.block_order in ("figure1", "kick_then_ising")
        # RX(b) = exp(-i b X / 2) in the figure1 convention.
        kick_factor = 0.5 if figure1 else 1.0
        for layer in range(self.arch.n_layers):
            # Disjoint blocks commute, so a layer factorises into diagonal Ising factors
            # and the X kicks of all blocks, applied in the configured order.
            layer_phase = self._layer_phase(layer).unsqueeze(0)
            if phase_before:
                state = state * layer_phase
            pairs = self.arch.layer_pairs(layer)
            if self.arch.is_encoding_layer(layer):
                feed = self.arch.input_indices(encoding_layer, input_dim)
                encoding_layer += 1
                kicks = (
                    self.hyper.input_strength
                    * inputs[:, torch.as_tensor(feed, device=self.device)]
                )
                block_angles = [
                    kick_factor * scale * kicks[:, block] for block in range(len(pairs))
                ]
            else:
                block_angles = [
                    torch.tensor(
                        kick_factor * scale * float(self.params.b[layer, block]),
                        dtype=self.real_dtype,
                        device=self.device,
                    )
                    for block in range(len(pairs))
                ]
            for (qa, qb), angle in zip(pairs, block_angles, strict=True):
                state = self._apply_x_rotation(state, qa, angle)
                state = self._apply_x_rotation(state, qb, angle)
            if phase_after:
                state = state * layer_phase
        return state


def pair_reduced_density_matrices(
    states: torch.Tensor, pairs: list[tuple[int, int]], n_qubits: int
) -> torch.Tensor:
    """Two-qubit reduced density matrices of pure states.

    Parameters
    ----------
    states : torch.Tensor
        Statevectors of shape ``(batch, 2**N)``.
    pairs : list of tuple of int
        Ordered qubit pairs ``(a, b)``; the RDM basis is ``|z_a z_b>`` with ``a`` the
        most significant bit.
    n_qubits : int
        ``N``.

    Returns
    -------
    torch.Tensor
        Shape ``(batch, len(pairs), 4, 4)``.
    """
    batch = states.shape[0]
    tensor = states.reshape((batch,) + (2,) * n_qubits)
    rdms = []
    for qa, qb in pairs:
        moved = torch.movedim(tensor, (qa + 1, qb + 1), (1, 2)).reshape(batch, 4, -1)
        rdms.append(torch.einsum("bir,bjr->bij", moved, moved.conj()))
    return torch.stack(rdms, dim=1)


def simulate_pair_rdms(
    reservoir: KickedIsingReservoir,
    inputs: np.ndarray,
    pairs: list[tuple[int, int]],
    chunk_size: int,
) -> np.ndarray:
    """Run the reservoir in chunks and return pair RDMs as complex128 numpy arrays.

    Parameters
    ----------
    reservoir : KickedIsingReservoir
        Frozen realisation.
    inputs : numpy.ndarray
        Shape ``(n_samples, input_dim)``.
    pairs : list of tuple of int
        Pairs whose RDMs are returned.
    chunk_size : int
        Samples simulated at once (bounds memory: ``chunk_size * 2**N`` amplitudes).

    Returns
    -------
    numpy.ndarray
        Shape ``(n_samples, len(pairs), 4, 4)``.
    """
    outputs = []
    n_samples = inputs.shape[0]
    for start in range(0, n_samples, chunk_size):
        chunk = torch.as_tensor(inputs[start : start + chunk_size])
        with torch.no_grad():
            states = reservoir.evolve(chunk)
            rdms = pair_reduced_density_matrices(states, pairs, reservoir.arch.n_qubits)
        outputs.append(rdms.to(torch.complex128).cpu().numpy())
        logger.debug(
            "Simulated samples %d-%d of %d", start, start + len(chunk), n_samples
        )
    return np.concatenate(outputs, axis=0)
