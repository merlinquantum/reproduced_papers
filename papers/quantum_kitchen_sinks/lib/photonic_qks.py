"""Photonic Quantum Kitchen Sinks via MerLin.

Per-episode random photonic circuit (entangling layer → angle encoding →
entangling layer); parameters are frozen (QKS is open-loop).  One shot is
sampled per episode from the output occupation distribution; the resulting
``n_modes``-bit pattern is the QKS feature contribution.

See README and INSIGHTS for the design rationale.
"""

from __future__ import annotations

from collections.abc import Sequence

import merlin as ml
import numpy as np
import torch

from .encoding import EpisodeEncoding, make_episodes

# MerLin parameterises a beam splitter by its reflectivity R = cos^2(theta / 2),
# so a balanced 50:50 splitter is theta = pi/2.  The library default of pi/4 is
# an 85:15 splitter, which caps the interference-fringe visibility of a
# dual-rail MZI at 0.5 instead of 1.0.
BALANCED_BEAMSPLITTER_THETA = np.pi / 2

ARCHITECTURES = ("random_mesh", "dual_rail_mzi", "dual_rail_klm_cnot")

# Post-selected KLM CZ: three beam splitters of reflectivity 1/3 (Ralph et al.),
# two of which pair a logical-|0> rail with a vacuum ancilla.  Success
# probability is exactly 1/9 and independent of the input.
KLM_BEAMSPLITTER_THETA = 2 * np.arccos(1 / np.sqrt(3))

# Mode layout for the two-qubit KLM ansatz:
#   0 = vacuum ancilla     1 = |0>_A   2 = |1>_A
#   3 = |1>_B              4 = |0>_B   5 = vacuum ancilla
# The two logical-|1> rails are adjacent so the central splitter can act on
# them (Perceval requires beam splitters on consecutive modes), and the photons
# enter on the outer rails so that P(|1>) = sin^2(theta/2) on each qubit.
KLM_RAIL_MODES = (1, 2, 3, 4)
KLM_INPUT_STATE = (0, 1, 0, 0, 1, 0)
KLM_ENCODED_MODES = (2, 3)


def _klm_cnot_circuit(theta_names: Sequence[str]) -> pcvl.Circuit:
    """RX(theta_0) (x) RX(theta_1) followed by a post-selected CNOT.

    Each qubit is a Mach-Zehnder (balanced splitter, encoding phase, balanced
    splitter) which is exactly ``RX(theta)`` in dual rail.  The CNOT is
    ``H_B . CZ . H_B`` with the CZ realised by the three 1/3 splitters; the
    construction yields ``Z_A . CNOT``, and the spurious ``Z`` on the control is
    diagonal, so it leaves computational-basis probabilities unchanged.
    """
    import perceval as pcvl

    bs50 = dict(theta=BALANCED_BEAMSPLITTER_THETA)
    circuit = pcvl.Circuit(6)
    circuit.add(1, pcvl.BS.H(**bs50))
    circuit.add(3, pcvl.BS.H(**bs50))
    for mode, name in zip(KLM_ENCODED_MODES, theta_names):
        circuit.add(mode, pcvl.PS(pcvl.P(name)))
    circuit.add(1, pcvl.BS.H(**bs50))
    circuit.add(3, pcvl.BS.H(**bs50))
    circuit.add(3, pcvl.BS.H(**bs50))
    for mode in (0, 2, 4):
        circuit.add(mode, pcvl.BS.Rx(theta=KLM_BEAMSPLITTER_THETA))
    circuit.add(3, pcvl.BS.H(**bs50))
    return circuit


def _validate_input_modes(n_modes: int, input_modes: Sequence[int]) -> list[int]:
    ordered_modes = []
    seen = set()
    for mode in input_modes:
        mode_int = int(mode)
        if mode_int < 0 or mode_int >= n_modes:
            raise ValueError("input_modes entries must lie in [0, n_modes)")
        if mode_int not in seen:
            ordered_modes.append(mode_int)
            seen.add(mode_int)
    return ordered_modes


def _default_input_state(
    n_modes: int,
    n_photons: int,
    input_modes: Sequence[int],
    computation_space: ml.ComputationSpace,
) -> list[int]:
    if n_photons > n_modes:
        raise ValueError("n_photons must be <= n_modes")
    ordered_modes = _validate_input_modes(n_modes, input_modes)
    if computation_space is ml.ComputationSpace.DUAL_RAIL:
        if n_modes % 2 != 0:
            raise ValueError("dual_rail requires an even number of modes")
        if n_photons * 2 != n_modes:
            raise ValueError("dual_rail requires n_photons = n_modes // 2")
        if len(ordered_modes) != n_photons:
            raise ValueError(
                "dual_rail requires exactly one encoded mode per logical qubit pair"
            )
        pair_to_mode = {}
        for mode in ordered_modes:
            pair_idx = mode // 2
            if pair_idx in pair_to_mode:
                raise ValueError(
                    "dual_rail input_modes must select at most one mode in each pair"
                )
            pair_to_mode[pair_idx] = mode
        if len(pair_to_mode) != n_photons:
            raise ValueError(
                "dual_rail input_modes must cover each logical qubit pair exactly once"
            )
        state = [0] * n_modes
        for pair_idx in range(n_photons):
            state[pair_to_mode[pair_idx]] = 1
        return state
    for mode in range(n_modes):
        if mode not in ordered_modes:
            ordered_modes.append(mode)
    state = [0] * n_modes
    for mode in ordered_modes[:n_photons]:
        state[mode] = 1
    return state


class PhotonicQKSFeaturizer:
    def __init__(
        self,
        n_modes: int,
        n_photons: int,
        n_episodes: int,
        sigma: float,
        encoding: str,
        n_layers: int = 1,
        shots_per_episode: int = 1,
        input_modes: Sequence[int] | None = None,
        angle_scale: float = 1.0,
        computation_space: ml.ComputationSpace | str = ml.ComputationSpace.UNBUNCHED,
        architecture: str = "random_mesh",
        mesh_after: bool = False,
    ) -> None:
        self.n_modes = int(n_modes)
        self.n_photons = int(n_photons)
        self.n_episodes = int(n_episodes)
        self.sigma = float(sigma)
        self.encoding = encoding
        self.n_layers = int(n_layers)
        self.shots_per_episode = int(shots_per_episode)
        self.computation_space = ml.ComputationSpace.coerce(computation_space)
        self.input_modes = (
            list(range(0, self.n_modes, 2))
            if input_modes is None
            and self.computation_space is ml.ComputationSpace.DUAL_RAIL
            else list(range(self.n_modes))
            if input_modes is None
            else list(input_modes)
        )
        if architecture == "dual_rail_klm_cnot":
            self.input_modes = list(KLM_ENCODED_MODES)
        self.input_modes = _validate_input_modes(self.n_modes, self.input_modes)
        if architecture not in ARCHITECTURES:
            raise ValueError(f"architecture must be one of {ARCHITECTURES}")
        if (
            architecture == "dual_rail_mzi"
            and self.computation_space is not ml.ComputationSpace.DUAL_RAIL
        ):
            raise ValueError("architecture='dual_rail_mzi' requires DUAL_RAIL")
        if architecture == "dual_rail_klm_cnot":
            # Two logical qubits on four rails plus two vacuum ancillas.  The
            # post-selection is ours, not MerLin's, so we read the unbunched
            # distribution and condition on one photon per rail pair.
            if (self.n_modes, self.n_photons) != (6, 2):
                raise ValueError(
                    "architecture='dual_rail_klm_cnot' requires n_modes=6, n_photons=2"
                )
            self.computation_space = ml.ComputationSpace.UNBUNCHED
        self.architecture = architecture
        self.mesh_after = bool(mesh_after)
        self.angle_scale = float(angle_scale)
        self.input_state = (
            list(KLM_INPUT_STATE)
            if architecture == "dual_rail_klm_cnot"
            else _default_input_state(
                self.n_modes,
                self.n_photons,
                self.input_modes,
                self.computation_space,
            )
        )
        self.episodes: list[EpisodeEncoding] = []
        self._layer_seeds: list[int] = []
        self.input_dim = 0

    def _build_layer(self, seed: int) -> ml.QuantumLayer:
        """Build one episode's frozen photonic circuit.

        ``random_mesh`` (the paper-agnostic default) sandwiches the angle
        encoding between two random meshes.  Because a random mesh is not
        balanced, the interference fringe it produces has low visibility -- the
        click probability barely moves with the input -- which is the dominant
        reason the photonic features carry far less signal than the gate-model
        ones.

        ``dual_rail_mzi`` puts a balanced 50:50 splitter on each rail pair
        either side of the encoding, i.e. a textbook MZI per logical qubit.
        That is *exactly* an ``RX(theta)`` on a dual-rail qubit: the even-rail
        click probability is ``sin^2(theta / 2)`` to numerical precision, so the
        photonic featurizer reproduces the gate-model ansatz rather than
        approximating it.  ``mesh_after`` optionally appends a random mesh to
        recover cross-qubit photonic mixing at the cost of that exactness.
        """
        if self.architecture == "dual_rail_klm_cnot":
            names = [f"px_{i}" for i in range(len(KLM_ENCODED_MODES))]
            layer = ml.QuantumLayer(
                circuit=_klm_cnot_circuit(names),
                input_parameters=["px"],
                input_state=self.input_state,
                n_photons=self.n_photons,
                measurement_strategy=ml.MeasurementStrategy.probs(
                    computation_space=self.computation_space
                ),
            )
            for p in layer.parameters():
                p.requires_grad = False
            return layer
        builder = ml.CircuitBuilder(n_modes=self.n_modes)
        if self.architecture == "dual_rail_mzi":
            pairs = [(2 * j, 2 * j + 1) for j in range(self.n_modes // 2)]
            builder.add_superpositions(
                targets=pairs, theta=BALANCED_BEAMSPLITTER_THETA, trainable=False
            )
            builder.add_angle_encoding(modes=self.input_modes, scale=self.angle_scale)
            builder.add_superpositions(
                targets=pairs, theta=BALANCED_BEAMSPLITTER_THETA, trainable=False
            )
            if self.mesh_after:
                builder.add_entangling_layer()
        else:
            builder.add_entangling_layer()
            builder.add_angle_encoding(modes=self.input_modes, scale=self.angle_scale)
            builder.add_entangling_layer()
        layer = ml.QuantumLayer(
            input_size=len(self.input_modes),
            builder=builder,
            input_state=self.input_state,
            n_photons=self.n_photons,
            measurement_strategy=ml.MeasurementStrategy.probs(
                computation_space=self.computation_space
            ),
        )
        for p in layer.parameters():
            p.requires_grad = False
        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        for p in layer.parameters():
            with torch.no_grad():
                p.copy_(torch.empty_like(p).uniform_(0.0, 2.0 * np.pi, generator=gen))
        return layer

    def fit_episodes(self, input_dim: int, seed: int = 0) -> PhotonicQKSFeaturizer:
        self.input_dim = int(input_dim)
        total_episodes = self.n_episodes * self.n_layers
        self.episodes = make_episodes(
            n_episodes=total_episodes,
            input_dim=input_dim,
            n_gate_params=len(self.input_modes),
            sigma=self.sigma,
            encoding=self.encoding,
            seed=seed,
        )
        self._layer_seeds = [seed + 1000 * (e + 1) for e in range(self.n_episodes)]
        return self

    def _build_outcome_table(self) -> np.ndarray:
        """Map a measurement outcome index to the detector click pattern.

        The table must match the basis MerLin reports probabilities in, which
        depends on the computation space:

        - ``UNBUNCHED``: ``C(n_modes, n_photons)`` outcomes, enumerated as
          lexicographic combinations of occupied modes.
        - ``DUAL_RAIL``: ``2 ** (n_modes // 2)`` outcomes; outcome ``i`` is the
          binary word with logical qubit 0 as the most significant bit, and
          qubit ``j`` places its photon in mode ``2 * j + bit_j``.

        Getting this wrong does not raise on its own: it silently relabels
        outcomes onto the wrong click patterns, which a *linear* classifier
        cannot undo.  ``_sample_outcomes`` therefore also checks the row count
        against the measured distribution.
        """
        if self.architecture == "dual_rail_klm_cnot":
            from itertools import combinations

            basis = list(combinations(range(self.n_modes), self.n_photons))
            keep, rows = [], []
            for index, occupied in enumerate(basis):
                a = [m for m in occupied if m in KLM_RAIL_MODES[:2]]
                b = [m for m in occupied if m in KLM_RAIL_MODES[2:]]
                if len(a) != 1 or len(b) != 1:
                    continue
                keep.append(index)
                row = np.zeros(len(KLM_RAIL_MODES), dtype=np.int8)
                row[KLM_RAIL_MODES.index(a[0])] = 1
                row[KLM_RAIL_MODES.index(b[0])] = 1
                rows.append(row)
            self._postselect_columns = np.asarray(keep, dtype=int)
            return np.stack(rows)
        if self.computation_space is ml.ComputationSpace.DUAL_RAIL:
            n_logical = self.n_modes // 2
            table = np.zeros((2**n_logical, self.n_modes), dtype=np.int8)
            for index in range(2**n_logical):
                for qubit in range(n_logical):
                    bit = (index >> (n_logical - 1 - qubit)) & 1
                    table[index, 2 * qubit + bit] = 1
            return table

        from itertools import combinations

        outcomes = list(combinations(range(self.n_modes), self.n_photons))
        table = np.zeros((len(outcomes), self.n_modes), dtype=np.int8)
        for i, combo in enumerate(outcomes):
            for m in combo:
                table[i, m] = 1
        return table

    def _sample_outcomes(
        self, probs: torch.Tensor, rng: np.random.Generator
    ) -> np.ndarray:
        probs_np = probs.detach().cpu().numpy().astype(np.float64)
        probs_np = np.clip(probs_np, 0.0, None)
        if not hasattr(self, "_outcome_table"):
            self._outcome_table = self._build_outcome_table()
        columns = getattr(self, "_postselect_columns", None)
        if columns is not None:
            # Condition on the heralding pattern (one photon per rail pair, none
            # in the ancillas).  For the KLM gadget the surviving mass is a
            # constant 1/9, so this renormalisation discards no information.
            probs_np = probs_np[:, columns]
        probs_np /= probs_np.sum(axis=1, keepdims=True)
        n = probs_np.shape[0]
        cum = np.cumsum(probs_np, axis=1)
        u = rng.uniform(size=(n, 1))
        outcome_indices = (u > cum).sum(axis=1)
        if not hasattr(self, "_outcome_table"):
            self._outcome_table = self._build_outcome_table()
        if self._outcome_table.shape[0] != probs_np.shape[1]:
            raise RuntimeError(
                f"outcome table has {self._outcome_table.shape[0]} rows but the "
                f"{self.computation_space.value} measurement returned "
                f"{probs_np.shape[1]} outcomes"
            )
        return self._outcome_table[outcome_indices]

    def transform(self, X: np.ndarray, seed: int = 0) -> np.ndarray:
        if not self._layer_seeds:
            raise RuntimeError("Call fit_episodes(...) before transform(...).")
        rng = np.random.default_rng(seed)
        feature_chunks = []
        for e in range(self.n_episodes):
            layer = self._build_layer(self._layer_seeds[e])
            episode_idx = e * self.n_layers + (self.n_layers - 1)
            episode = self.episodes[episode_idx]
            theta_np = X @ episode.omega.T + episode.beta
            theta = torch.from_numpy(theta_np.astype(np.float32, copy=False))
            probs = layer(theta)
            if self.shots_per_episode == 1:
                bits = self._sample_outcomes(probs, rng)
            else:
                width = self._outcome_table.shape[1]
                acc = np.zeros((X.shape[0], width), dtype=np.float32)
                for _ in range(self.shots_per_episode):
                    acc += self._sample_outcomes(probs, rng).astype(np.float32)
                bits = acc / self.shots_per_episode
            feature_chunks.append(np.asarray(bits, dtype=np.float32))
        return np.concatenate(feature_chunks, axis=1)


__all__ = ["PhotonicQKSFeaturizer"]
