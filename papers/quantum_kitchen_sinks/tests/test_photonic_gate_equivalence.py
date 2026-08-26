"""A dual-rail photonic qubit must reproduce the gate-model ansatz exactly.

Any single-qubit gate is implementable deterministically in dual rail, so the
photonic featurizer is not an approximation of the gate model at one qubit --
it should agree to numerical precision.  A balanced 50:50 splitter either side
of the phase encoding is a textbook MZI, whose even-rail click probability is
``sin^2(theta / 2)``: precisely the paper's ``RX(theta)`` ansatz read out in the
computational basis.

The subtlety this pins down is MerLin's beam-splitter convention,
``R = cos^2(theta / 2)``.  The library default ``theta = pi/4`` is an 85:15
splitter and caps fringe visibility at 0.5; a random mesh is worse still.  Both
still *run*, and both silently produce features carrying a fraction of the
signal, so only an equivalence check like this one catches it.
"""

import numpy as np
import pytest
from common import PROJECT_DIR  # noqa: F401  (sets sys.path for `lib`)
from lib.circuits import make_ansatz_probs, qubit_marginals
from lib.encoding import encode_batch, make_episodes
from lib.photonic_qks import BALANCED_BEAMSPLITTER_THETA, PhotonicQKSFeaturizer

INPUT_DIM = 32
N_EPISODES = 6
SIGMA = 0.35


def _inputs(n: int = 40) -> np.ndarray:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(n, INPUT_DIM))
    return (x - x.mean(1, keepdims=True)) / x.std(1, keepdims=True)


def _photonic_even_rail_marginals(n_qubits: int, x: np.ndarray) -> np.ndarray:
    """Even-rail click probability per logical qubit, shot noise removed."""
    import torch

    feat = PhotonicQKSFeaturizer(
        n_modes=2 * n_qubits,
        n_photons=n_qubits,
        n_episodes=N_EPISODES,
        sigma=SIGMA,
        encoding="tile",
        computation_space="DUAL_RAIL",
        architecture="dual_rail_mzi",
    )
    feat.fit_episodes(input_dim=INPUT_DIM, seed=0)
    table = feat._build_outcome_table()
    blocks = []
    for e in range(N_EPISODES):
        layer = feat._build_layer(feat._layer_seeds[e])
        episode = feat.episodes[e]
        theta = x @ episode.omega.T + episode.beta
        probs = layer(torch.from_numpy(theta.astype(np.float32))).detach().numpy()
        probs = np.clip(probs.astype(np.float64), 0.0, None)
        probs /= probs.sum(axis=1, keepdims=True)
        blocks.append((probs @ table)[:, 0::2])
    return np.concatenate(blocks, axis=1)


def _independent_rx_marginals(n_qubits: int, x: np.ndarray) -> np.ndarray:
    episodes = make_episodes(
        n_episodes=N_EPISODES,
        input_dim=INPUT_DIM,
        n_gate_params=n_qubits,
        sigma=SIGMA,
        encoding="tile",
        seed=0,
    )
    angles = encode_batch(x, episodes)
    return np.concatenate(
        [np.sin(angles[e] / 2.0) ** 2 for e in range(N_EPISODES)], axis=1
    )


def test_balanced_beamsplitter_is_fifty_fifty():
    """R = cos^2(theta/2); the MerLin default of pi/4 would be 85:15."""
    assert np.cos(BALANCED_BEAMSPLITTER_THETA / 2.0) ** 2 == pytest.approx(0.5)


@pytest.mark.parametrize("n_qubits", [1, 2])
def test_dual_rail_mzi_matches_the_gate_ansatz(n_qubits):
    x = _inputs()
    photonic = _photonic_even_rail_marginals(n_qubits, x)
    gate = _independent_rx_marginals(n_qubits, x)
    assert photonic.shape == gate.shape
    assert np.abs(photonic - gate).max() < 1e-5, (
        "dual-rail MZI must reproduce RX(theta) exactly, not approximately"
    )


def test_one_qubit_photonic_matches_the_cnot1_ansatz():
    """End to end against the actual gate-model code path used for Fig. 5."""
    x = _inputs()
    photonic = _photonic_even_rail_marginals(1, x)
    episodes = make_episodes(
        n_episodes=N_EPISODES,
        input_dim=INPUT_DIM,
        n_gate_params=1,
        sigma=SIGMA,
        encoding="tile",
        seed=0,
    )
    probs_of = make_ansatz_probs("cnot1", 1)
    angles = encode_batch(x, episodes)
    gate = np.concatenate(
        [qubit_marginals(probs_of(angles[e], 1), 1) for e in range(N_EPISODES)], axis=1
    )
    assert np.abs(photonic - gate).max() < 1e-5


def test_dual_rail_mzi_requires_dual_rail_space():
    with pytest.raises(ValueError, match="DUAL_RAIL"):
        PhotonicQKSFeaturizer(
            n_modes=4,
            n_photons=2,
            n_episodes=1,
            sigma=0.1,
            encoding="tile",
            computation_space="UNBUNCHED",
            architecture="dual_rail_mzi",
        )
