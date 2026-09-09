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


def _klm_marginal_distribution(x: np.ndarray, sigma: float, n_episodes: int):
    """Noiseless post-selected outcome distribution of the KLM two-qubit ansatz."""
    import torch
    from lib.photonic_qks import KLM_RAIL_MODES, PhotonicQKSFeaturizer

    feat = PhotonicQKSFeaturizer(
        n_modes=6,
        n_photons=2,
        n_episodes=n_episodes,
        sigma=sigma,
        encoding="tile",
        architecture="dual_rail_klm_cnot",
    )
    feat.fit_episodes(input_dim=x.shape[1], seed=0)
    table = feat._build_outcome_table()
    # order outcomes as (bit_A, bit_B) with A = rail mode 2, B = rail mode 3
    bits = [
        (int(row[KLM_RAIL_MODES.index(2)]), int(row[KLM_RAIL_MODES.index(3)]))
        for row in table
    ]
    order = sorted(range(len(bits)), key=lambda i: bits[i])
    out, herald = [], []
    for e in range(n_episodes):
        layer = feat._build_layer(feat._layer_seeds[e])
        episode = feat.episodes[e]
        theta = x @ episode.omega.T + episode.beta
        probs = layer(torch.from_numpy(theta.astype(np.float32))).detach().numpy()
        probs = np.clip(probs.astype(np.float64), 0.0, None)
        kept = probs[:, feat._postselect_columns]
        herald.append(kept.sum(axis=1))
        out.append((kept / kept.sum(axis=1, keepdims=True))[:, order])
    return out, np.concatenate(herald), feat


def test_klm_cnot_matches_the_gate_cnot2_ansatz():
    """The post-selected KLM CNOT reproduces Fig. 2(a) exactly, not approximately.

    A dual-rail CNOT is not deterministic -- it needs post-selection, at the
    textbook 1/9 success probability -- but conditioned on the herald it is the
    exact gate. So the two-qubit photonic ansatz must agree with the gate-model
    `cnot2` distribution to numerical precision, and it does.
    """
    x = _inputs()
    photonic, _, feat = _klm_marginal_distribution(x, SIGMA, N_EPISODES)
    probs_of = make_ansatz_probs("cnot2", 2)
    for e, block in enumerate(photonic):
        episode = feat.episodes[e]
        theta = x @ episode.omega.T + episode.beta
        assert np.abs(block - probs_of(theta, 1)).max() < 1e-5


def test_klm_success_probability_is_one_ninth():
    """The KLM gadget must herald at exactly 1/9, independently of the input.

    This is computed from the circuit unitary over the full Fock space, not from
    MerLin's ``UNBUNCHED`` probabilities -- those are already renormalised over
    unbunched outcomes, so the fraction surviving *our* herald is conditional on
    that and is not 1/9.  Conditioning twice on nested sets leaves the final
    distribution correct, which is what the equivalence test above checks.

    An input-dependent success probability would mean the post-selection is
    leaking information about the data rather than merely heralding the gate.
    """
    import perceval as pcvl  # noqa: F401
    from lib.photonic_qks import KLM_RAIL_MODES, _klm_cnot_circuit

    outcomes = [(a, b) for a in KLM_RAIL_MODES[:2] for b in KLM_RAIL_MODES[2:]]
    rates = []
    for theta in [(0.0, 0.0), (0.7, 2.1), (np.pi, 1.3), (2.4, 5.0)]:
        circuit = _klm_cnot_circuit(["px_0", "px_1"])
        for param, value in zip(circuit.get_parameters(), theta):
            param.set_value(value)
        # Perceval's compute_unitary uses the transposed index convention
        # relative to "amplitude from input mode i to output mode k".
        u = np.array(circuit.compute_unitary()).T
        i, j = 1, 4
        amps = [
            u[i, rail_a] * u[j, rail_b] + u[i, rail_b] * u[j, rail_a]
            for rail_a, rail_b in outcomes
        ]
        rates.append(float(np.sum(np.abs(amps) ** 2)))
    assert np.allclose(rates, 1.0 / 9.0, atol=1e-9), rates


def test_klm_requires_six_modes_and_two_photons():
    with pytest.raises(ValueError, match="n_modes=6"):
        PhotonicQKSFeaturizer(
            n_modes=4,
            n_photons=2,
            n_episodes=1,
            sigma=0.1,
            encoding="tile",
            architecture="dual_rail_klm_cnot",
        )


def test_merlin_fock_basis_is_reverse_lexicographic():
    """The threshold click table depends on MerLin's Fock ordering, so pin it.

    Nothing in the API states the ordering, and getting it wrong silently
    permutes the feature columns -- which a linear classifier cannot undo.
    """
    from itertools import product

    import merlin as ml
    import perceval as pcvl
    import torch

    states = sorted(
        (t for t in product(range(3), repeat=4) if sum(t) == 2), reverse=True
    )
    for index, occupation in enumerate(states):
        circuit = pcvl.Circuit(4)
        circuit.add(0, pcvl.PS(pcvl.P("px_0")))
        layer = ml.QuantumLayer(
            circuit=circuit,
            input_parameters=["px"],
            input_state=list(occupation),
            n_photons=2,
            measurement_strategy=ml.MeasurementStrategy.probs(
                computation_space=ml.ComputationSpace.FOCK
            ),
        )
        for parameter in layer.parameters():
            parameter.requires_grad = False
        probs = layer(torch.zeros(1, 1)).detach().numpy()[0]
        assert int(probs.argmax()) == index, (occupation, index, probs.argmax())


def test_threshold_click_table_keeps_bunched_events():
    """Bunched outcomes are click patterns, not failures.

    That is precisely what makes this readout deterministic: nothing is
    discarded, so there is no heralding and no success probability to pay.
    """
    from lib.photonic_qks import _threshold_click_table

    table = _threshold_click_table(4, 2)
    assert table.shape == (10, 4), "10 Fock states for 2 photons in 4 modes"
    counts = table.sum(axis=1)
    assert sorted(set(counts.tolist())) == [1, 2]
    assert (counts == 1).sum() == 4, "one bunched state per mode, one click each"
    assert (counts == 2).sum() == 6, "C(4, 2) unbunched states, two clicks each"


@pytest.mark.parametrize("mixing", ["none", "splitter", "mesh"])
def test_mzi_threshold_is_deterministic(mixing):
    """No post-selection: every outcome is kept, so no column is ever dropped."""
    from lib.photonic_qks import PhotonicQKSFeaturizer

    feat = PhotonicQKSFeaturizer(
        n_modes=4,
        n_photons=2,
        n_episodes=2,
        sigma=0.4,
        encoding="tile",
        architecture="mzi_threshold",
        mixing=mixing,
    )
    feat.fit_episodes(input_dim=INPUT_DIM, seed=0)
    assert feat.computation_space.name == "FOCK"
    features = feat.transform(_inputs(20), seed=0)
    assert not hasattr(feat, "_postselect_columns"), "nothing is post-selected"
    assert features.shape == (20, 2 * 4)
    assert set(np.unique(features).tolist()) <= {0.0, 1.0}


def test_mzi_threshold_requires_the_four_mode_layout():
    """The layout that puts both logical-|1> rails adjacent is 4 modes only."""
    from lib.photonic_qks import PhotonicQKSFeaturizer

    with pytest.raises(ValueError, match="n_modes=4"):
        PhotonicQKSFeaturizer(
            n_modes=6,
            n_photons=3,
            n_episodes=1,
            sigma=0.1,
            encoding="tile",
            architecture="mzi_threshold",
        )
