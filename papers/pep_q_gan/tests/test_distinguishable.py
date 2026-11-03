"""A physical negative control for the official tagged-photon sampling path."""

import exqalibur as xq
import numpy as np
import perceval as pc
from pep_q_gan.lib.circuits import fock_moments
from pep_q_gan.lib.distinguishable import sample_distinguishable


def test_tagged_photons_restore_hom_coincidences():
    xq.Config.compute_max_thread_count = 1
    circuit = pc.Circuit(2).add(0, pc.BS.H())
    dist, metadata = sample_distinguishable(circuit, [1, 1], 10000, 20261022)
    assert metadata["input_state"] == "|{0},{1}>"
    assert metadata["global_perf"] == 1
    assert np.all(dist.sum(axis=1) == 2)
    assert abs(np.mean(np.all(dist == [1, 1], axis=1)) - 0.5) < 0.03
    backend = pc.Clifford2017Backend()
    backend.set_circuit(circuit)
    backend.set_input_state(pc.BasicState([1, 1]))
    pc.random_seed(20261022)
    indist = np.asarray([list(s) for s in backend.samples(10000)])
    assert not np.any(np.all(indist == [1, 1], axis=1))
    unitary = np.asarray(circuit.compute_unitary())
    mean, covariance = fock_moments(unitary, [1, 1], "dist")
    np.testing.assert_allclose(mean, [1, 1], atol=1e-12)
    np.testing.assert_allclose(covariance, [[0.5, -0.5], [-0.5, 0.5]], atol=1e-12)
