from __future__ import annotations

import numpy as np
import pytest
import torch
from common import REPO_ROOT
from power_of_QNN.lib.fisher import (
    effective_dimension,
    empirical_fisher,
    normalize_fisher,
)
from power_of_QNN.lib.fisher_experiments import fisher_for_draw
from power_of_QNN.lib.quantum import QubitQNN

UPSTREAM = REPO_ROOT / "data" / "power_of_QNN" / "sources" / "effective_dimension"
N_PAPER = [
    1000,
    2000,
    8000,
    10000,
    40000,
    60000,
    100000,
    150000,
    200000,
    500000,
    1000000,
]
needs_upstream = pytest.mark.skipif(
    not (UPSTREAM / "effective_dimension").exists(),
    reason="authors' cache absent; recreate it with the command in LOG.md (Available Resources)",
)


def test_fisher_matches_explicit_per_sample_sum():
    model = QubitQNN(2, "zz", 2, 1)
    generator = torch.Generator().manual_seed(3)
    thetas = torch.rand(2, model.d, dtype=torch.float64, generator=generator)
    inputs = torch.randn(3, 2, dtype=torch.float64, generator=generator)
    ours = empirical_fisher(model, thetas, inputs, chunk_size=2)
    for t in range(2):
        expected = torch.zeros(model.d, model.d, dtype=torch.float64)
        for j in range(3):
            theta = thetas[t : t + 1].clone().requires_grad_(True)
            probabilities = model.probabilities(inputs[j : j + 1], theta)[0]
            for k in range(2):
                (gradient,) = torch.autograd.grad(
                    probabilities[k], theta, retain_graph=True
                )
                expected += (
                    torch.outer(gradient[0], gradient[0]) / probabilities[k].detach()
                )
        assert torch.allclose(ours[t], expected / 3)


def test_normalization_sets_mean_trace_to_d():
    fishers = np.stack([np.diag([1.0, 2.0, 3.0]), np.diag([3.0, 2.0, 1.0])])
    f_hat, mean_trace = normalize_fisher(fishers)
    assert mean_trace == pytest.approx(6.0)
    assert np.trace(f_hat, axis1=1, axis2=2).mean() == pytest.approx(3.0)


def test_effective_dimension_approaches_rank_for_large_n():
    # Remark 3.3: d_{gamma,n} -> max rank as n -> infinity.
    f_hat = np.repeat(np.diag([2.0, 1.0, 0.0])[None], 4, axis=0)
    values = effective_dimension(f_hat, [1e6, 1e60, 1e200])
    assert values[-1] == pytest.approx(2.0, abs=0.02)
    assert values[0] > values[-1] - 1.0


@needs_upstream
def test_formula_reproduces_published_fig3a_source_data():
    published = np.loadtxt(
        UPSTREAM
        / "Effective_dimension_plots/plot_fig/source_data/effective_dimension.txt"
    )
    files = [
        "4qubits_9layer_f_hats_dep2.npy",
        "fhat4_[4 4 4 2]_ed.npy",
        "4qubits_9layer_f_hats_pauli.npy",
    ]
    for row, filename in enumerate(files):
        f_hat = np.load(UPSTREAM / "Effective_dimension_plots/data" / filename)
        ours = effective_dimension(f_hat, N_PAPER) / f_hat.shape[1]
        assert np.abs(ours - published[row]).max() < 1e-9, filename


@needs_upstream
@pytest.mark.parametrize(
    "feature_map,reps,filename",
    [
        ("zz", 2, "4qubits_9layer_f_hats_dep2.npy"),
        ("z", 1, "4qubits_9layer_f_hats_pauli.npy"),
    ],
)
def test_quantum_fhat_matches_authors_arrays_elementwise(feature_map, reps, filename):
    # V-00: the authors' draws (RandomState(0), theta ~ U[0,1], then x ~ N(0,1)) through our
    # torch circuits reproduce their Qiskit F_hat.
    model = QubitQNN(4, feature_map, reps, 9)
    fisher_cfg = {
        "num_thetas": 100,
        "num_inputs": 100,
        "theta_low": 0.0,
        "theta_high": 1.0,
        "chunk_size": 100,
        "n_values": N_PAPER,
        "gamma": 1.0,
        "zero_tolerance": 1e-10,
    }
    ours = fisher_for_draw(model, fisher_cfg, 0, torch.device("cpu"))["f_hat"]
    cached = np.load(
        UPSTREAM / "Eigenvalue_distribution_plots/main_figure/data" / filename
    )
    assert np.abs(ours - cached).max() < 1e-9
