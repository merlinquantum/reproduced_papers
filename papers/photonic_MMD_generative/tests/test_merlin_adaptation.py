"""Phase 04 MerLin adaptation: equivalence with the JAX reproduction, MerLin circuit/noise semantics, runner."""

from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from common import load_runtime_ready_config
from photonic_MMD_generative.lib import circuits, estimator, exact, kernels, qcbm_merlin

M, N = 6, 2
MODES = [1, 3]


@pytest.fixture(scope="module")
def shared_inputs():
    rng = np.random.default_rng(0)
    params = np.asarray(jax.random.normal(jax.random.PRNGKey(3), (2 * M * M,)))
    K = (rng.random((40, M)) < 0.3).astype(np.float64)
    Z = (2 * rng.integers(0, 2, size=(64, N)) - 1).astype(np.float64)
    X = np.zeros((200, M))
    X[np.arange(200)[:, None], np.argsort(rng.random((200, M)), axis=1)[:, :N]] = 1
    return params, K, Z, X


def test_haar_unitary_identical_in_torch_and_jax(shared_inputs):
    params, *_ = shared_inputs
    U_jax = np.asarray(circuits.haar_unitary(jnp.asarray(params), M))
    U_torch = qcbm_merlin.haar_unitary_torch(torch.as_tensor(params), M).numpy()
    assert np.max(np.abs(U_jax - U_torch)) < 1e-12


@pytest.mark.parametrize("photon_model", ["indistinguishable", "distinguishable"])
def test_torch_estimator_equals_jax_on_identical_inputs(shared_inputs, photon_model):
    params, K, Z, X = shared_inputs
    U = np.asarray(circuits.haar_unitary(jnp.asarray(params), M))
    U_S = U[:, MODES]
    if photon_model == "indistinguishable":
        jax_terms = np.asarray(
            estimator.qcbm_mmd_terms(
                jnp.asarray(U_S), jnp.asarray(K), jnp.asarray(Z), jnp.asarray(X)
            )
        )
    else:
        jax_terms = np.asarray(
            estimator.distinguishable_mmd_terms(
                jnp.asarray(U_S), jnp.asarray(K), jnp.asarray(X)
            )
        )
    torch_terms = qcbm_merlin.mmd_terms_torch(
        torch.as_tensor(U_S),
        torch.as_tensor(K),
        torch.as_tensor(Z),
        torch.as_tensor(X),
        photon_model,
    ).numpy()
    assert np.max(np.abs(jax_terms - torch_terms)) < 1e-12


def test_merlin_mesh_is_unitary_has_identity_point_and_gradients():
    interferometer = qcbm_merlin.MerlinInterferometer(8, "merlin_mesh", "cpu")
    assert interferometer.num_parameters == 8 * 7
    params = interferometer.initial_parameters(
        {"strategy": "identity", "perturbation": 0.0}, jax.random.PRNGKey(0)
    )
    U = interferometer.unitary(params)
    assert float((U.abs() - torch.eye(8)).abs().max()) < 1e-12
    theta = (
        torch.rand(interferometer.num_parameters, dtype=torch.float64) * 6.28
    ).requires_grad_()
    U = interferometer.unitary(theta)
    assert float((U.conj().T @ U - torch.eye(8, dtype=U.dtype)).abs().max()) < 1e-12
    U[:, :2].abs().pow(2)[0].sum().backward()
    assert theta.grad is not None and torch.isfinite(theta.grad).all()


def test_haar_initial_point_matches_jax_backend():
    interferometer = qcbm_merlin.MerlinInterferometer(M, "haar", "cpu")
    key = jax.random.PRNGKey(5)
    from photonic_MMD_generative.lib import initialization

    jax_params, _ = initialization.initial_parameters(
        {"strategy": "identity", "perturbation": 0.01}, key, M, "haar"
    )
    torch_params = interferometer.initial_parameters(
        {"strategy": "identity", "perturbation": 0.01}, key
    )
    assert np.array_equal(np.asarray(jax_params), torch_params.numpy())


def test_merlin_loss_matches_jax_loss_for_same_key(shared_inputs):
    params, _, _, X = shared_inputs
    sampler = kernels.make_operator_sampler({"type": "gaussian", "sigma": 1.0}, M, N)
    jax_loss = estimator.make_mmd_loss(
        circuits.make_unitary_fn(M, "haar"), MODES, sampler, 300, 200
    )
    merlin_loss = qcbm_merlin.make_loss(
        qcbm_merlin.MerlinInterferometer(M, "haar", "cpu"),
        MODES,
        sampler,
        300,
        200,
        "indistinguishable",
        chunk=70,
    )
    key = jax.random.PRNGKey(11)
    a = float(jax_loss(jnp.asarray(params), key, jnp.asarray(X)))
    b = float(merlin_loss(torch.as_tensor(params), key, torch.as_tensor(X)))
    assert abs(a - b) < 1e-12


def test_noise_model_endpoints_match_exact_permanents(shared_inputs):
    # V = 1: indistinguishable photons; V = 0: fully distinguishable (the B5 twin's statistics).
    params, *_ = shared_inputs
    U = np.asarray(circuits.haar_unitary(jnp.asarray(params), M))
    for visibility, distinguishable in ((1.0, False), (0.0, True)):
        merlin = qcbm_merlin.noisy_distribution(U, MODES, visibility, 1.0)
        reference = exact.output_distribution(U, MODES, distinguishable=distinguishable)
        assert max(abs(merlin.get(k, 0.0) - v) for k, v in reference.items()) < 1e-10


def test_detectors_and_loss_postselection(shared_inputs):
    params, *_ = shared_inputs
    U = np.asarray(circuits.haar_unitary(jnp.asarray(params), M))
    lossy = qcbm_merlin.noisy_distribution(U, MODES, 1.0, 0.8)
    pnr, pnr_rate = qcbm_merlin.detect(lossy, N, "pnr")
    thr, thr_rate = qcbm_merlin.detect(lossy, N, "threshold")
    assert pnr_rate == pytest.approx(
        0.8**2, rel=1e-9
    )  # both photons must survive uniform loss
    assert thr_rate <= pnr_rate + 1e-12 and all(max(k) <= 1 for k in thr)
    assert sum(pnr.values()) == pytest.approx(1.0) and sum(
        thr.values()
    ) == pytest.approx(1.0)


def test_exact_vs_samples_mmd_is_unbiased_for_matching_distribution():
    rng = np.random.default_rng(1)
    support = [s for s in exact.fock_states(5, 2) if max(s) <= 1]
    weights = rng.random(len(support))
    weights /= weights.sum()
    distribution = dict(zip(support, weights, strict=True))
    values = []
    for _ in range(20):
        idx = rng.choice(len(support), size=400, p=weights)
        values.append(
            qcbm_merlin.exact_vs_samples_mmd(
                distribution, np.array([support[i] for i in idx]), 1.0
            )
        )
    assert abs(np.mean(values)) < 3 * np.std(values) / np.sqrt(len(values)) + 1e-9


@pytest.mark.parametrize("ansatz", ["haar", "merlin_mesh"])
def test_runner_end_to_end_with_merlin_backend(tmp_path, ansatz):
    from photonic_MMD_generative.lib import runner as paper_runner

    cfg = load_runtime_ready_config()
    cfg["data_root"] = str(tmp_path / "data")
    cfg["model"].update(backend="merlin", ansatz=ansatz)
    cfg["training"]["steps"] = 3
    cfg["evaluation"]["num_model_samples"] = 100
    cfg["evaluation"]["noise_study"] = {
        "indistinguishability": [1.0, 0.0],
        "transmittance": [1.0],
        "detectors": ["pnr", "threshold"],
        "sigma": 1.0,
    }
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    paper_runner.train_and_evaluate(cfg, run_dir)
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["backend"] == "merlin" and metrics["training"]["steps_run"] == 3
    assert metrics["deployable"] is True
    assert len(metrics["noise_study"]) == 4


def test_jax_and_merlin_backends_start_identically(tmp_path):
    # Same seed, same init, same K/Z key chain: the first training estimate must agree.
    from photonic_MMD_generative.lib import runner as paper_runner

    first = {}
    for backend in ("jax", "merlin"):
        cfg = load_runtime_ready_config()
        cfg["data_root"] = str(tmp_path / "data")
        cfg["model"]["backend"] = backend
        cfg["training"]["steps"] = 2
        cfg["evaluation"]["sample_metrics"] = False
        run_dir = tmp_path / backend
        run_dir.mkdir()
        paper_runner.train_and_evaluate(cfg, run_dir)
        first[backend] = float(
            (run_dir / "losses.csv").read_text().splitlines()[1].split(",")[1]
        )
    assert first["jax"] == pytest.approx(first["merlin"], abs=1e-12)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="MerLin Clifford sampler needs CUDA"
)
def test_merlin_clifford_sampler_is_seeded():
    U = np.asarray(
        circuits.haar_unitary(
            jax.random.normal(jax.random.PRNGKey(0), (2 * 16 * 16,)), 16
        )
    )
    a = qcbm_merlin.merlin_clifford_samples(U, [0, 2, 4], 200, 7, "cuda")
    b = qcbm_merlin.merlin_clifford_samples(U, [0, 2, 4], 200, 7, "cuda")
    assert np.array_equal(a, b) and (a.sum(axis=1) == 3).all()
