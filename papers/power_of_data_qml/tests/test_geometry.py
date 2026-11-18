"""EXP-T: numerical checks of the paper's geometry identities (claims T1, T2).

* T1: the model complexity ``s_K`` equals ``||w||^2`` of the trained kernel model.
* T2: ``s_C <= g^2 s_Q``, and the engineered labels attain equality (App. F).
* The approximate dimension lies in ``[1, N]`` with the stated extremes (AMB-8).
"""

from __future__ import annotations

import common  # noqa: F401  (sets sys.path)
import numpy as np
import pytest
from lib import geometry
from lib.kernels import normalize_trace


def random_kernel(
    n_points: int, rank: int, seed: int, ridge: float = 0.1
) -> np.ndarray:
    factors = np.random.default_rng(seed).standard_normal((n_points, rank))
    return normalize_trace(factors @ factors.T + ridge * np.eye(n_points))


def unregularized_g(quantum: np.ndarray, classical: np.ndarray) -> float:
    root = geometry.psd_sqrt(normalize_trace(quantum))
    matrix = root @ np.linalg.inv(normalize_trace(classical)) @ root
    return float(np.sqrt(np.linalg.eigvalsh(matrix)[-1]))


def test_effective_dimension_extremes():
    n_points = 40
    vector = np.random.default_rng(0).standard_normal(n_points)
    assert geometry.effective_dimension(np.outer(vector, vector)) == pytest.approx(1.0)
    assert geometry.effective_dimension(np.eye(n_points)) == pytest.approx(n_points)
    middle = geometry.effective_dimension(random_kernel(n_points, 5, 1, ridge=1e-3))
    assert 1.0 < middle < n_points


def test_model_complexity_equals_weight_norm():
    # Explicit feature map Phi; trace-normalize via a global rescaling of Phi.
    rng = np.random.default_rng(2)
    phi = rng.standard_normal((30, 8))
    phi *= np.sqrt(30 / np.trace(phi @ phi.T))
    kernel = phi @ phi.T
    labels = rng.standard_normal(30)
    for lam in (0.01, 0.5):
        weights = phi.T @ np.linalg.solve(kernel + lam * np.eye(30), labels)
        assert geometry.model_complexity(kernel, labels, lam) == pytest.approx(
            weights @ weights, rel=1e-9
        )


def test_geometric_difference_bounds_complexity_ratio():
    quantum = random_kernel(25, 25, 3)
    classical = random_kernel(25, 25, 4)
    g = unregularized_g(quantum, classical)
    spectrum = geometry.ClassicalSpectrum.from_kernel("c", classical)
    g_small_lambda, _ = geometry.geometric_difference(
        geometry.psd_sqrt(quantum), spectrum, 1e-12
    )
    assert g_small_lambda == pytest.approx(g, rel=1e-6)
    rng = np.random.default_rng(5)
    for _ in range(50):
        labels = rng.standard_normal(25)
        s_classical = geometry.model_complexity(classical, labels, 0.0)
        s_quantum = geometry.model_complexity(quantum, labels, 0.0)
        assert s_classical <= g**2 * s_quantum * (1 + 1e-9)


def test_identical_kernels_have_unit_geometric_difference():
    kernel = random_kernel(20, 20, 6)
    assert unregularized_g(kernel, kernel) == pytest.approx(1.0, rel=1e-8)


@pytest.mark.parametrize("lam", [1e-12, 0.05])
def test_engineered_targets_saturate_the_bound(lam):
    quantum = random_kernel(30, 30, 7)
    classical = random_kernel(30, 30, 8)
    constructed = geometry.engineered_labels(
        quantum,
        classical,
        lam,
        np.random.default_rng(9),
        random_label_probability=0.0,
        threshold="sign",
    )
    targets = constructed.real_targets
    # s_Q = v^T v = 1 by construction, and s_C(lam) = g_gen^2 s_Q (App. F).
    s_quantum = geometry.model_complexity(quantum, targets, 0.0)
    s_classical = geometry.model_complexity(classical, targets, lam)
    assert s_quantum == pytest.approx(1.0, rel=1e-6)
    assert s_classical == pytest.approx(constructed.g_gen**2 * s_quantum, rel=1e-6)
    assert np.all(np.abs(constructed.labels) == 1)


def test_random_relabelling_rate():
    quantum = random_kernel(400, 400, 10)
    classical = random_kernel(400, 400, 11)
    constructed = geometry.engineered_labels(
        quantum,
        classical,
        0.01,
        np.random.default_rng(12),
        random_label_probability=0.1,
        threshold="median",
    )
    # A random sign differs from the clean label half the time: about 5% change.
    assert 0.02 < constructed.flipped_fraction < 0.09
    assert abs(np.mean(constructed.labels == 1) - 0.5) < 0.06


def test_minimum_geometric_difference_follows_lambda_rule():
    quantum = random_kernel(40, 40, 13, ridge=0.01)
    spectra = [
        geometry.ClassicalSpectrum.from_kernel(
            f"k{seed}", random_kernel(40, 10, seed, ridge=0.01)
        )
        for seed in (14, 15, 16)
    ]
    result = geometry.minimum_geometric_difference(quantum, spectra)
    sqrt_quantum = geometry.psd_sqrt(quantum)
    for spectrum in spectra:
        selected = result.per_kernel[spectrum.name]
        assert selected is not None and selected["g_tra"] < geometry.G_TRA_THRESHOLD
        larger = [lam for lam in geometry.LAMBDA_GRID if lam > selected["lambda"]]
        for lam in larger:
            assert (
                geometry.geometric_difference(sqrt_quantum, spectrum, lam)[1]
                >= geometry.G_TRA_THRESHOLD
            )
    assert result.g == min(entry["g_gen"] for entry in result.per_kernel.values())


def test_prediction_error_bound_unregularized_limit():
    kernel = random_kernel(30, 30, 17)
    labels = np.random.default_rng(18).standard_normal(30)
    expected = np.sqrt(geometry.model_complexity(kernel, labels, 0.0) / 30)
    assert geometry.prediction_error_bound(kernel, labels, 1e-12) == pytest.approx(
        expected, rel=1e-6
    )


def test_inadmissible_lambda_is_flagged_not_hidden():
    # A rank-one classical kernel cannot fit a full-rank quantum kernel: g_tra stays
    # large for every lambda, so the result must say the threshold was not met (AMB-18).
    quantum = random_kernel(30, 30, 19)
    vector = np.random.default_rng(20).standard_normal(30)
    spectra = [
        geometry.ClassicalSpectrum.from_kernel(
            "rank_one", np.outer(vector, vector) + 1e-9 * np.eye(30)
        )
    ]
    result = geometry.minimum_geometric_difference(quantum, spectra)
    assert result.threshold_met is False
    assert result.per_kernel["rank_one"] is None
    assert result.g_tra >= geometry.G_TRA_THRESHOLD


def test_engineered_labels_do_not_depend_on_eigenvector_sign():
    quantum = random_kernel(50, 50, 21)
    classical = random_kernel(50, 50, 22)
    first = geometry.engineered_labels(
        quantum,
        classical,
        0.01,
        np.random.default_rng(23),
        random_label_probability=0.1,
        threshold="sign",
    )
    vector = np.linalg.eigh(np.eye(50))[1][:, -1]
    assert np.sign(vector[np.argmax(np.abs(vector))]) != 0
    second = geometry.engineered_labels(
        quantum,
        classical,
        0.01,
        np.random.default_rng(23),
        random_label_probability=0.1,
        threshold="sign",
    )
    np.testing.assert_array_equal(first.labels, second.labels)
    assert first.real_targets[np.argmax(np.abs(first.real_targets))] != 0


def test_single_class_engineered_labels_fail_loudly():
    # Entrywise-positive (Gaussian) kernels make the top eigenvector single-signed
    # (Perron-Frobenius), so sign(y') is one class; the construction must refuse
    # rather than return a trivial task (AMB-15), while the median split is balanced.
    points = np.random.default_rng(0).standard_normal((40, 3))
    distances = ((points[:, None, :] - points[None, :, :]) ** 2).sum(-1)
    quantum = np.exp(-0.05 * distances)
    classical = np.exp(-0.5 * distances)
    with pytest.raises(ValueError, match="single class"):
        geometry.engineered_labels(
            quantum,
            classical,
            0.01,
            np.random.default_rng(1),
            random_label_probability=0.1,
            threshold="sign",
        )
    balanced = geometry.engineered_labels(
        quantum,
        classical,
        0.01,
        np.random.default_rng(1),
        random_label_probability=0.0,
        threshold="median",
    )
    assert np.mean(balanced.labels == 1) == pytest.approx(0.5)
