"""Focused unit tests for preprocessing and trace-scaling protocols."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


process = load_module("qve_process_for_protocol_tests", "qve/process.py")
core = load_module("qve_core_for_protocol_tests", "qve/core.py")


TRAIN_FEATURES = np.array([[-1.0], [0.0], [1.0]])
HELD_OUT_WITHIN_TRAIN_RANGE = np.array([[0.5]])
HELD_OUT_EXTREME = np.array([[10.0]])


def test_legacy_minmax_held_out_extrema_change_transformed_training_data():
    train_with_inner_value, _ = process.data_prepare_cv(
        1,
        TRAIN_FEATURES,
        HELD_OUT_WITHIN_TRAIN_RANGE,
        fix_leakage=False,
    )
    train_with_extreme, _ = process.data_prepare_cv(
        1,
        TRAIN_FEATURES,
        HELD_OUT_EXTREME,
        fix_leakage=False,
    )

    assert not np.allclose(train_with_inner_value, train_with_extreme)


def test_train_only_minmax_is_independent_of_held_out_extrema():
    train_with_inner_value, _ = process.data_prepare_cv(
        1,
        TRAIN_FEATURES,
        HELD_OUT_WITHIN_TRAIN_RANGE,
        fix_leakage=True,
    )
    train_with_extreme, _ = process.data_prepare_cv(
        1,
        TRAIN_FEATURES,
        HELD_OUT_EXTREME,
        fix_leakage=True,
    )

    np.testing.assert_allclose(train_with_inner_value, train_with_extreme)


def test_train_only_minmax_does_not_clip_held_out_values():
    _, held_out = process.data_prepare_cv(
        1,
        TRAIN_FEATURES,
        HELD_OUT_EXTREME,
        fix_leakage=True,
    )

    assert np.any((held_out < -1.0) | (held_out > 1.0))


def test_train_only_full_svd_training_transform_is_held_out_independent():
    train_for_validation, _ = process.data_prepare_cv(
        1,
        TRAIN_FEATURES,
        HELD_OUT_WITHIN_TRAIN_RANGE,
        fix_leakage=True,
        svd_solver="full",
    )
    train_for_test, _ = process.data_prepare_cv(
        1,
        TRAIN_FEATURES,
        HELD_OUT_EXTREME,
        fix_leakage=True,
        svd_solver="full",
    )

    np.testing.assert_allclose(train_for_validation, train_for_test)


def test_legacy_trace_normalization_skips_rectangular_matrices():
    square = np.array([[2.0, 1.0], [1.0, 2.0]])
    rectangular = np.array([[4.0, 8.0], [12.0, 16.0], [20.0, 24.0]])

    normalized_square = core.normalize_kernel_trace(square)
    normalized_rectangular = core.normalize_kernel_trace(rectangular)

    np.testing.assert_allclose(normalized_square, square / np.trace(square))
    np.testing.assert_array_equal(normalized_rectangular, rectangular)


def test_train_and_cross_normalization_uses_training_trace_for_both():
    train = np.array([[2.0, 1.0], [1.0, 2.0]])
    cross = np.array([[4.0, 8.0], [12.0, 16.0], [20.0, 24.0]])
    train_trace = np.trace(train)

    normalized_train, normalized_cross = core.normalize_train_and_cross_kernel_trace(
        train, cross
    )

    np.testing.assert_allclose(normalized_train, train / train_trace)
    np.testing.assert_allclose(normalized_cross, cross / train_trace)


@pytest.mark.parametrize(
    ("train", "cross"),
    [
        (np.ones(3), np.ones((1, 3))),
        (np.ones((2, 3)), np.ones((1, 2))),
        (np.eye(2), np.ones(2)),
        (np.eye(2), np.ones((1, 3))),
    ],
)
def test_train_and_cross_normalization_rejects_invalid_dimensions(train, cross):
    with pytest.raises(ValueError):
        core.normalize_train_and_cross_kernel_trace(train, cross)


@pytest.mark.parametrize(
    "train",
    [
        np.zeros((2, 2)),
        -np.eye(2),
        np.array([[np.nan, 0.0], [0.0, 1.0]]),
        np.array([[np.inf, 0.0], [0.0, 1.0]]),
    ],
)
def test_train_and_cross_normalization_rejects_invalid_trace(train):
    with pytest.raises(ValueError):
        core.normalize_train_and_cross_kernel_trace(train, np.ones((1, 2)))


@pytest.fixture(scope="module")
def qsvm_script():
    return load_module(
        "qsvm_script_for_protocol_tests",
        "scripts/qsvm_cuda_embeddings_insurance.py",
    )


def test_pure_qsvm_legacy_trace_preserves_rectangular_cross_kernel(qsvm_script):
    train = np.array([[2.0, 1.0], [1.0, 2.0]])
    cross = np.array([[4.0, 8.0]])

    normalized_train, normalized_cross = qsvm_script.apply_hybrid_kernel(
        train,
        cross,
        np.zeros((2, 1)),
        np.zeros((1, 1)),
        normalize_method="trace",
        trace_protocol="legacy_square_only",
    )

    np.testing.assert_allclose(normalized_train, train / np.trace(train))
    np.testing.assert_array_equal(normalized_cross, cross)


def test_pure_qsvm_train_trace_scales_train_and_cross(qsvm_script):
    train = np.array([[2.0, 1.0], [1.0, 2.0]])
    cross = np.array([[4.0, 8.0]])

    normalized_train, normalized_cross = qsvm_script.apply_hybrid_kernel(
        train,
        cross,
        np.zeros((2, 1)),
        np.zeros((1, 1)),
        normalize_method="trace",
        trace_protocol="train_trace",
    )

    np.testing.assert_allclose(normalized_train, train / np.trace(train))
    np.testing.assert_allclose(normalized_cross, cross / np.trace(train))


def test_non_trace_normalization_is_independent_of_trace_protocol(qsvm_script):
    train = np.array([[2.0, 1.0], [1.0, 2.0]])
    cross = np.array([[4.0, 8.0]])

    legacy = qsvm_script.apply_hybrid_kernel(
        train,
        cross,
        np.zeros((2, 1)),
        np.zeros((1, 1)),
        normalize_method="frobenius",
        trace_protocol="legacy_square_only",
    )
    corrected = qsvm_script.apply_hybrid_kernel(
        train,
        cross,
        np.zeros((2, 1)),
        np.zeros((1, 1)),
        normalize_method="frobenius",
        trace_protocol="train_trace",
    )

    np.testing.assert_allclose(legacy[0], corrected[0])
    np.testing.assert_allclose(legacy[1], corrected[1])


def test_hybrid_legacy_trace_only_scales_square_kernels(qsvm_script):
    quantum_train = np.array([[2.0, 1.0], [1.0, 2.0]])
    quantum_cross = np.array([[4.0, 8.0]])
    data_train = np.array([[0.0], [1.0]])
    data_cross = np.array([[2.0]])
    alpha = 0.25

    hybrid_train, hybrid_cross = qsvm_script.apply_hybrid_kernel(
        quantum_train,
        quantum_cross,
        data_train,
        data_cross,
        use_hybrid=True,
        alpha=alpha,
        classical_kernel="linear",
        normalize_method="trace",
        trace_protocol="legacy_square_only",
    )

    classical_train = np.array([[0.0, 0.0], [0.0, 1.0]])
    classical_cross = np.array([[0.0, 2.0]])
    expected_train = alpha * quantum_train / np.trace(quantum_train) + (
        1 - alpha
    ) * classical_train / np.trace(classical_train)
    expected_cross = alpha * quantum_cross + (1 - alpha) * classical_cross
    np.testing.assert_allclose(hybrid_train, expected_train)
    np.testing.assert_allclose(hybrid_cross, expected_cross)


def test_hybrid_train_trace_scales_each_kernel_pair_before_mixing(qsvm_script):
    quantum_train = np.array([[2.0, 1.0], [1.0, 2.0]])
    quantum_cross = np.array([[4.0, 8.0]])
    data_train = np.array([[0.0], [1.0]])
    data_cross = np.array([[2.0]])
    alpha = 0.25

    hybrid_train, hybrid_cross = qsvm_script.apply_hybrid_kernel(
        quantum_train,
        quantum_cross,
        data_train,
        data_cross,
        use_hybrid=True,
        alpha=alpha,
        classical_kernel="linear",
        normalize_method="trace",
        trace_protocol="train_trace",
    )

    classical_train = np.array([[0.0, 0.0], [0.0, 1.0]])
    classical_cross = np.array([[0.0, 2.0]])
    expected_train = alpha * quantum_train / np.trace(quantum_train) + (
        1 - alpha
    ) * classical_train / np.trace(classical_train)
    expected_cross = alpha * quantum_cross / np.trace(quantum_train) + (
        1 - alpha
    ) * classical_cross / np.trace(classical_train)
    np.testing.assert_allclose(hybrid_train, expected_train)
    np.testing.assert_allclose(hybrid_cross, expected_cross)


def test_merlin_train_trace_uses_each_matching_training_trace():
    merlin_script = load_module(
        "merlin_script_for_protocol_tests",
        "scripts/merlin_fidelity_kernel.py",
    )
    train_validation = np.array([[2.0, 1.0], [1.0, 2.0]])
    validation_cross = np.array([[4.0, 8.0]])
    train_test = np.array([[3.0, 1.0], [1.0, 3.0]])
    test_cross = np.array([[12.0, 18.0]])

    normalized = merlin_script.normalize_kernel_pairs(
        train_validation,
        validation_cross,
        train_test,
        test_cross,
        "train_trace",
    )

    np.testing.assert_allclose(
        normalized[0], train_validation / np.trace(train_validation)
    )
    np.testing.assert_allclose(
        normalized[1], validation_cross / np.trace(train_validation)
    )
    np.testing.assert_allclose(normalized[2], train_test / np.trace(train_test))
    np.testing.assert_allclose(normalized[3], test_cross / np.trace(train_test))


def test_qsvm_cpu_train_trace_smoke(qsvm_script):
    random = np.random.RandomState(7)
    features = random.normal(size=(20, 4))
    labels = np.tile([0, 1], 10)
    comm = qsvm_script.MPI.COMM_WORLD

    results = qsvm_script.run_qsvm_splits(
        features,
        labels,
        n_qubits=2,
        seed=3,
        device_id=0,
        comm_mpi=comm,
        rank=0,
        size=1,
        class_names=["0", "1"],
        use_hybrid=False,
        alpha=0.5,
        classical_kernel="rbf",
        normalize_method="trace",
        trace_protocol="train_trace",
        c_values=[1.0],
        backend="cpu",
    )

    assert results["trace_protocol"] == "train_trace"
    assert results["normalize_method"] == "trace"
    assert results["cross_kernel_scaled_with_train_trace"] is True
