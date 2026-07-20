"""Focused tests for gate-based qSVM/QKNN models."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PAPER_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PAPER_ROOT))

from lib.encoders import (  # noqa: E402
    PhotonicAmplitudeReservoirEncoder,
    PhotonicFidelityKernelEncoder,
    PhotonicReservoirEncoder,
    PhotonicReservoirStateEncoder,
    encoder_hybrid_features,
    encoder_photonic,
    encoder_photonic_amplitude,
    encoder_photonic_fidelity_kernel,
    encoder_photonic_state_amplitudes,
    encoder_qknn_distances,
    encoder_qknn_state_vectors,
    encoder_qsvm_fidelity_kernel,
    euclidean_distances_from_dot,
    explicit_dot_product,
    fock_output_size,
    minimal_fock_modes,
    nearest_psd_kernel,
    required_qubits,
    state_fidelity_kernel,
    unbunched_output_size,
)
from lib.models import (  # noqa: E402
    GateStateFidelitySVMModel,
    HybridSklearnModel,
    PhotonicKNNFidelityKernelModel,
    PhotonicKNNStateFidelityModel,
    PhotonicSVMFidelityKernelModel,
    PhotonicSVMStateFidelityModel,
    QKNNFidelityModel,
    make_knn,
    make_model,
)


def test_required_qubits_for_angle_amplitude_and_zz():
    assert required_qubits("angle", 5) == 5
    assert required_qubits("amplitude", 5) == 3
    assert required_qubits("zz", 5) == 5


def test_minimal_fock_modes_selects_smallest_basis():
    assert minimal_fock_modes(10, 3) == 3
    assert minimal_fock_modes(11, 3) == 4
    assert unbunched_output_size(5, 3) == 10


def test_encoder_qsvm_fidelity_kernel_is_symmetric_for_training_gram():
    X = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=float)

    gram = encoder_qsvm_fidelity_kernel(
        X,
        X,
        encoding="angle",
        device_name="default.qubit",
        angle_rotation="X",
    )

    assert gram.shape == (2, 2)
    assert np.allclose(gram, gram.T)
    assert np.allclose(np.diag(gram), 1.0)


def test_encoder_qsvm_zz_kernel_is_symmetric_for_training_gram():
    X = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=float)

    gram = encoder_qsvm_fidelity_kernel(
        X,
        X,
        encoding="zz",
        device_name="default.qubit",
        zz_entanglement="full",
    )

    assert gram.shape == (2, 2)
    assert np.allclose(gram, gram.T)
    assert np.allclose(np.diag(gram), 1.0)


def test_encoder_qsvm_parallel_matches_serial_kernel():
    X = np.array([[0.1, 0.2], [0.3, 0.4], [0.7, 0.1]], dtype=float)

    serial = encoder_qsvm_fidelity_kernel(
        X,
        X,
        encoding="angle",
        device_name="default.qubit",
        angle_rotation="X",
    )
    parallel = encoder_qsvm_fidelity_kernel(
        X,
        X,
        encoding="angle",
        device_name="default.qubit",
        angle_rotation="X",
        n_jobs=2,
        parallel_chunk_size=1,
        parallel_min_rows=1,
    )

    assert np.allclose(parallel, serial)


def test_gate_state_fidelity_matches_qsvm_kernel():
    X = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=float)

    kernel = encoder_qsvm_fidelity_kernel(
        X,
        X,
        encoding="angle",
        device_name="default.qubit",
        angle_rotation="X",
    )
    states = encoder_qknn_state_vectors(
        X,
        encoding="angle",
        device_name="default.qubit",
        angle_rotation="X",
    )
    explicit = state_fidelity_kernel(states, states)

    assert np.allclose(explicit, kernel)


def test_hybrid_amplitude_default_readout_returns_pauli_z_features():
    X = np.array([[0.1, 0.2, 0.3], [0.3, 0.2, 0.1]], dtype=float)

    transformed = encoder_hybrid_features(
        X,
        encoding="amplitude",
        device_name="default.qubit",
    )

    assert transformed.shape == (2, 2)


def test_hybrid_amplitude_state_real_readout_pads_to_power_of_two():
    X = np.array([[0.1, 0.2, 0.3], [0.3, 0.2, 0.1]], dtype=float)

    transformed = encoder_hybrid_features(
        X,
        encoding="amplitude",
        device_name="default.qubit",
        readout="state_real",
    )

    assert transformed.shape == (2, 4)


def test_hybrid_zz_default_readout_returns_pauli_z_features():
    X = np.array([[0.1, 0.2], [0.3, 0.2]], dtype=float)

    transformed = encoder_hybrid_features(
        X,
        encoding="zz",
        device_name="default.qubit",
    )

    assert transformed.shape == (2, 2)


def test_hybrid_parallel_matches_serial_features():
    X = np.array([[0.1, 0.2], [0.3, 0.2], [0.7, 0.1]], dtype=float)

    serial = encoder_hybrid_features(
        X,
        encoding="zz",
        device_name="default.qubit",
    )
    parallel = encoder_hybrid_features(
        X,
        encoding="zz",
        device_name="default.qubit",
        n_jobs=2,
        parallel_chunk_size=1,
        parallel_min_rows=1,
    )

    assert np.allclose(parallel, serial)


def test_hybrid_zz_state_real_imag_readout_exposes_state_features():
    X = np.array([[0.1, 0.2], [0.3, 0.2]], dtype=float)

    transformed = encoder_hybrid_features(
        X,
        encoding="zz",
        device_name="default.qubit",
        readout="state_real_imag",
    )

    assert transformed.shape == (2, 8)


def test_make_model_uses_separate_pi_over_two_zz_scale_by_default():
    zz_model = make_model("qsvm_zz", {})
    angle_model = make_model("qsvm_angle", {})
    overridden_zz_model = make_model("qsvm_zz", {"zz_angle_scale": 0.75})

    assert np.isclose(zz_model.angle_scale, np.pi / 2.0)
    assert np.isclose(angle_model.angle_scale, np.pi)
    assert np.isclose(overridden_zz_model.angle_scale, 0.75)


def test_hybrid_model_standardizes_explicit_quantum_features_by_default():
    X = np.array(
        [[0.1, 0.2], [0.5, 0.7], [1.0, 0.4], [1.2, 1.1]],
        dtype=float,
    )
    y = np.array([0, 0, 1, 1])

    model = HybridSklearnModel(
        estimator=make_knn(n_neighbors=1),
        encoding="angle",
        device_name="default.qubit",
        angle_rotation="Y",
    ).fit(X, y)

    assert model.feature_standardizer_ is not None
    assert np.allclose(model.X_train_features_.mean(axis=0), 0.0)
    assert model.metadata()["hybrid_standardize_features"] is True


def test_quantum_knn_predicts_binary_labels():
    X_train = np.array([[0.1, 0.2], [0.8, 0.7], [0.2, 0.1], [0.7, 0.8]])
    y_train = np.array([0, 1, 0, 1])
    X_test = np.array([[0.12, 0.18], [0.75, 0.75]])

    model = QKNNFidelityModel(
        n_neighbors=1,
        encoding="angle",
        device_name="default.qubit",
        angle_rotation="X",
    ).fit(X_train, y_train)

    pred, _ = model.predict_with_scores(X_test)

    assert pred.tolist() == [0, 1]


def test_encoder_qknn_distances_match_state_fidelity():
    left_states = np.array([[1.0, 0.0], [1 / np.sqrt(2), 1 / np.sqrt(2)]])
    right_states = np.array([[1.0, 0.0], [0.0, 1.0]])

    distances = encoder_qknn_distances(left_states, right_states)

    assert np.allclose(distances, [[0.0, 1.0], [0.5, 0.5]])


def test_state_fidelity_kernel_matches_inner_product_norm_square():
    left_states = np.array([[1.0, 0.0], [1 / np.sqrt(2), 1j / np.sqrt(2)]])
    right_states = np.array([[1.0, 0.0], [0.0, 1.0]])

    kernel = state_fidelity_kernel(left_states, right_states)

    assert np.allclose(kernel, [[1.0, 0.0], [0.5, 0.5]])


def test_encoder_qknn_parallel_matches_serial_state_vectors():
    X = np.array([[0.1, 0.2], [0.8, 0.7], [0.2, 0.1]], dtype=float)

    serial = encoder_qknn_state_vectors(
        X,
        encoding="angle",
        device_name="default.qubit",
    )
    parallel = encoder_qknn_state_vectors(
        X,
        encoding="angle",
        device_name="default.qubit",
        n_jobs=2,
        parallel_chunk_size=1,
        parallel_min_rows=1,
    )

    assert np.allclose(parallel, serial)


def test_quantum_knn_exposes_vote_share_probabilities():
    X_train = np.array([[0.1, 0.2], [0.8, 0.7], [0.2, 0.1], [0.7, 0.8]])
    y_train = np.array([0, 1, 0, 1])
    X_test = np.array([[0.12, 0.18], [0.75, 0.75]])

    model = QKNNFidelityModel(
        n_neighbors=1,
        encoding="angle",
        device_name="default.qubit",
        angle_rotation="X",
    ).fit(X_train, y_train)

    _, probabilities = model.predict_with_scores(X_test)

    assert probabilities.shape == (2,)
    assert np.all((0.0 <= probabilities) & (probabilities <= 1.0))


def test_explicit_dot_product_distance_matches_numpy_norm():
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    B = np.array([[1.0, 0.0], [0.0, 1.0]])

    dot = explicit_dot_product(A, B)
    distances = euclidean_distances_from_dot(A, B)

    assert np.allclose(dot, A @ B.T)
    assert np.allclose(distances, np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2))


def test_photonic_reservoir_encoder_returns_fock_probabilities():
    X = np.array([[0.1, 0.2, 0.3], [0.4, 0.1, 0.2]], dtype=float)
    encoder = PhotonicReservoirEncoder(
        n_modes=3,
        n_photons=2,
        seed=5,
        dtype="float64",
        batch_size=1,
    )

    features = encoder.encode(X)

    assert encoder.output_size == fock_output_size(3, 2)
    assert features.shape == (2, 6)
    assert np.allclose(features.sum(axis=1), 1.0, atol=1e-5)


def test_encoder_photonic_returns_features_and_encoder():
    X = np.array([[0.1, 0.2, 0.3]], dtype=float)

    features, encoder = encoder_photonic(
        X,
        n_modes=3,
        n_photons=2,
        seed=5,
        dtype="float64",
        batch_size=1,
    )

    assert encoder.output_size == fock_output_size(3, 2)
    assert features.shape == (1, 6)


def test_encoder_photonic_can_return_unbunched_features():
    X = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=float)

    features, encoder = encoder_photonic(
        X,
        n_modes=4,
        n_photons=2,
        seed=5,
        dtype="float64",
        batch_size=1,
        computation_space="UNBUNCHED",
    )

    assert encoder.output_size == unbunched_output_size(4, 2)
    assert features.shape == (1, 6)
    assert np.all(features >= 0.0)


def test_encoder_photonic_amplitude_pads_to_fock_basis():
    X = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=float)

    features, encoder = encoder_photonic_amplitude(
        X,
        n_modes=None,
        n_photons=3,
        seed=5,
        dtype="float64",
        batch_size=1,
    )

    assert isinstance(encoder, PhotonicAmplitudeReservoirEncoder)
    assert encoder.n_modes == 3
    assert encoder.input_size == fock_output_size(3, 3)
    assert features.shape == (1, 10)
    assert np.allclose(features.sum(axis=1), 1.0, atol=1e-8)


def test_encoder_photonic_state_amplitudes_returns_normalized_states():
    X = np.array([[0.1, 0.2, 0.3]], dtype=float)

    states, encoder = encoder_photonic_state_amplitudes(
        X,
        n_modes=3,
        n_photons=2,
        seed=5,
        dtype="float64",
        batch_size=1,
    )

    assert isinstance(encoder, PhotonicReservoirStateEncoder)
    assert encoder.output_size == fock_output_size(3, 2)
    assert states.shape == (1, 6)
    assert np.iscomplexobj(states)
    assert np.allclose(np.linalg.norm(states, axis=1), 1.0, atol=1e-8)


def test_encoder_photonic_state_amplitudes_can_use_unbunched_space():
    X = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=float)

    states, encoder = encoder_photonic_state_amplitudes(
        X,
        n_modes=4,
        n_photons=2,
        seed=5,
        dtype="float64",
        batch_size=1,
        computation_space="UNBUNCHED",
    )

    assert encoder.output_size == unbunched_output_size(4, 2)
    assert states.shape == (1, 6)
    assert np.iscomplexobj(states)
    assert np.allclose(np.linalg.norm(states, axis=1), 1.0, atol=1e-8)


def test_encoder_photonic_fidelity_kernel_returns_training_gram():
    X = np.array([[0.1, 0.2, 0.3], [0.4, 0.1, 0.2]], dtype=float)

    gram, encoder = encoder_photonic_fidelity_kernel(
        X,
        n_modes=3,
        n_photons=2,
        seed=5,
        dtype="float64",
    )

    assert isinstance(encoder, PhotonicFidelityKernelEncoder)
    assert gram.shape == (2, 2)
    assert np.allclose(gram, gram.T)
    assert np.allclose(np.diag(gram), 1.0)


def test_encoder_photonic_fidelity_parallel_matches_serial_kernel():
    X = np.array([[0.1, 0.2, 0.3], [0.4, 0.1, 0.2]], dtype=float)

    serial, _ = encoder_photonic_fidelity_kernel(
        X,
        n_modes=3,
        n_photons=2,
        seed=5,
        dtype="float64",
        force_psd=False,
    )
    parallel, encoder = encoder_photonic_fidelity_kernel(
        X,
        n_modes=3,
        n_photons=2,
        seed=5,
        dtype="float64",
        force_psd=False,
        n_jobs=2,
        parallel_chunk_size=1,
        parallel_min_rows=1,
    )

    assert encoder.merlin_force_psd is False
    assert np.allclose(parallel, serial, atol=1e-6)


def test_nearest_psd_kernel_returns_unit_diagonal_psd_matrix():
    matrix = np.array([[1.0, 1.2], [1.2, 1.0]], dtype=float)

    projected = nearest_psd_kernel(matrix)

    assert np.allclose(projected, projected.T)
    assert np.allclose(np.diag(projected), 1.0)
    assert np.linalg.eigvalsh(projected).min() >= -1e-10


def test_photonic_fidelity_svm_model_predicts_binary_labels():
    X_train = np.array([[0.1, 0.2, 0.3], [0.4, 0.1, 0.2]])
    y_train = np.array([0, 1])
    X_test = np.array([[0.11, 0.21, 0.31], [0.39, 0.09, 0.19]])

    model = PhotonicSVMFidelityKernelModel(
        n_modes=3,
        n_photons=2,
        seed=5,
        dtype="float64",
    ).fit(X_train, y_train)
    pred, score = model.predict_with_scores(X_test)

    assert pred.shape == (2,)
    assert set(pred.tolist()) <= {0, 1}
    assert score is not None
    assert score.shape == (2,)
    assert model.metadata()["photonic_kernel"] == "fidelity"


def test_gate_state_fidelity_svm_model_predicts_binary_labels():
    X_train = np.array([[0.1, 0.2], [0.8, 0.7], [0.2, 0.1], [0.7, 0.8]])
    y_train = np.array([0, 1, 0, 1])
    X_test = np.array([[0.12, 0.18], [0.75, 0.75]])

    model = GateStateFidelitySVMModel(
        encoding="angle",
        device_name="default.qubit",
        angle_rotation="X",
    ).fit(X_train, y_train)
    pred, score = model.predict_with_scores(X_test)

    assert pred.shape == (2,)
    assert set(pred.tolist()) <= {0, 1}
    assert score is not None
    assert model.metadata()["state_fidelity_kernel"] == "explicit_amplitudes"


def test_photonic_state_svm_model_predicts_binary_labels():
    X_train = np.array([[0.1, 0.2, 0.3], [0.4, 0.1, 0.2]])
    y_train = np.array([0, 1])
    X_test = np.array([[0.11, 0.21, 0.31], [0.39, 0.09, 0.19]])

    model = PhotonicSVMStateFidelityModel(
        n_modes=3,
        n_photons=2,
        seed=5,
        dtype="float64",
    ).fit(X_train, y_train)
    pred, score = model.predict_with_scores(X_test)

    assert pred.shape == (2,)
    assert set(pred.tolist()) <= {0, 1}
    assert score is not None
    assert model.metadata()["state_fidelity_kernel"] == "explicit_amplitudes"


def test_photonic_model_infers_modes_from_prepared_features():
    X_train = np.array([[0.1, 0.2, 0.3], [0.4, 0.1, 0.2]])
    y_train = np.array([0, 1])

    model = PhotonicSVMFidelityKernelModel(
        n_modes=None,
        n_photons=2,
        seed=5,
        dtype="float64",
    ).fit(X_train, y_train)

    assert model.metadata()["photonic_n_modes"] == X_train.shape[1]


def test_photonic_hybrid_amplitude_model_uses_fock_basis_features():
    X_train = np.array(
        [[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1], [0.2, 0.1, 0.3, 0.4]],
        dtype=float,
    )
    y_train = np.array([0, 1, 0])
    X_test = np.array([[0.12, 0.18, 0.31, 0.39]], dtype=float)

    model = make_model(
        "photonic_hybrid_knn_amplitude",
        {
            "photonic_n_photons": 3,
            "encoder_batch_size": 1,
            "photonic_dtype": "float64",
            "photonic_qknn_neighbors": 1,
        },
    ).fit(X_train, y_train)
    pred, score = model.predict_with_scores(X_test)

    assert pred.shape == (1,)
    assert score is not None
    assert model.metadata()["photonic_encoding"] == "amplitude"
    assert model.metadata()["photonic_n_modes"] == 3
    assert model.metadata()["photonic_input_size"] == 10


def test_photonic_hybrid_amplitude_expands_modes_for_unbunched_space():
    X_train = np.array(
        [[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1], [0.2, 0.1, 0.3, 0.4]],
        dtype=float,
    )
    y_train = np.array([0, 1, 0])

    model = make_model(
        "photonic_hybrid_knn_amplitude",
        {
            "photonic_n_photons": 3,
            "encoder_batch_size": 1,
            "photonic_dtype": "float64",
            "photonic_qknn_neighbors": 1,
            "photonic_computation_space": "UNBUNCHED",
        },
    ).fit(X_train, y_train)

    assert model.metadata()["photonic_computation_space"] == "UNBUNCHED"
    assert model.metadata()["photonic_n_modes"] == 4
    assert model.metadata()["photonic_output_size"] == unbunched_output_size(4, 3)


def test_photonic_fidelity_knn_model_predicts_binary_labels():
    X_train = np.array([[0.1, 0.2, 0.3], [0.4, 0.1, 0.2]])
    y_train = np.array([0, 1])
    X_test = np.array([[0.11, 0.21, 0.31], [0.39, 0.09, 0.19]])

    model = PhotonicKNNFidelityKernelModel(
        n_modes=3,
        n_photons=2,
        seed=5,
        dtype="float64",
        n_neighbors=1,
    ).fit(X_train, y_train)
    pred, score = model.predict_with_scores(X_test)

    assert pred.shape == (2,)
    assert set(pred.tolist()) <= {0, 1}
    assert score is not None
    assert score.shape == (2,)
    metadata = model.metadata()
    assert metadata["photonic_kernel"] == "fidelity"
    assert metadata["photonic_distance"] == "1_minus_fidelity"
    assert metadata["photonic_knn_neighbors"] == 1


def test_photonic_state_knn_model_predicts_binary_labels():
    X_train = np.array([[0.1, 0.2, 0.3], [0.4, 0.1, 0.2]])
    y_train = np.array([0, 1])
    X_test = np.array([[0.11, 0.21, 0.31], [0.39, 0.09, 0.19]])

    model = PhotonicKNNStateFidelityModel(
        n_modes=3,
        n_photons=2,
        seed=5,
        dtype="float64",
        n_neighbors=1,
    ).fit(X_train, y_train)
    pred, score = model.predict_with_scores(X_test)

    assert pred.shape == (2,)
    assert set(pred.tolist()) <= {0, 1}
    assert score is not None
    metadata = model.metadata()
    assert metadata["state_fidelity_kernel"] == "explicit_amplitudes"
    assert metadata["photonic_distance"] == "1_minus_fidelity"
    assert metadata["photonic_knn_neighbors"] == 1
