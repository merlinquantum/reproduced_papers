"""High-level model wrappers for Sodar et al."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .encoders import (
    Encoding_type,
    GateFidelityKernelEncoder,
    GateHybridFeatureEncoder,
    GateStateVectorEncoder,
    HybridReadout_type,
    PhotonicAmplitudeReservoirEncoder,
    PhotonicReservoirEncoder,
    PhotonicReservoirStateEncoder,
    encoder_photonic_fidelity_kernel,
    encoder_qknn_distances,
    euclidean_distances_from_dot,
    minimal_photonic_modes,
    normalize_photonic_computation_space,
    photonic_output_size,
    required_qubits,
    state_fidelity_kernel,
)

MODEL_REGISTRY = (
    "svm_classical",
    "knn_classical",
    "qsvm_angle",
    "qsvm_amplitude",
    "qsvm_zz",
    "qknn_angle",
    "qknn_amplitude",
    "qknn_zz",
    "state_svm_angle",
    "state_svm_amplitude",
    "state_svm_zz",
    "state_knn_angle",
    "state_knn_amplitude",
    "state_knn_zz",
    "hybrid_svm_angle",
    "hybrid_svm_amplitude",
    "hybrid_svm_zz",
    "hybrid_knn_angle",
    "hybrid_knn_amplitude",
    "hybrid_knn_zz",
    "photonic_hybrid_svm_angle",
    "photonic_hybrid_svm_amplitude",
    "photonic_state_svm_angle",
    "photonic_fidelity_svm_angle",
    "photonic_hybrid_knn_angle",
    "photonic_hybrid_knn_amplitude",
    "photonic_state_knn_angle",
    "photonic_fidelity_knn_angle",
)
GATE_BASED_MODELS = tuple(
    model
    for model in MODEL_REGISTRY
    if model.startswith(("qsvm_", "qknn_", "state_", "hybrid_"))
)
PHOTONIC_BASED_MODELS = (
    "photonic_hybrid_svm_angle",
    "photonic_hybrid_svm_amplitude",
    "photonic_state_svm_angle",
    "photonic_fidelity_svm_angle",
    "photonic_hybrid_knn_angle",
    "photonic_hybrid_knn_amplitude",
    "photonic_state_knn_angle",
    "photonic_fidelity_knn_angle",
)


def encoding_from_model(name: str) -> Encoding_type | None:
    """Return the quantum encoding implied by a model name.

    Parameters
    ----------
    name : str
        Model variant name.

    Returns
    -------
    {"angle", "amplitude", "zz"} | None
        Encoding name, or None for purely classical baselines.
    """

    if name.endswith("_angle"):
        return "angle"
    if name.endswith("_amplitude"):
        return "amplitude"
    if name.endswith("_zz"):
        return "zz"
    return None


def make_svc(
    *, kernel: str = "rbf", c_value: float = 1.0, gamma: str | float = "scale"
):
    """Create an sklearn SVC with the configured classical kernel."""

    return SVC(kernel=kernel, C=c_value, gamma=gamma)


def make_knn(
    *,
    n_neighbors: int = 7,
    metric: str = "euclidean",
    weights: str = "distance",
    n_jobs: int | None = None,
):
    """Create an sklearn KNN classifier."""

    return KNeighborsClassifier(
        n_neighbors=n_neighbors,
        metric=metric,
        weights=weights,
        n_jobs=n_jobs,
    )


def positive_class_scores(model, X: np.ndarray) -> np.ndarray | None:
    """Return a continuous score for class ``1`` when the model exposes one."""

    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(X), dtype=float)
        if scores.ndim == 1:
            return scores
        classes = np.asarray(getattr(model, "classes_", []))
        class_index = int(np.flatnonzero(classes == 1)[0]) if 1 in classes else -1
        return scores[:, class_index]
    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(X), dtype=float)
        classes = np.asarray(getattr(model, "classes_", []))
        class_index = int(np.flatnonzero(classes == 1)[0]) if 1 in classes else -1
        return probabilities[:, class_index]
    return None


def _timed_predict_and_score(
    estimator: Any,
    X: np.ndarray,
    metadata: dict[str, Any],
    prefix: str,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Run sklearn prediction and score extraction with stage timings."""

    start = time.perf_counter()
    predictions = estimator.predict(X)
    metadata[f"profile_{prefix}_estimator_predict_s"] = time.perf_counter() - start
    start = time.perf_counter()
    scores = positive_class_scores(estimator, X)
    metadata[f"profile_{prefix}_score_s"] = time.perf_counter() - start
    return predictions, scores


def _class_one_probabilities(
    classes: np.ndarray, probabilities: np.ndarray
) -> np.ndarray:
    class_index = int(np.flatnonzero(classes == 1)[0]) if 1 in classes else -1
    return probabilities[:, class_index]


def _fidelity_distances_from_kernel(
    kernel_matrix: np.ndarray,
    *,
    training: bool = False,
) -> np.ndarray:
    """Convert fidelity similarities into precomputed KNN distances.

    Parameters
    ----------
    kernel_matrix : np.ndarray
        Fidelity matrix with values ideally in ``[0, 1]``.
    training : bool, optional
        Whether the matrix is the square train Gram matrix. Default value is
        False.

    Returns
    -------
    np.ndarray
        Pairwise distances ``1 - fidelity`` clipped to the numerical range
        accepted by sklearn's precomputed-distance KNN.
    """

    distances = 1.0 - np.asarray(kernel_matrix, dtype=np.float64)
    distances = np.clip(distances, 0.0, 1.0)
    if training:
        np.fill_diagonal(distances, 0.0)
    return distances


@dataclass
class ExperimentModel:
    """Base class for qSVM reproduction model wrappers."""

    metadata_: dict[str, Any] = field(default_factory=dict, init=False)
    n_features_: int | None = field(default=None, init=False)
    n_qubits_: int | None = field(default=None, init=False)

    def fit(self, X: np.ndarray, y: np.ndarray) -> ExperimentModel:
        """Fit the model on training data."""

        raise NotImplementedError

    def predict_with_scores(
        self,
        X: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Predict labels and optional class-1 scores for input samples."""

        raise NotImplementedError

    def predict_train_with_scores(self) -> tuple[np.ndarray, np.ndarray | None]:
        """Predict labels and optional scores for the fitted train set."""

        return self.predict_with_scores(self.X_train_)

    def metadata(self) -> dict[str, Any]:
        """Return model metadata captured during fitting."""

        return dict(self.metadata_)

    def n_qubits_for(self, n_features: int) -> int | None:
        """Return the number of qubits expected for input features."""

        return None


@dataclass
class SklearnModel(ExperimentModel):
    """Thin sklearn estimator wrapper with the shared model interface."""

    estimator: Any

    def fit(self, X: np.ndarray, y: np.ndarray) -> SklearnModel:
        """Fit the wrapped sklearn estimator."""

        self.X_train_ = np.asarray(X, dtype=np.float64)
        self.y_train_ = np.asarray(y, dtype=np.int64)
        start = time.perf_counter()
        self.estimator.fit(self.X_train_, self.y_train_)
        estimator_time = time.perf_counter() - start
        self.n_features_ = int(self.X_train_.shape[1])
        self.n_qubits_ = None
        self.metadata_ = {"profile_fit_estimator_s": float(estimator_time)}
        return self

    def predict_with_scores(
        self,
        X: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Predict with the wrapped sklearn estimator."""

        X = np.asarray(X, dtype=np.float64)
        return _timed_predict_and_score(
            self.estimator,
            X,
            self.metadata_,
            "predict",
        )

    def predict_train_with_scores(self) -> tuple[np.ndarray, np.ndarray | None]:
        """Predict labels and optional scores on cached train inputs."""

        return _timed_predict_and_score(
            self.estimator,
            self.X_train_,
            self.metadata_,
            "train_predict",
        )


@dataclass
class GateBasedModel(ExperimentModel):
    """Base class for PennyLane gate-based model wrappers."""

    encoding: Encoding_type = "angle"
    device_name: str = "lightning.qubit"
    angle_rotation: str = "X"
    angle_scale: float | None = None
    zz_reps: int = 1
    zz_entanglement: str = "full"
    zz_alpha: float = 2.0
    n_jobs: int | None = 1
    parallel_chunk_size: int | None = None
    parallel_min_rows: int = 64

    def n_qubits_for(self, n_features: int) -> int | None:
        """Return the number of qubits required by the encoding."""

        return required_qubits(self.encoding, n_features)

    def _scaled_inputs(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if self.angle_scale is None:
            return X
        return X * float(self.angle_scale)

    def _base_metadata(self) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "parallel_n_jobs": int(self.n_jobs or 1),
            "parallel_chunk_size": self.parallel_chunk_size,
            "parallel_min_rows": int(self.parallel_min_rows),
        }
        if self.angle_scale is not None:
            scale_key = (
                "zz_angle_scale" if self.encoding == "zz" else "quantum_angle_scale"
            )
            metadata[scale_key] = float(self.angle_scale)
        if self.encoding == "zz":
            metadata.update(
                {
                    "zz_reps": int(self.zz_reps),
                    "zz_entanglement": self.zz_entanglement,
                    "zz_alpha": float(self.zz_alpha),
                }
            )
        return metadata

    def _zz_kwargs(self) -> dict[str, Any]:
        return {
            "zz_reps": int(self.zz_reps),
            "zz_entanglement": self.zz_entanglement,
            "zz_alpha": float(self.zz_alpha),
        }

    def _parallel_kwargs(self) -> dict[str, Any]:
        return {
            "n_jobs": self.n_jobs,
            "parallel_chunk_size": self.parallel_chunk_size,
            "parallel_min_rows": int(self.parallel_min_rows),
        }


@dataclass
class QSVMFidelityKernelModel(GateBasedModel):
    """QSVM model using a precomputed PennyLane fidelity kernel."""

    c_value: float = 1.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> QSVMFidelityKernelModel:
        """Fit an SVC on the encoded fidelity Gram matrix."""

        self.X_train_original_ = np.asarray(X, dtype=np.float64)
        self.X_train_ = self._scaled_inputs(X)
        self.y_train_ = np.asarray(y, dtype=np.int64)
        self.kernel_encoder_ = GateFidelityKernelEncoder(
            n_features=self.X_train_.shape[1],
            encoding=self.encoding,
            device_name=self.device_name,
            angle_rotation=self.angle_rotation,
            **self._zz_kwargs(),
            **self._parallel_kwargs(),
        )
        self.K_train_ = self.kernel_encoder_.kernel_matrix(
            self.X_train_,
            self.X_train_,
        )
        self.estimator = SVC(kernel="precomputed", C=float(self.c_value))
        self.estimator.fit(self.K_train_, self.y_train_)
        self.n_features_ = int(self.X_train_original_.shape[1])
        self.n_qubits_ = self.n_qubits_for(self.n_features_)
        self.metadata_ = self._base_metadata()
        return self

    def _kernel_to_train(self, X: np.ndarray) -> np.ndarray:
        return self.kernel_encoder_.kernel_matrix(
            self._scaled_inputs(X),
            self.X_train_,
        )

    def predict_with_scores(
        self,
        X: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Predict labels and SVC margins from a test Gram matrix."""

        gram = self._kernel_to_train(X)
        return self.estimator.predict(gram), positive_class_scores(self.estimator, gram)

    def predict_train_with_scores(self) -> tuple[np.ndarray, np.ndarray | None]:
        """Predict labels and margins on the train Gram matrix."""

        return (
            self.estimator.predict(self.K_train_),
            positive_class_scores(self.estimator, self.K_train_),
        )


@dataclass
class GateStateFidelitySVMModel(GateBasedModel):
    """SVM using explicit gate-based state amplitudes as a fidelity kernel."""

    c_value: float = 1.0
    normalize_inputs: bool = False
    entangle: bool = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> GateStateFidelitySVMModel:
        """Encode state vectors once and fit an SVC on their fidelities."""

        self.X_train_original_ = np.asarray(X, dtype=np.float64)
        self.X_train_ = self._scaled_inputs(X)
        self.y_train_ = np.asarray(y, dtype=np.int64)
        profile = {}
        start = time.perf_counter()
        self.state_encoder_ = GateStateVectorEncoder(
            n_features=self.X_train_.shape[1],
            encoding=self.encoding,
            device_name=self.device_name,
            angle_rotation=self.angle_rotation,
            normalize_inputs=self.normalize_inputs,
            entangle=self.entangle,
            **self._zz_kwargs(),
            **self._parallel_kwargs(),
        )
        profile["profile_fit_encoder_setup_s"] = time.perf_counter() - start
        start = time.perf_counter()
        self.train_states_ = self.state_encoder_.encode(self.X_train_)
        profile["profile_fit_encode_s"] = time.perf_counter() - start
        start = time.perf_counter()
        self.K_train_ = state_fidelity_kernel(self.train_states_, self.train_states_)
        profile["profile_fit_similarity_s"] = time.perf_counter() - start
        self.estimator = SVC(kernel="precomputed", C=float(self.c_value))
        start = time.perf_counter()
        self.estimator.fit(self.K_train_, self.y_train_)
        profile["profile_fit_estimator_s"] = time.perf_counter() - start
        self.n_features_ = int(self.X_train_original_.shape[1])
        self.n_qubits_ = self.n_qubits_for(self.n_features_)
        self.metadata_ = self._base_metadata()
        self.metadata_.update(
            {
                "state_fidelity_kernel": "explicit_amplitudes",
                "state_output_size": int(self.train_states_.shape[1]),
                "state_normalize_inputs": bool(self.normalize_inputs),
                "state_entangle": bool(self.entangle),
            }
        )
        self.metadata_.update(profile)
        return self

    def _kernel_to_train(
        self, X: np.ndarray, *, prefix: str | None = None
    ) -> np.ndarray:
        start = time.perf_counter()
        states = self.state_encoder_.encode(self._scaled_inputs(X))
        encode_time = time.perf_counter() - start
        start = time.perf_counter()
        gram = state_fidelity_kernel(states, self.train_states_)
        similarity_time = time.perf_counter() - start
        if prefix is not None:
            self.metadata_[f"profile_{prefix}_encode_s"] = float(encode_time)
            self.metadata_[f"profile_{prefix}_similarity_s"] = float(similarity_time)
        return gram

    def predict_with_scores(
        self,
        X: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Predict labels and SVC margins from explicit state fidelities."""

        gram = self._kernel_to_train(X, prefix="predict")
        return _timed_predict_and_score(
            self.estimator,
            gram,
            self.metadata_,
            "predict",
        )

    def predict_train_with_scores(self) -> tuple[np.ndarray, np.ndarray | None]:
        """Predict labels and margins on the train fidelity matrix."""

        return _timed_predict_and_score(
            self.estimator,
            self.K_train_,
            self.metadata_,
            "train_predict",
        )


@dataclass
class QKNNFidelityModel(GateBasedModel):
    """QKNN model using fidelity distances between encoded state vectors."""

    n_neighbors: int = 5
    normalize_inputs: bool = False
    entangle: bool = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> QKNNFidelityModel:
        """Encode and store training states."""

        self.X_train_original_ = np.asarray(X, dtype=np.float64)
        self.X_train_ = self._scaled_inputs(X)
        self.y_train_ = np.asarray(y, dtype=np.int64)
        self.classes_ = np.unique(self.y_train_)
        profile = {}
        start = time.perf_counter()
        self.state_encoder_ = GateStateVectorEncoder(
            n_features=self.X_train_.shape[1],
            encoding=self.encoding,
            device_name=self.device_name,
            angle_rotation=self.angle_rotation,
            normalize_inputs=self.normalize_inputs,
            entangle=self.entangle,
            **self._zz_kwargs(),
            **self._parallel_kwargs(),
        )
        profile["profile_fit_encoder_setup_s"] = time.perf_counter() - start
        start = time.perf_counter()
        self.train_states_ = self.state_encoder_.encode(self.X_train_)
        profile["profile_fit_encode_s"] = time.perf_counter() - start
        self.n_features_ = int(self.X_train_original_.shape[1])
        self.n_qubits_ = self.n_qubits_for(self.n_features_)
        self.metadata_ = self._base_metadata()
        self.metadata_["qknn_neighbors"] = min(
            int(self.n_neighbors),
            len(self.y_train_),
        )
        self.metadata_.update(profile)
        return self

    def _distances(self, X: np.ndarray, *, prefix: str | None = None) -> np.ndarray:
        start = time.perf_counter()
        states = self.state_encoder_.encode(self._scaled_inputs(X))
        encode_time = time.perf_counter() - start
        start = time.perf_counter()
        distances = encoder_qknn_distances(states, self.train_states_)
        distance_time = time.perf_counter() - start
        if prefix is not None:
            self.metadata_[f"profile_{prefix}_encode_s"] = float(encode_time)
            self.metadata_[f"profile_{prefix}_distance_s"] = float(distance_time)
        return distances

    def _probabilities_from_distances(
        self,
        distances: np.ndarray,
        *,
        prefix: str | None = None,
    ) -> np.ndarray:
        start = time.perf_counter()
        k = min(int(self.n_neighbors), len(self.y_train_))
        probabilities = np.zeros((distances.shape[0], len(self.classes_)), dtype=float)
        for row_index, row in enumerate(distances):
            nearest = np.argsort(row)[:k]
            labels, counts = np.unique(self.y_train_[nearest], return_counts=True)
            for label, count in zip(labels, counts):
                class_index = int(np.flatnonzero(self.classes_ == label)[0])
                probabilities[row_index, class_index] = count / k
        if prefix is not None:
            self.metadata_[f"profile_{prefix}_vote_s"] = float(
                time.perf_counter() - start
            )
        return probabilities

    def predict_with_scores(
        self,
        X: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Predict labels and vote-share scores from fidelity distances."""

        probabilities = self._probabilities_from_distances(
            self._distances(X, prefix="predict"),
            prefix="predict",
        )
        start = time.perf_counter()
        predictions = self.classes_[np.argmax(probabilities, axis=1)]
        self.metadata_["profile_predict_estimator_predict_s"] = (
            time.perf_counter() - start
        )
        start = time.perf_counter()
        scores = _class_one_probabilities(self.classes_, probabilities)
        self.metadata_["profile_predict_score_s"] = time.perf_counter() - start
        return predictions.astype(np.int64), scores

    def predict_train_with_scores(self) -> tuple[np.ndarray, np.ndarray | None]:
        """Predict labels using cached train state vectors."""

        start = time.perf_counter()
        train_distances = encoder_qknn_distances(self.train_states_, self.train_states_)
        self.metadata_["profile_train_predict_distance_s"] = time.perf_counter() - start
        probabilities = self._probabilities_from_distances(
            train_distances,
            prefix="train_predict",
        )
        start = time.perf_counter()
        predictions = self.classes_[np.argmax(probabilities, axis=1)]
        self.metadata_["profile_train_predict_estimator_predict_s"] = (
            time.perf_counter() - start
        )
        start = time.perf_counter()
        scores = _class_one_probabilities(self.classes_, probabilities)
        self.metadata_["profile_train_predict_score_s"] = time.perf_counter() - start
        return predictions.astype(np.int64), scores


@dataclass
class GateStateFidelityKNNModel(QKNNFidelityModel):
    """KNN alias exposing explicit gate-based state-fidelity distances."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> GateStateFidelityKNNModel:
        """Encode state vectors once and store metadata for the explicit path."""

        super().fit(X, y)
        self.metadata_.update(
            {
                "state_fidelity_kernel": "explicit_amplitudes",
                "state_output_size": int(self.train_states_.shape[1]),
                "state_distance": "1_minus_fidelity",
            }
        )
        return self


@dataclass
class HybridSklearnModel(GateBasedModel):
    """Hybrid model that transforms quantum features before sklearn fitting."""

    estimator: Any = None
    hybrid_readout: HybridReadout_type = "pauli_z"
    standardize_features: bool = True

    def fit(self, X: np.ndarray, y: np.ndarray) -> HybridSklearnModel:
        """Encode quantum features and fit the wrapped sklearn estimator."""

        self.X_train_original_ = np.asarray(X, dtype=np.float64)
        self.X_train_ = self._scaled_inputs(X)
        self.y_train_ = np.asarray(y, dtype=np.int64)
        profile = {}
        start = time.perf_counter()
        self.feature_encoder_ = GateHybridFeatureEncoder(
            n_features=self.X_train_.shape[1],
            encoding=self.encoding,
            device_name=self.device_name,
            angle_rotation=self.angle_rotation,
            readout=self.hybrid_readout,
            **self._zz_kwargs(),
            **self._parallel_kwargs(),
        )
        profile["profile_fit_encoder_setup_s"] = time.perf_counter() - start
        start = time.perf_counter()
        raw_features = self.feature_encoder_.encode(self.X_train_)
        profile["profile_fit_encode_s"] = time.perf_counter() - start
        self.feature_standardizer_ = None
        start = time.perf_counter()
        if self.standardize_features:
            self.feature_standardizer_ = StandardScaler()
            self.X_train_features_ = self.feature_standardizer_.fit_transform(
                raw_features
            )
        else:
            self.X_train_features_ = raw_features
        profile["profile_fit_standardize_s"] = time.perf_counter() - start
        start = time.perf_counter()
        self.estimator.fit(self.X_train_features_, self.y_train_)
        profile["profile_fit_estimator_s"] = time.perf_counter() - start
        self.n_features_ = int(self.X_train_original_.shape[1])
        self.n_qubits_ = self.n_qubits_for(self.n_features_)
        self.metadata_ = self._base_metadata()
        self.metadata_["hybrid_output_features"] = int(self.X_train_features_.shape[1])
        self.metadata_["hybrid_readout"] = self.hybrid_readout
        self.metadata_["hybrid_standardize_features"] = bool(self.standardize_features)
        self.metadata_.update(profile)
        return self

    def _features(self, X: np.ndarray, *, prefix: str | None = None) -> np.ndarray:
        start = time.perf_counter()
        raw_features = self.feature_encoder_.encode(self._scaled_inputs(X))
        encode_time = time.perf_counter() - start
        start = time.perf_counter()
        if self.feature_standardizer_ is None:
            features = raw_features
        else:
            features = self.feature_standardizer_.transform(raw_features)
        standardize_time = time.perf_counter() - start
        if prefix is not None:
            self.metadata_[f"profile_{prefix}_encode_s"] = float(encode_time)
            self.metadata_[f"profile_{prefix}_standardize_s"] = float(standardize_time)
        return features

    def predict_with_scores(
        self,
        X: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Predict labels and scores from transformed quantum features."""

        features = self._features(X, prefix="predict")
        return _timed_predict_and_score(
            self.estimator,
            features,
            self.metadata_,
            "predict",
        )

    def predict_train_with_scores(self) -> tuple[np.ndarray, np.ndarray | None]:
        """Predict labels and scores on cached train features."""

        return _timed_predict_and_score(
            self.estimator,
            self.X_train_features_,
            self.metadata_,
            "train_predict",
        )


@dataclass
class PhotonicModel(ExperimentModel):
    """Base class for MerLin photonic reservoir model wrappers."""

    n_modes: int | None = None
    n_photons: int = 3
    seed: int = 42
    device: str = "cpu"
    dtype: str = "float64"
    phase_scale: float = np.pi
    batch_size: int = 32
    photonic_encoding: str = "angle"
    computation_space: str = "FOCK"
    standardize_features: bool = True
    feature_cache: dict[tuple[Any, ...], dict[str, Any]] | None = None
    n_jobs: int | None = 1
    parallel_chunk_size: int | None = None
    parallel_min_rows: int = 64

    def _resolve_n_modes(self, X: np.ndarray) -> int:
        if self.photonic_encoding == "angle":
            n_modes = X.shape[1] if self.n_modes is None else int(self.n_modes)
            if n_modes != X.shape[1]:
                raise ValueError(
                    f"Photonic angle model expects {n_modes} modes, but received "
                    f"{X.shape[1]} prepared features."
                )
            return n_modes
        if self.photonic_encoding == "amplitude":
            n_modes = (
                max(
                    int(self.n_photons),
                    minimal_photonic_modes(
                        X.shape[1],
                        int(self.n_photons),
                        self.computation_space,
                    ),
                )
                if self.n_modes is None
                else int(self.n_modes)
            )
            basis_size = photonic_output_size(
                n_modes,
                int(self.n_photons),
                self.computation_space,
            )
            if basis_size < X.shape[1]:
                raise ValueError(
                    "Photonic amplitude model requires an output basis at least as "
                    f"large as the feature vector; got basis size {basis_size} "
                    f"for {X.shape[1]} features."
                )
            return n_modes
        raise ValueError("Photonic encoding must be either 'angle' or 'amplitude'.")

    def _cache_key(self, X: np.ndarray, n_modes: int) -> tuple[Any, ...]:
        return (
            "photonic_raw",
            self.photonic_encoding,
            id(X),
            tuple(X.shape),
            int(n_modes),
            int(self.n_photons),
            int(self.seed),
            self.device,
            self.dtype,
            float(self.phase_scale),
            int(self.batch_size),
            normalize_photonic_computation_space(self.computation_space),
        )

    def _feature_encoder_key(self, X: np.ndarray, n_modes: int) -> tuple[Any, ...]:
        return (
            self.photonic_encoding,
            int(X.shape[1]),
            int(n_modes),
            int(self.n_photons),
            int(self.seed),
            self.device,
            self.dtype,
            float(self.phase_scale),
            int(self.batch_size),
            normalize_photonic_computation_space(self.computation_space),
        )

    def _ensure_feature_encoder(
        self,
        X: np.ndarray,
        n_modes: int,
    ) -> PhotonicReservoirEncoder | PhotonicAmplitudeReservoirEncoder:
        key = self._feature_encoder_key(X, n_modes)
        if getattr(self, "feature_encoder_key_", None) == key:
            return self.feature_encoder_
        if self.photonic_encoding == "angle":
            encoder = PhotonicReservoirEncoder(
                n_modes=n_modes,
                n_photons=int(self.n_photons),
                seed=int(self.seed),
                device=self.device,
                dtype=self.dtype,
                phase_scale=float(self.phase_scale),
                batch_size=int(self.batch_size),
                computation_space=normalize_photonic_computation_space(
                    self.computation_space
                ),
            )
        elif self.photonic_encoding == "amplitude":
            encoder = PhotonicAmplitudeReservoirEncoder(
                n_features=X.shape[1],
                n_modes=n_modes,
                n_photons=int(self.n_photons),
                seed=int(self.seed),
                device=self.device,
                dtype=self.dtype,
                batch_size=int(self.batch_size),
                computation_space=normalize_photonic_computation_space(
                    self.computation_space
                ),
            )
        else:
            raise ValueError("Photonic encoding must be either 'angle' or 'amplitude'.")
        self.feature_encoder_ = encoder
        self.feature_encoder_key_ = key
        return encoder

    def _feature_metadata(
        self,
        X: np.ndarray,
        n_modes: int,
        encoder: PhotonicReservoirEncoder | PhotonicAmplitudeReservoirEncoder,
        feature_time: float,
    ) -> dict[str, Any]:
        if self.photonic_encoding == "angle":
            input_state = encoder.input_state
            input_size = int(n_modes)
            phase_scale = float(self.phase_scale)
        else:
            input_state = "data_amplitude_statevector"
            input_size = int(encoder.input_size)
            phase_scale = None
        return {
            "photonic_n_modes": n_modes,
            "photonic_n_photons": int(self.n_photons),
            "photonic_input_state": input_state,
            "photonic_computation_space": normalize_photonic_computation_space(
                self.computation_space
            ),
            "photonic_encoding": self.photonic_encoding,
            "photonic_input_size": input_size,
            "photonic_source_features": int(X.shape[1]),
            "photonic_output_size": int(encoder.output_size),
            "photonic_expected_output_size": photonic_output_size(
                n_modes,
                int(self.n_photons),
                self.computation_space,
            ),
            "photonic_phase_scale": phase_scale,
            "feature_time_s": float(feature_time),
        }

    def _raw_features(self, X: np.ndarray) -> tuple[np.ndarray, dict[str, Any], bool]:
        X = np.asarray(X, dtype=np.float64)
        n_modes = self._resolve_n_modes(X)
        key = self._cache_key(X, n_modes)
        if self.feature_cache is not None and key in self.feature_cache:
            self._ensure_feature_encoder(X, n_modes)
            cached = self.feature_cache[key]
            return cached["features"], dict(cached["metadata"]), True
        start = time.perf_counter()
        encoder = self._ensure_feature_encoder(X, n_modes)
        features = encoder.encode(X)
        feature_time = time.perf_counter() - start
        metadata = self._feature_metadata(X, n_modes, encoder, feature_time)
        if self.feature_cache is not None:
            self.feature_cache[key] = {
                "features": features,
                "metadata": dict(metadata),
            }
        return features, metadata, False

    def _fit_photonic_features(self, X: np.ndarray) -> np.ndarray:
        raw_start = time.perf_counter()
        raw_features, metadata, cache_hit = self._raw_features(X)
        metadata["profile_fit_encode_s"] = float(time.perf_counter() - raw_start)
        self.feature_standardizer_ = None
        start = time.perf_counter()
        if self.standardize_features:
            self.feature_standardizer_ = StandardScaler()
            features = self.feature_standardizer_.fit_transform(raw_features)
        else:
            features = raw_features
        metadata["profile_fit_standardize_s"] = float(time.perf_counter() - start)
        metadata["photonic_standardize_features"] = bool(self.standardize_features)
        metadata["feature_cache_hit"] = cache_hit
        self.metadata_ = metadata
        self.X_train_features_ = features
        self.n_features_ = int(features.shape[1])
        self.n_qubits_ = None
        return features

    def _transform_photonic_features(self, X: np.ndarray) -> np.ndarray:
        raw_start = time.perf_counter()
        raw_features, _, _ = self._raw_features(X)
        self.metadata_["profile_predict_encode_s"] = float(
            time.perf_counter() - raw_start
        )
        start = time.perf_counter()
        if self.feature_standardizer_ is None:
            features = raw_features
        else:
            features = self.feature_standardizer_.transform(raw_features)
        self.metadata_["profile_predict_standardize_s"] = float(
            time.perf_counter() - start
        )
        return features

    def _state_cache_key(self, X: np.ndarray, n_modes: int) -> tuple[Any, ...]:
        return (
            "photonic_state_amplitudes",
            id(X),
            tuple(X.shape),
            int(n_modes),
            int(self.n_photons),
            int(self.seed),
            self.device,
            self.dtype,
            float(self.phase_scale),
            int(self.batch_size),
            normalize_photonic_computation_space(self.computation_space),
        )

    def _state_encoder_key(self, X: np.ndarray, n_modes: int) -> tuple[Any, ...]:
        return (
            int(X.shape[1]),
            int(n_modes),
            int(self.n_photons),
            int(self.seed),
            self.device,
            self.dtype,
            float(self.phase_scale),
            int(self.batch_size),
            normalize_photonic_computation_space(self.computation_space),
        )

    def _ensure_state_encoder(
        self,
        X: np.ndarray,
        n_modes: int,
    ) -> PhotonicReservoirStateEncoder:
        key = self._state_encoder_key(X, n_modes)
        if getattr(self, "state_encoder_key_", None) == key:
            return self.state_encoder_
        encoder = PhotonicReservoirStateEncoder(
            n_modes=n_modes,
            n_photons=int(self.n_photons),
            seed=int(self.seed),
            device=self.device,
            dtype=self.dtype,
            phase_scale=float(self.phase_scale),
            batch_size=int(self.batch_size),
            computation_space=normalize_photonic_computation_space(
                self.computation_space
            ),
        )
        self.state_encoder_ = encoder
        self.state_encoder_key_ = key
        return encoder

    def _state_metadata(
        self,
        X: np.ndarray,
        n_modes: int,
        encoder: PhotonicReservoirStateEncoder,
        state_time: float,
    ) -> dict[str, Any]:
        return {
            "photonic_n_modes": n_modes,
            "photonic_n_photons": int(self.n_photons),
            "photonic_input_state": encoder.input_state,
            "photonic_computation_space": normalize_photonic_computation_space(
                self.computation_space
            ),
            "photonic_encoding": "angle",
            "photonic_kernel": "state_fidelity",
            "photonic_state_readout": "amplitudes",
            "photonic_input_size": int(n_modes),
            "photonic_source_features": int(X.shape[1]),
            "photonic_output_size": int(encoder.output_size),
            "photonic_expected_output_size": photonic_output_size(
                n_modes,
                int(self.n_photons),
                self.computation_space,
            ),
            "photonic_phase_scale": float(self.phase_scale),
            "feature_time_s": float(state_time),
        }

    def _state_amplitudes(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, dict[str, Any], bool]:
        X = np.asarray(X, dtype=np.float64)
        if self.photonic_encoding != "angle":
            raise ValueError(
                "Photonic state-fidelity models currently use angle inputs."
            )
        n_modes = self._resolve_n_modes(X)
        key = self._state_cache_key(X, n_modes)
        if self.feature_cache is not None and key in self.feature_cache:
            self._ensure_state_encoder(X, n_modes)
            cached = self.feature_cache[key]
            return cached["states"], dict(cached["metadata"]), True
        start = time.perf_counter()
        encoder = self._ensure_state_encoder(X, n_modes)
        states = encoder.encode(X)
        state_time = time.perf_counter() - start
        metadata = self._state_metadata(X, n_modes, encoder, state_time)
        if self.feature_cache is not None:
            self.feature_cache[key] = {
                "states": states,
                "metadata": dict(metadata),
            }
        return states, metadata, False


@dataclass
class PhotonicSVMHybridModel(PhotonicModel):
    """Photonic reservoir SVM using explicit features and an sklearn kernel."""

    c_value: float = 1.0
    kernel: str = "rbf"
    gamma: str | float = "scale"

    def fit(self, X: np.ndarray, y: np.ndarray) -> PhotonicSVMHybridModel:
        """Fit an SVC on explicit photonic reservoir features."""

        self.y_train_ = np.asarray(y, dtype=np.int64)
        train_features = self._fit_photonic_features(X)
        self.estimator = make_svc(
            kernel=self.kernel,
            c_value=float(self.c_value),
            gamma=self.gamma,
        )
        start = time.perf_counter()
        self.estimator.fit(train_features, self.y_train_)
        self.metadata_["profile_fit_estimator_s"] = float(time.perf_counter() - start)
        self.metadata_["photonic_hybrid_svm_kernel"] = self.kernel
        self.metadata_["photonic_hybrid_svm_gamma"] = self.gamma
        return self

    def predict_with_scores(
        self,
        X: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Predict labels and scores from explicit photonic features."""

        features = self._transform_photonic_features(X)
        return _timed_predict_and_score(
            self.estimator,
            features,
            self.metadata_,
            "predict",
        )

    def predict_train_with_scores(self) -> tuple[np.ndarray, np.ndarray | None]:
        """Predict labels and scores on the train features."""

        return _timed_predict_and_score(
            self.estimator,
            self.X_train_features_,
            self.metadata_,
            "train_predict",
        )


@dataclass
class PhotonicSVMFidelityKernelModel(PhotonicModel):
    """Photonic reservoir SVM using MerLin's fidelity kernel."""

    c_value: float = 1.0
    shots: int = 0
    sampling_method: str = "multinomial"
    force_psd: bool = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> PhotonicSVMFidelityKernelModel:
        """Fit an SVC on a MerLin photonic fidelity Gram matrix."""

        self.X_train_ = np.asarray(X, dtype=np.float64)
        self.y_train_ = np.asarray(y, dtype=np.int64)
        n_modes = self._resolve_n_modes(self.X_train_)
        start = time.perf_counter()
        self.K_train_, self.kernel_encoder_ = encoder_photonic_fidelity_kernel(
            self.X_train_,
            None,
            n_modes=n_modes,
            n_photons=int(self.n_photons),
            seed=int(self.seed),
            device=self.device,
            dtype=self.dtype,
            phase_scale=float(self.phase_scale),
            shots=int(self.shots),
            sampling_method=self.sampling_method,
            force_psd=bool(self.force_psd),
            computation_space=normalize_photonic_computation_space(
                self.computation_space
            ),
            n_jobs=self.n_jobs,
            parallel_chunk_size=self.parallel_chunk_size,
            parallel_min_rows=int(self.parallel_min_rows),
        )
        kernel_time = time.perf_counter() - start
        self.estimator = SVC(kernel="precomputed", C=float(self.c_value))
        self.estimator.fit(self.K_train_, self.y_train_)
        self.n_features_ = int(self.X_train_.shape[1])
        self.n_qubits_ = None
        self.metadata_ = {
            "photonic_n_modes": n_modes,
            "photonic_n_photons": int(self.n_photons),
            "photonic_input_state": self.kernel_encoder_.input_state,
            "photonic_computation_space": normalize_photonic_computation_space(
                self.computation_space
            ),
            "photonic_encoding": "angle",
            "photonic_kernel": "fidelity",
            "photonic_phase_scale": float(self.phase_scale),
            "photonic_fidelity_shots": int(self.shots),
            "photonic_fidelity_sampling_method": self.sampling_method,
            "photonic_fidelity_force_psd": bool(self.force_psd),
            "photonic_fidelity_merlin_force_psd": bool(
                self.kernel_encoder_.merlin_force_psd
            ),
            "parallel_n_jobs": int(self.kernel_encoder_.n_jobs or 1),
            "parallel_chunk_size": self.parallel_chunk_size,
            "parallel_min_rows": int(self.parallel_min_rows),
            "photonic_fidelity_kernel_time_s": float(kernel_time),
            "photonic_empty_postselection_shim": bool(
                self.kernel_encoder_.cleared_empty_postselection
            ),
        }
        return self

    def predict_with_scores(
        self,
        X: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Predict from MerLin photonic fidelity similarities to train data."""

        gram = self.kernel_encoder_.kernel_matrix(X, self.X_train_)
        return self.estimator.predict(gram), positive_class_scores(self.estimator, gram)

    def predict_train_with_scores(self) -> tuple[np.ndarray, np.ndarray | None]:
        """Predict labels and margins on the train fidelity Gram matrix."""

        return (
            self.estimator.predict(self.K_train_),
            positive_class_scores(self.estimator, self.K_train_),
        )


@dataclass
class PhotonicSVMStateFidelityModel(PhotonicModel):
    """Photonic reservoir SVM using explicit Fock amplitudes."""

    c_value: float = 1.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> PhotonicSVMStateFidelityModel:
        """Fit an SVC on explicit photonic state fidelities."""

        self.X_train_ = np.asarray(X, dtype=np.float64)
        self.y_train_ = np.asarray(y, dtype=np.int64)
        start = time.perf_counter()
        self.train_states_, metadata, cache_hit = self._state_amplitudes(self.X_train_)
        metadata["profile_fit_encode_s"] = float(time.perf_counter() - start)
        start = time.perf_counter()
        self.K_train_ = state_fidelity_kernel(self.train_states_, self.train_states_)
        metadata["profile_fit_similarity_s"] = float(time.perf_counter() - start)
        self.estimator = SVC(kernel="precomputed", C=float(self.c_value))
        start = time.perf_counter()
        self.estimator.fit(self.K_train_, self.y_train_)
        metadata["profile_fit_estimator_s"] = float(time.perf_counter() - start)
        self.n_features_ = int(self.X_train_.shape[1])
        self.n_qubits_ = None
        metadata.update(
            {
                "state_fidelity_kernel": "explicit_amplitudes",
                "feature_cache_hit": cache_hit,
            }
        )
        self.metadata_ = metadata
        return self

    def _kernel_to_train(
        self, X: np.ndarray, *, prefix: str | None = None
    ) -> np.ndarray:
        start = time.perf_counter()
        states, _, _ = self._state_amplitudes(X)
        encode_time = time.perf_counter() - start
        start = time.perf_counter()
        gram = state_fidelity_kernel(states, self.train_states_)
        similarity_time = time.perf_counter() - start
        if prefix is not None:
            self.metadata_[f"profile_{prefix}_encode_s"] = float(encode_time)
            self.metadata_[f"profile_{prefix}_similarity_s"] = float(similarity_time)
        return gram

    def predict_with_scores(
        self,
        X: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Predict from explicit photonic state fidelities to train data."""

        gram = self._kernel_to_train(X, prefix="predict")
        return _timed_predict_and_score(
            self.estimator,
            gram,
            self.metadata_,
            "predict",
        )

    def predict_train_with_scores(self) -> tuple[np.ndarray, np.ndarray | None]:
        """Predict labels and margins on the train fidelity matrix."""

        return _timed_predict_and_score(
            self.estimator,
            self.K_train_,
            self.metadata_,
            "train_predict",
        )


@dataclass
class PhotonicKNNHybridModel(PhotonicModel):
    """Photonic reservoir KNN using explicit Euclidean distances."""

    n_neighbors: int = 5
    weights: str = "distance"

    def fit(self, X: np.ndarray, y: np.ndarray) -> PhotonicKNNHybridModel:
        """Fit a precomputed-distance KNN on photonic reservoir features."""

        self.y_train_ = np.asarray(y, dtype=np.int64)
        train_features = self._fit_photonic_features(X)
        start = time.perf_counter()
        self.D_train_ = euclidean_distances_from_dot(train_features, train_features)
        self.metadata_["profile_fit_distance_s"] = float(time.perf_counter() - start)
        n_neighbors = min(int(self.n_neighbors), len(self.y_train_))
        self.estimator = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            metric="precomputed",
            weights=self.weights,
            n_jobs=self.n_jobs,
        )
        start = time.perf_counter()
        self.estimator.fit(self.D_train_, self.y_train_)
        self.metadata_["profile_fit_estimator_s"] = float(time.perf_counter() - start)
        self.metadata_["explicit_distance"] = "euclidean_from_dot_product"
        self.metadata_["photonic_knn_neighbors"] = n_neighbors
        return self

    def predict_with_scores(
        self,
        X: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Predict from photonic Euclidean distances to train features."""

        features = self._transform_photonic_features(X)
        start = time.perf_counter()
        distances = euclidean_distances_from_dot(features, self.X_train_features_)
        self.metadata_["profile_predict_distance_s"] = float(
            time.perf_counter() - start
        )
        return _timed_predict_and_score(
            self.estimator,
            distances,
            self.metadata_,
            "predict",
        )

    def predict_train_with_scores(self) -> tuple[np.ndarray, np.ndarray | None]:
        """Predict labels and probabilities on the train distance matrix."""

        return _timed_predict_and_score(
            self.estimator,
            self.D_train_,
            self.metadata_,
            "train_predict",
        )


@dataclass
class PhotonicKNNFidelityKernelModel(PhotonicModel):
    """Photonic reservoir KNN using MerLin fidelity-kernel distances."""

    n_neighbors: int = 5
    weights: str = "distance"
    shots: int = 0
    sampling_method: str = "multinomial"
    force_psd: bool = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> PhotonicKNNFidelityKernelModel:
        """Fit a precomputed-distance KNN on MerLin fidelity similarities."""

        self.X_train_ = np.asarray(X, dtype=np.float64)
        self.y_train_ = np.asarray(y, dtype=np.int64)
        n_modes = self._resolve_n_modes(self.X_train_)
        start = time.perf_counter()
        self.K_train_, self.kernel_encoder_ = encoder_photonic_fidelity_kernel(
            self.X_train_,
            None,
            n_modes=n_modes,
            n_photons=int(self.n_photons),
            seed=int(self.seed),
            device=self.device,
            dtype=self.dtype,
            phase_scale=float(self.phase_scale),
            shots=int(self.shots),
            sampling_method=self.sampling_method,
            force_psd=bool(self.force_psd),
            computation_space=normalize_photonic_computation_space(
                self.computation_space
            ),
            n_jobs=self.n_jobs,
            parallel_chunk_size=self.parallel_chunk_size,
            parallel_min_rows=int(self.parallel_min_rows),
        )
        kernel_time = time.perf_counter() - start
        self.D_train_ = _fidelity_distances_from_kernel(self.K_train_, training=True)
        n_neighbors = min(int(self.n_neighbors), len(self.y_train_))
        self.estimator = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            metric="precomputed",
            weights=self.weights,
            n_jobs=self.n_jobs,
        )
        self.estimator.fit(self.D_train_, self.y_train_)
        self.n_features_ = int(self.X_train_.shape[1])
        self.n_qubits_ = None
        self.metadata_ = {
            "photonic_n_modes": n_modes,
            "photonic_n_photons": int(self.n_photons),
            "photonic_input_state": self.kernel_encoder_.input_state,
            "photonic_computation_space": normalize_photonic_computation_space(
                self.computation_space
            ),
            "photonic_encoding": "angle",
            "photonic_kernel": "fidelity",
            "photonic_distance": "1_minus_fidelity",
            "photonic_phase_scale": float(self.phase_scale),
            "photonic_fidelity_shots": int(self.shots),
            "photonic_fidelity_sampling_method": self.sampling_method,
            "photonic_fidelity_force_psd": bool(self.force_psd),
            "photonic_fidelity_merlin_force_psd": bool(
                self.kernel_encoder_.merlin_force_psd
            ),
            "parallel_n_jobs": int(self.kernel_encoder_.n_jobs or 1),
            "parallel_chunk_size": self.parallel_chunk_size,
            "parallel_min_rows": int(self.parallel_min_rows),
            "photonic_fidelity_kernel_time_s": float(kernel_time),
            "photonic_empty_postselection_shim": bool(
                self.kernel_encoder_.cleared_empty_postselection
            ),
            "photonic_knn_neighbors": n_neighbors,
        }
        return self

    def predict_with_scores(
        self,
        X: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Predict from MerLin photonic fidelity distances to train data."""

        gram = self.kernel_encoder_.kernel_matrix(X, self.X_train_)
        distances = _fidelity_distances_from_kernel(gram)
        return self.estimator.predict(distances), positive_class_scores(
            self.estimator,
            distances,
        )

    def predict_train_with_scores(self) -> tuple[np.ndarray, np.ndarray | None]:
        """Predict labels and probabilities on the train fidelity distances."""

        return (
            self.estimator.predict(self.D_train_),
            positive_class_scores(self.estimator, self.D_train_),
        )


@dataclass
class PhotonicKNNStateFidelityModel(PhotonicModel):
    """Photonic reservoir KNN using explicit Fock-amplitude fidelities."""

    n_neighbors: int = 5
    weights: str = "distance"

    def fit(self, X: np.ndarray, y: np.ndarray) -> PhotonicKNNStateFidelityModel:
        """Fit a precomputed-distance KNN on explicit photonic fidelities."""

        self.X_train_ = np.asarray(X, dtype=np.float64)
        self.y_train_ = np.asarray(y, dtype=np.int64)
        start = time.perf_counter()
        self.train_states_, metadata, cache_hit = self._state_amplitudes(self.X_train_)
        metadata["profile_fit_encode_s"] = float(time.perf_counter() - start)
        start = time.perf_counter()
        self.K_train_ = state_fidelity_kernel(self.train_states_, self.train_states_)
        metadata["profile_fit_similarity_s"] = float(time.perf_counter() - start)
        start = time.perf_counter()
        self.D_train_ = _fidelity_distances_from_kernel(self.K_train_, training=True)
        metadata["profile_fit_distance_s"] = float(time.perf_counter() - start)
        n_neighbors = min(int(self.n_neighbors), len(self.y_train_))
        self.estimator = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            metric="precomputed",
            weights=self.weights,
            n_jobs=self.n_jobs,
        )
        start = time.perf_counter()
        self.estimator.fit(self.D_train_, self.y_train_)
        metadata["profile_fit_estimator_s"] = float(time.perf_counter() - start)
        self.n_features_ = int(self.X_train_.shape[1])
        self.n_qubits_ = None
        metadata.update(
            {
                "photonic_distance": "1_minus_fidelity",
                "photonic_knn_neighbors": n_neighbors,
                "state_fidelity_kernel": "explicit_amplitudes",
                "feature_cache_hit": cache_hit,
            }
        )
        self.metadata_ = metadata
        return self

    def _distances_to_train(
        self,
        X: np.ndarray,
        *,
        prefix: str | None = None,
    ) -> np.ndarray:
        start = time.perf_counter()
        states, _, _ = self._state_amplitudes(X)
        encode_time = time.perf_counter() - start
        start = time.perf_counter()
        gram = state_fidelity_kernel(states, self.train_states_)
        similarity_time = time.perf_counter() - start
        start = time.perf_counter()
        distances = _fidelity_distances_from_kernel(gram)
        distance_time = time.perf_counter() - start
        if prefix is not None:
            self.metadata_[f"profile_{prefix}_encode_s"] = float(encode_time)
            self.metadata_[f"profile_{prefix}_similarity_s"] = float(similarity_time)
            self.metadata_[f"profile_{prefix}_distance_s"] = float(distance_time)
        return distances

    def predict_with_scores(
        self,
        X: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Predict from explicit photonic fidelity distances to train data."""

        distances = self._distances_to_train(X, prefix="predict")
        return _timed_predict_and_score(
            self.estimator,
            distances,
            self.metadata_,
            "predict",
        )

    def predict_train_with_scores(self) -> tuple[np.ndarray, np.ndarray | None]:
        """Predict labels and probabilities on the train fidelity distances."""

        return _timed_predict_and_score(
            self.estimator,
            self.D_train_,
            self.metadata_,
            "train_predict",
        )


def _angle_scale_for_encoding(
    cfg: dict[str, Any], encoding: Encoding_type
) -> float | None:
    if encoding == "angle":
        return float(cfg.get("quantum_angle_scale", np.pi))
    if encoding == "zz":
        return float(cfg.get("zz_angle_scale", np.pi / 2.0))
    return None


def _zz_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "zz_reps": int(cfg.get("zz_reps", 1)),
        "zz_entanglement": str(cfg.get("zz_entanglement", "full")),
        "zz_alpha": float(cfg.get("zz_alpha", 2.0)),
    }


def _int_config(cfg: dict[str, Any], key: str, default: int) -> int:
    value = cfg.get(key, default)
    return int(default if value is None else value)


def _parallel_kwargs_from_config(cfg: dict[str, Any]) -> dict[str, Any]:
    chunk_size = cfg.get("parallel_chunk_size")
    if chunk_size is None:
        chunk_size = cfg.get("encoder_batch_size")
    return {
        "n_jobs": _int_config(cfg, "parallel_n_jobs", 1),
        "parallel_chunk_size": None if chunk_size is None else int(chunk_size),
        "parallel_min_rows": _int_config(cfg, "parallel_min_rows", 64),
    }


def _encoder_batch_size(cfg: dict[str, Any]) -> int:
    value = cfg.get("encoder_batch_size")
    return int(32 if value is None else value)


def _photonic_n_modes(cfg: dict[str, Any]) -> int | None:
    value = cfg.get("photonic_n_modes")
    return None if value is None else int(value)


def _photonic_n_photons(cfg: dict[str, Any]) -> int:
    return int(cfg.get("photonic_n_photons", 3))


def _photonic_computation_space(cfg: dict[str, Any]) -> str:
    return normalize_photonic_computation_space(
        str(cfg.get("photonic_computation_space", "FOCK"))
    )


def _photonic_common_kwargs(
    cfg: dict[str, Any],
    feature_cache: dict[tuple[Any, ...], dict[str, Any]] | None,
) -> dict[str, Any]:
    return {
        "n_modes": _photonic_n_modes(cfg),
        "n_photons": _photonic_n_photons(cfg),
        "seed": int(cfg.get("photonic_seed", cfg.get("seed", 42))),
        "device": str(cfg.get("photonic_device", cfg.get("device", "cpu"))),
        "dtype": str(cfg.get("photonic_dtype", cfg.get("dtype", "float64"))),
        "phase_scale": float(cfg.get("photonic_phase_scale", np.pi)),
        "batch_size": _encoder_batch_size(cfg),
        "computation_space": _photonic_computation_space(cfg),
        "feature_cache": feature_cache,
        **_parallel_kwargs_from_config(cfg),
    }


def _photonic_hybrid_standardize(cfg: dict[str, Any]) -> bool:
    return bool(cfg.get("photonic_standardize_features", True))


def _photonic_svm_c(cfg: dict[str, Any]) -> float:
    return float(cfg.get("photonic_qsvm_c", cfg.get("qsvm_c", 1.0)))


def _photonic_fidelity_svm_c(cfg: dict[str, Any]) -> float:
    return float(
        cfg.get(
            "photonic_fidelity_qsvm_c",
            cfg.get("photonic_qsvm_c", cfg.get("qsvm_c", 1.0)),
        )
    )


def _photonic_knn_neighbors(cfg: dict[str, Any]) -> int:
    return int(cfg.get("photonic_qknn_neighbors", 5))


def _photonic_knn_weights(cfg: dict[str, Any]) -> str:
    return str(cfg.get("photonic_knn_weights", cfg.get("knn_weights", "distance")))


def _photonic_fidelity_knn_neighbors(cfg: dict[str, Any]) -> int:
    return int(
        cfg.get(
            "photonic_fidelity_qknn_neighbors",
            cfg.get("photonic_qknn_neighbors", 5),
        )
    )


def _photonic_fidelity_knn_weights(cfg: dict[str, Any]) -> str:
    return str(
        cfg.get(
            "photonic_fidelity_knn_weights",
            cfg.get(
                "photonic_knn_weights",
                cfg.get("knn_weights", "distance"),
            ),
        )
    )


def _photonic_fidelity_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "shots": int(cfg.get("photonic_fidelity_shots", 0)),
        "sampling_method": str(
            cfg.get("photonic_fidelity_sampling_method", "multinomial")
        ),
        "force_psd": bool(cfg.get("photonic_fidelity_force_psd", False)),
    }


def make_model(
    name: str,
    cfg: dict[str, Any],
    *,
    feature_cache: dict[tuple[Any, ...], dict[str, Any]] | None = None,
) -> ExperimentModel:
    """Create the configured high-level model wrapper.

    Parameters
    ----------
    name : str
        Model variant name from ``MODEL_REGISTRY``.
    cfg : dict
        Resolved experiment configuration.
    feature_cache : dict | None, optional
        Per-seed cache shared by photonic model wrappers. Default value is None.

    Returns
    -------
    ExperimentModel
        Model wrapper exposing ``fit`` and prediction methods.
    """

    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model {name!r}; expected one of {MODEL_REGISTRY}.")
    if name == "svm_classical":
        return SklearnModel(
            make_svc(
                kernel=str(cfg.get("svm_kernel", "rbf")),
                c_value=float(cfg.get("svm_c", 1.0)),
                gamma=cfg.get("svm_gamma", "scale"),
            )
        )
    if name == "knn_classical":
        return SklearnModel(
            make_knn(
                n_neighbors=int(cfg.get("knn_neighbors", 7)),
                metric=str(cfg.get("knn_metric", "euclidean")),
                weights=str(cfg.get("knn_weights", "distance")),
                n_jobs=_int_config(cfg, "parallel_n_jobs", 1),
            )
        )
    if name.startswith("state_svm_"):
        encoding = encoding_from_model(name)
        assert encoding is not None
        return GateStateFidelitySVMModel(
            encoding=encoding,
            device_name=str(cfg.get("quantum_device", "lightning.qubit")),
            angle_rotation=str(
                cfg.get("kernel_angle_rotation", cfg.get("angle_rotation", "X"))
            ),
            angle_scale=_angle_scale_for_encoding(cfg, encoding),
            c_value=float(cfg.get("qsvm_c", 1.0)),
            normalize_inputs=bool(cfg.get("qknn_normalize_inputs", False)),
            entangle=bool(cfg.get("qknn_entangle", False)),
            **_zz_kwargs(cfg),
            **_parallel_kwargs_from_config(cfg),
        )
    if name.startswith("state_knn_"):
        encoding = encoding_from_model(name)
        assert encoding is not None
        return GateStateFidelityKNNModel(
            encoding=encoding,
            device_name=str(cfg.get("quantum_device", "lightning.qubit")),
            angle_rotation=str(
                cfg.get("qknn_angle_rotation", cfg.get("angle_rotation", "X"))
            ),
            angle_scale=_angle_scale_for_encoding(cfg, encoding),
            n_neighbors=int(cfg.get("qknn_neighbors", 5)),
            normalize_inputs=bool(cfg.get("qknn_normalize_inputs", False)),
            entangle=bool(cfg.get("qknn_entangle", False)),
            **_zz_kwargs(cfg),
            **_parallel_kwargs_from_config(cfg),
        )
    if name.startswith("qsvm_"):
        encoding = encoding_from_model(name)
        assert encoding is not None
        return QSVMFidelityKernelModel(
            encoding=encoding,
            device_name=str(cfg.get("quantum_device", "lightning.qubit")),
            angle_rotation=str(
                cfg.get("kernel_angle_rotation", cfg.get("angle_rotation", "X"))
            ),
            angle_scale=_angle_scale_for_encoding(cfg, encoding),
            c_value=float(cfg.get("qsvm_c", 1.0)),
            **_zz_kwargs(cfg),
            **_parallel_kwargs_from_config(cfg),
        )
    if name.startswith("qknn_"):
        encoding = encoding_from_model(name)
        assert encoding is not None
        return QKNNFidelityModel(
            encoding=encoding,
            device_name=str(cfg.get("quantum_device", "lightning.qubit")),
            angle_rotation=str(
                cfg.get("qknn_angle_rotation", cfg.get("angle_rotation", "X"))
            ),
            angle_scale=_angle_scale_for_encoding(cfg, encoding),
            n_neighbors=int(cfg.get("qknn_neighbors", 5)),
            normalize_inputs=bool(cfg.get("qknn_normalize_inputs", False)),
            entangle=bool(cfg.get("qknn_entangle", False)),
            **_zz_kwargs(cfg),
            **_parallel_kwargs_from_config(cfg),
        )
    if name.startswith("hybrid_"):
        encoding = encoding_from_model(name)
        assert encoding is not None
        estimator = (
            make_svc(
                kernel=str(cfg.get("svm_kernel", "rbf")),
                c_value=float(cfg.get("svm_c", 1.0)),
                gamma=cfg.get("svm_gamma", "scale"),
            )
            if name.startswith("hybrid_svm_")
            else make_knn(
                n_neighbors=int(cfg.get("knn_neighbors", 7)),
                metric=str(cfg.get("knn_metric", "euclidean")),
                weights=str(cfg.get("knn_weights", "distance")),
                n_jobs=_int_config(cfg, "parallel_n_jobs", 1),
            )
        )
        return HybridSklearnModel(
            encoding=encoding,
            device_name=str(cfg.get("quantum_device", "lightning.qubit")),
            angle_rotation=str(cfg.get("hybrid_angle_rotation", "Y")),
            angle_scale=_angle_scale_for_encoding(cfg, encoding),
            estimator=estimator,
            hybrid_readout=str(cfg.get("hybrid_readout", "pauli_z")),
            standardize_features=bool(cfg.get("hybrid_standardize_features", True)),
            **_zz_kwargs(cfg),
            **_parallel_kwargs_from_config(cfg),
        )
    if name == "photonic_hybrid_svm_angle":
        return PhotonicSVMHybridModel(
            **_photonic_common_kwargs(cfg, feature_cache),
            standardize_features=_photonic_hybrid_standardize(cfg),
            c_value=_photonic_svm_c(cfg),
            kernel=str(cfg.get("svm_kernel", "rbf")),
            gamma=cfg.get("svm_gamma", "scale"),
        )
    if name == "photonic_hybrid_svm_amplitude":
        return PhotonicSVMHybridModel(
            **_photonic_common_kwargs(cfg, feature_cache),
            photonic_encoding="amplitude",
            standardize_features=_photonic_hybrid_standardize(cfg),
            c_value=_photonic_svm_c(cfg),
            kernel=str(cfg.get("svm_kernel", "rbf")),
            gamma=cfg.get("svm_gamma", "scale"),
        )
    if name == "photonic_fidelity_svm_angle":
        return PhotonicSVMFidelityKernelModel(
            **_photonic_common_kwargs(cfg, feature_cache),
            standardize_features=False,
            c_value=_photonic_fidelity_svm_c(cfg),
            **_photonic_fidelity_kwargs(cfg),
        )
    if name == "photonic_state_svm_angle":
        return PhotonicSVMStateFidelityModel(
            **_photonic_common_kwargs(cfg, feature_cache),
            standardize_features=False,
            c_value=_photonic_fidelity_svm_c(cfg),
        )
    if name == "photonic_hybrid_knn_angle":
        return PhotonicKNNHybridModel(
            **_photonic_common_kwargs(cfg, feature_cache),
            standardize_features=_photonic_hybrid_standardize(cfg),
            n_neighbors=_photonic_knn_neighbors(cfg),
            weights=_photonic_knn_weights(cfg),
        )
    if name == "photonic_hybrid_knn_amplitude":
        return PhotonicKNNHybridModel(
            **_photonic_common_kwargs(cfg, feature_cache),
            photonic_encoding="amplitude",
            standardize_features=_photonic_hybrid_standardize(cfg),
            n_neighbors=_photonic_knn_neighbors(cfg),
            weights=_photonic_knn_weights(cfg),
        )
    if name == "photonic_fidelity_knn_angle":
        return PhotonicKNNFidelityKernelModel(
            **_photonic_common_kwargs(cfg, feature_cache),
            standardize_features=False,
            n_neighbors=_photonic_fidelity_knn_neighbors(cfg),
            weights=_photonic_fidelity_knn_weights(cfg),
            **_photonic_fidelity_kwargs(cfg),
        )
    if name == "photonic_state_knn_angle":
        return PhotonicKNNStateFidelityModel(
            **_photonic_common_kwargs(cfg, feature_cache),
            standardize_features=False,
            n_neighbors=_photonic_fidelity_knn_neighbors(cfg),
            weights=_photonic_fidelity_knn_weights(cfg),
        )
    raise ValueError(f"Unhandled model {name!r}.")
