"""Smoke tests for Nearest Centroid Classification."""

import numpy as np
from lib.classifier import MLQuantumNearestCentroid, QuantumNearestCentroid
from lib.synthetic_data import generate_synthetic_data


def test_classifiers_exist():
    """Test that classifier classes can be instantiated."""
    clf_q = QuantumNearestCentroid(repetitions=10)
    clf_ml = MLQuantumNearestCentroid(n=4, repetitions=10)
    assert clf_q is not None
    assert clf_ml is not None


def test_synthetic_data_generation():
    """Test synthetic data generation."""
    X, y, centroids = generate_synthetic_data(
        n_clusters=2,
        n_dimensions=4,
        n_points_per_cluster=5,
        seed=42,
    )
    assert X.shape == (10, 4)
    assert y.shape == (10,)
    assert len(centroids) == 2


def test_quantum_classifier_fit_predict():
    """Test that quantum classifier can fit and predict."""
    # Generate simple synthetic data
    X, y, _ = generate_synthetic_data(
        n_clusters=2,
        n_dimensions=4,
        n_points_per_cluster=10,
        min_centroid_distance=0.5,
        seed=123,
    )

    # Scale to [0, 1]
    X = (X - X.min()) / (X.max() - X.min() + 1e-8)

    # Shuffle to mix classes (data is generated sequentially by cluster)
    indices = np.arange(len(y))
    np.random.seed(42)
    np.random.shuffle(indices)
    X, y = X[indices], y[indices]

    X_train, X_test = X[:10], X[10:]
    y_train, y_test = y[:10], y[10:]

    # Verify both classes are present in train set
    assert len(np.unique(y_train)) == 2, "Training set must have both classes"

    # Fit and predict with Cirq backend
    clf = QuantumNearestCentroid(repetitions=100)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    assert y_pred.shape == y_test.shape
    # With well-separated clusters, accuracy should be reasonable
    accuracy = np.mean(y_pred == y_test)
    assert accuracy >= 0.5  # Better than random


def test_merlin_classifier_fit_predict():
    """Test that MerLin classifier can fit and predict."""
    X, y, _ = generate_synthetic_data(
        n_clusters=2,
        n_dimensions=4,
        n_points_per_cluster=10,
        min_centroid_distance=0.5,
        seed=123,
    )

    X = (X - X.min()) / (X.max() - X.min() + 1e-8)

    # Shuffle to mix classes
    indices = np.arange(len(y))
    np.random.seed(42)
    np.random.shuffle(indices)
    X, y = X[indices], y[indices]

    X_train, X_test = X[:10], X[10:]
    y_train, y_test = y[:10], y[10:]

    # Verify both classes are present
    assert len(np.unique(y_train)) == 2, "Training set must have both classes"

    clf = MLQuantumNearestCentroid(n=4, repetitions=100)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    assert y_pred.shape == y_test.shape
    accuracy = np.mean(y_pred == y_test)
    assert accuracy >= 0.5


def test_train_and_evaluate_import():
    """Test that train_and_evaluate can be imported."""
    from lib.runner import train_and_evaluate

    assert callable(train_and_evaluate)
