from __future__ import annotations

import numpy as np
from common import PROJECT_DIR  # noqa: F401  (puts the repo on sys.path)
from QNN_hyperparameters.lib.circuits import build_ansatz
from QNN_hyperparameters.lib.qnn import (
    QNNClassifier,
    fit_beta_parameters,
    initial_point,
)

MODEL = {
    "feature_map": "ZFeatureMap",
    "feature_map_entanglement": None,
    "feature_map_reps": 2,
    "ansatz": "RealAmplitudes",
    "ansatz_entanglement": "linear",
    "ansatz_reps": 3,
    "init": "qiskit_default",
}


def test_author_assign_parameters_does_not_set_an_initial_point():
    """The author code's initialization call leaves the ansatz unchanged.

    ``opt.py`` calls ``vqc.ansatz.assign_parameters(values)`` and discards the
    returned circuit; the ansatz keeps all its free parameters, so the VQC
    still starts from qiskit-ml's default random point.
    """

    ansatz = build_ansatz("RealAmplitudes", 7, "linear")
    bound = ansatz.assign_parameters(np.zeros(ansatz.num_parameters))
    assert ansatz.num_parameters == 28
    assert bound.num_parameters == 0


def test_initial_points_shapes_and_values():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 4))
    assert initial_point("qiskit_default", 10, 3, X, rng)[0] is None
    assert np.all(initial_point("zeros", 10, 3, X, rng)[0] == 0)
    beta_mu, details = initial_point("beta_mu", 10, 3, X, rng)
    assert np.allclose(beta_mu, details["beta_mean"])
    a, b = fit_beta_parameters(X)
    assert a > 0 and b > 0
    normal, details = initial_point("normal", 2000, 4, X, rng)
    assert abs(normal.std() - 0.5) < 0.05 and details["sigma2"] == 0.25


def test_fit_logs_evaluations_and_learns_separable_data(caplog):
    rng = np.random.default_rng(3)
    X = np.vstack([rng.normal(0.2, 0.05, (30, 2)), rng.normal(0.9, 0.05, (30, 2))])
    y = np.array([0] * 30 + [1] * 30)
    model = QNNClassifier(
        {**MODEL}, n_qubits=2, shots=None, rng=np.random.default_rng(0)
    )
    with caplog.at_level("INFO"):
        result = model.fit(
            X,
            y,
            {
                "optimizer": "COBYLA",
                "maxiter": 60,
                "tol": 1e-3,
                "maxfev": 1000,
                "shots": None,
            },
        )
    assert result.evaluations == len(result.losses) <= 60
    assert result.losses[-1] < result.losses[0]
    assert (
        sum("TRAIN_EVALUATION_COMPLETED" in r.message for r in caplog.records)
        == result.evaluations
    )
    scores = model.evaluate(X, y)
    assert set(scores) == {"accuracy", "f1_weighted"}
