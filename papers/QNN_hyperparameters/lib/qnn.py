"""VQC training as in the author code, with the initialization made explicit.

Training follows qiskit-machine-learning ``VQC`` (0.6.0 in the author
environment, same semantics in 0.8.4): class probabilities from the
``x % num_classes`` readout, one-hot targets in sorted label order, loss
``mean_i -log2(clip(p_i[y_i], 1e-10))`` over the 400 training rows, one
optimizer call, prediction by arg-max of a fresh shot-sampled forward pass.

Initial points (``model.init``):

- ``qiskit_default``: what every author run actually used. The author code
  calls ``ansatz.assign_parameters(...)`` and discards the returned copy, so
  qiskit-ml falls back to ``algorithm_globals.random.random(num_weights)``,
  i.e. U[0, 1).
- ``uniform``: U[0, 1), passed explicitly (same distribution as above).
- ``beta``: BEINIT (Kulshrestha and Safro); fit a beta distribution to the
  MinMax-scaled training features (author ``init_beta``) and draw weights.
- ``beta_mu``: every weight at the fitted beta mean.
- ``normal_beta``: normal with the fitted beta mean and standard deviation.
- ``normal``: N(0, 1/L) with L the ansatz repetitions (Zhang et al.).
- ``zeros``: all zeros.

All except ``qiskit_default`` are passed to the optimizer as the initial
point, which the author code intended but did not do.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import beta as beta_distribution
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import MinMaxScaler

from .circuits import build_ansatz, build_feature_map
from .simulator import CompiledCircuit, class_probabilities

logger = logging.getLogger(__name__)

INITS = (
    "qiskit_default",
    "uniform",
    "beta",
    "beta_mu",
    "normal_beta",
    "normal",
    "zeros",
)
OPTIMIZERS = ("COBYLA", "SPSA", "NelderMead")


def fit_beta_parameters(X_train: np.ndarray) -> tuple[float, float]:
    """Fit BEINIT's beta distribution to the training features.

    Port of the author's ``init_beta`` (adapted from BEINIT): MinMax-scale the
    features to [0, 1], nudge exact 0 and 1 by 1e-8, then
    ``scipy.stats.beta.fit(values, floc=0, fscale=1)``.
    """

    values = MinMaxScaler().fit_transform(X_train).reshape(-1).copy()
    values[values <= 0] += 1e-8
    values[values >= 1] -= 1e-8
    a, b, _, _ = beta_distribution.fit(values, floc=0, fscale=1)
    return float(a), float(b)


def initial_point(
    init: str,
    num_weights: int,
    ansatz_reps: int,
    X_train: np.ndarray,
    rng: np.random.Generator,
):
    """Return the optimizer start point, or None for qiskit-ml's own default.

    Returns
    -------
    tuple
        ``(point_or_None, details)`` where details records fitted parameters.
    """

    if init == "qiskit_default":
        return None, {}
    if init == "uniform":
        return rng.random(num_weights), {}
    if init == "zeros":
        return np.zeros(num_weights), {}
    if init == "normal":
        return rng.normal(0.0, np.sqrt(1.0 / ansatz_reps), num_weights), {
            "sigma2": 1.0 / ansatz_reps
        }
    if init in ("beta", "beta_mu", "normal_beta"):
        a, b = fit_beta_parameters(X_train)
        mean = a / (a + b)
        std = float(np.sqrt(a * b / ((a + b) ** 2 * (a + b + 1))))
        details = {"beta_a": a, "beta_b": b, "beta_mean": mean, "beta_std": std}
        if init == "beta":
            return rng.beta(a, b, num_weights), details
        if init == "beta_mu":
            return np.full(num_weights, mean), details
        return rng.normal(mean, std, num_weights), details
    raise ValueError(f"unknown init {init!r}; expected one of {INITS}")


def build_optimizer(training_cfg: dict):
    """Qiskit optimizer with the budgets the author runs actually used.

    ``training.maxiter`` must already be the executed budget (see
    ``configs/*``): the author ``get_configurations`` swaps the SPSA and
    Nelder-Mead budgets. COBYLA receives ``tol``; SPSA ignores it (author
    ``get_spsa``); Nelder-Mead is adaptive with qiskit's default
    ``maxfev=1000``.
    """

    from qiskit_machine_learning.optimizers import COBYLA, NELDER_MEAD, SPSA

    name = training_cfg["optimizer"]
    maxiter = int(training_cfg["maxiter"])
    if name == "COBYLA":
        return COBYLA(maxiter=maxiter, tol=training_cfg["tol"])
    if name == "SPSA":
        return SPSA(maxiter=maxiter)
    if name == "NelderMead":
        return NELDER_MEAD(
            maxiter=maxiter,
            tol=training_cfg["tol"],
            adaptive=True,
            maxfev=int(training_cfg["maxfev"]),
        )
    raise ValueError(f"unknown optimizer {name!r}; expected one of {OPTIMIZERS}")


@dataclass
class TrainingResult:
    """Outcome of one VQC fit."""

    weights: np.ndarray
    losses: list = field(default_factory=list)
    evaluations: int = 0
    fit_seconds: float = 0.0
    initial_point_details: dict = field(default_factory=dict)
    classes: np.ndarray | None = None


class QNNClassifier:
    """Feature map + ansatz classifier trained like qiskit-ml ``VQC``.

    Parameters
    ----------
    model_cfg : dict
        ``feature_map``, ``feature_map_entanglement``, ``feature_map_reps``,
        ``ansatz``, ``ansatz_entanglement``, ``ansatz_reps``, ``init``.
    n_qubits : int
        Number of input features.
    shots : int | None
        Shots per sample and forward pass (author: 1024).
    rng : numpy.random.Generator
        Source of initial points and shot noise.
    noise_model : str | None
        ``FakePerth`` routes evaluation through Qiskit Aer with that noise
        model; None uses the batched statevector simulator.
    """

    def __init__(
        self,
        model_cfg: dict,
        n_qubits: int,
        shots: int | None,
        rng,
        noise_model: str | None = None,
    ):
        self.model_cfg = model_cfg
        self.shots = shots
        self.rng = rng
        self.feature_map = build_feature_map(
            model_cfg["feature_map"],
            n_qubits,
            model_cfg["feature_map_entanglement"],
            model_cfg["feature_map_reps"],
        )
        self.ansatz = build_ansatz(
            model_cfg["ansatz"],
            n_qubits,
            model_cfg["ansatz_entanglement"],
            model_cfg["ansatz_reps"],
        )
        self.num_weights = self.ansatz.num_parameters
        self.noise_model = noise_model
        if noise_model is None:
            self._compiled = CompiledCircuit(self.feature_map, self.ansatz)
        else:
            self._aer_qnn = _build_aer_qnn(
                self.feature_map, self.ansatz, noise_model, shots
            )
        self.classes = None
        self.weights = None
        # Called with the evaluation count after every loss evaluation, so a
        # sweep coordinator can report progress of a running candidate.
        self.progress_hook = None

    def class_probabilities(self, X: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Shot-sampled class probabilities, shape ``(batch, num_classes)``."""

        num_classes = len(self.classes)
        if self.noise_model is None:
            return class_probabilities(
                self._compiled, X, weights, num_classes, self.shots, self.rng
            )
        self._aer_qnn.set_interpret(lambda x: x % num_classes, num_classes)
        return np.asarray(self._aer_qnn.forward(X, weights))

    def fit(self, X: np.ndarray, y: np.ndarray, training_cfg: dict) -> TrainingResult:
        """Minimize the VQC cross-entropy and log every loss evaluation."""

        self.classes = np.unique(y)
        targets = np.searchsorted(self.classes, y)
        start, details = initial_point(
            self.model_cfg["init"],
            self.num_weights,
            self.model_cfg["ansatz_reps"],
            X,
            self.rng,
        )
        if start is None:
            # qiskit-ml TrainableModel._choose_initial_point: U[0, 1).
            start = self.rng.random(self.num_weights)
        result = TrainingResult(
            weights=start, initial_point_details=details, classes=self.classes
        )
        budget = int(training_cfg["maxiter"])

        def objective(weights):
            probabilities = self.class_probabilities(X, weights)
            loss = float(
                np.mean(
                    -np.log2(
                        np.clip(
                            probabilities[np.arange(len(targets)), targets], 1e-10, None
                        )
                    )
                )
            )
            result.evaluations += 1
            result.losses.append(loss)
            # The author callback counts objective evaluations; log each one.
            logger.info(
                "TRAIN_EVALUATION_COMPLETED | evaluation=%d | budget_maxiter=%d | loss=%.6f",
                result.evaluations,
                budget,
                loss,
            )
            if self.progress_hook is not None:
                self.progress_hook(result.evaluations)
            return loss

        optimizer = build_optimizer(training_cfg)
        began = time.perf_counter()
        optimum = optimizer.minimize(fun=objective, x0=np.asarray(start, dtype=float))
        result.fit_seconds = time.perf_counter() - began
        result.weights = np.asarray(optimum.x, dtype=float)
        self.weights = result.weights
        return result

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Arg-max of a fresh shot-sampled forward pass (qiskit-ml ``predict``)."""

        return self.classes[
            np.argmax(self.class_probabilities(X, self.weights), axis=1)
        ]

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Accuracy and weighted F1, the author's two metrics."""

        predictions = self.predict(X)
        return {
            "accuracy": float(accuracy_score(y, predictions)),
            "f1_weighted": float(f1_score(y, predictions, average="weighted")),
        }


def _build_aer_qnn(feature_map, ansatz, noise_model: str, shots: int | None):
    """SamplerQNN on Aer with a backend noise model, as in the author's noisy runs.

    The author used ``BackendSampler(QasmSimulator(noise_model=...))``: the
    circuit is transpiled to the noise model's basis gates without a coupling
    map. We reproduce that with a preset pass manager on those basis gates.
    """

    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_aer.noise import NoiseModel
    from qiskit_aer.primitives import SamplerV2
    from qiskit_machine_learning.neural_networks import SamplerQNN

    if noise_model != "FakePerth":
        raise ValueError(f"unsupported noise model {noise_model!r}")
    from qiskit_ibm_runtime.fake_provider import FakePerth

    model = NoiseModel.from_backend(FakePerth())
    circuit = feature_map.compose(ansatz)
    circuit.measure_all()
    pass_manager = generate_preset_pass_manager(
        basis_gates=model.basis_gates, optimization_level=1
    )
    transpiled = pass_manager.run(circuit)
    sampler = SamplerV2(
        default_shots=shots, options={"backend_options": {"noise_model": model}}
    )
    return SamplerQNN(
        circuit=transpiled,
        input_params=list(feature_map.parameters),
        weight_params=list(ansatz.parameters),
        sampler=sampler,
        interpret=lambda x: x % 2,
        output_shape=2,
        input_gradients=False,
    )
