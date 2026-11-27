"""Models trained in Sec. VI and the fair classical baseline.

``HamiltonianNeuron`` is the paper's quantized neuron: the output on a state
rho is ``Tr[phi(H(theta)) rho]`` with ``H(theta) = sum_j theta_j H_j``
(Eqs. (16), (117)-(118)). The paper's "quantum" model (TFIM, Heisenberg) and
its "classical" model (IM, FCIM, all terms diagonal) are both instances; only
the term set differs.

``PauliFeatureNeuron`` is baseline B2 of the LOG.md fair-baseline plan: a
classical neuron ``phi(sum_j theta_j <H_j>_rho)`` on the expectation values of
the same Pauli terms. It sees exactly the measurement data a classical learner
gets from local Pauli measurements of rho and has the same parameter count.
With ``phi = linear`` it equals the paper's quantum linear model
``Tr[H(theta) rho]`` (Fig. 9, baseline B3).

Losses:

* ``squared``: Eq. (44), ``mean_m (f(rho_m) - y_m)**2``.
* ``logistic``: Eq. (56), ``mean_m T Tr[ln(I + exp(-y_m H(theta) / T)) rho_m]``
  for the Hamiltonian neuron, and the classical logistic loss Eq. (52) on the
  features for B2. The activation is not used by the logistic loss.

Classification prediction rules (LOG.md D4):

* ``activation``: sign of ``Tr[g_T(H(theta)) rho]`` with ``g_T = tanh(./T)``,
  the paper's rule (p. 19: compare ``Tr[phi(H_Q(theta)) rho_m]`` with ``y_m``,
  where ``phi = g_T`` for classification).
* ``linear``: sign of ``Tr[H(theta) rho]``, the rule of
  ``logloss_nqubits.ipynb::calculate_accuracy``.

A zero output is mapped to +1, as in the notebook.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .activations import (
    Activation,
    divided_differences,
    make_activation,
    matrix_function,
)
from .pauli import build_terms, is_commuting

LOSSES = ("squared", "logistic")
PREDICTION_RULES = ("activation", "linear")


def _sign_with_positive_zero(values: np.ndarray) -> np.ndarray:
    return np.where(values >= 0.0, 1.0, -1.0)


def expectations(observable: np.ndarray, states: np.ndarray) -> np.ndarray:
    """Return ``Tr[O rho_m]`` for every state as one matrix-vector product.

    Same values as the einsum contraction over (a, b) to 1e-15, about 7x
    faster at n = 9 because it runs as one BLAS call over the flattened
    ``(M, d*d)`` state array, which is a view, not a copy.
    """
    flat_states = states.reshape(len(states), -1)
    return np.real(flat_states @ observable.T.reshape(-1))


def weighted_state_sum(weights: np.ndarray, states: np.ndarray) -> np.ndarray:
    """Return ``sum_m w_m rho_m`` as one vector-matrix product."""
    flat_states = states.reshape(len(states), -1)
    dim = states.shape[-1]
    return (weights.astype(states.dtype) @ flat_states).reshape(dim, dim)


@dataclass
class Dataset:
    """Full-batch dataset of density matrices.

    Parameters
    ----------
    states : numpy.ndarray
        Complex array of shape ``(M, d, d)``.
    targets : numpy.ndarray
        Real targets (regression) or labels in {-1, +1}, shape ``(M,)``.
    """

    states: np.ndarray
    targets: np.ndarray
    feature_cache: dict = field(default_factory=dict, repr=False)

    def pauli_features(self, labels: list[str], terms: np.ndarray) -> np.ndarray:
        """Return ``<H_j>_rho_m`` as a real ``(M, J)`` array, cached per term list."""
        key = tuple(labels)
        if key not in self.feature_cache:
            self.feature_cache[key] = np.real(
                np.einsum("jab,mba->mj", terms, self.states)
            )
        return self.feature_cache[key]


class HamiltonianNeuron:
    """Activation observable ``phi(H(theta))`` evaluated on input states.

    Parameters
    ----------
    term_set : str
        Pauli term set, see :func:`lib.pauli.term_labels`.
    n_qubits : int
        Number of qubits.
    include_identity : bool
        Whether the identity (bias) term is part of ``H(theta)``.
    activation : str
        Activation name for the squared loss and the ``activation``
        prediction rule, see :func:`lib.activations.make_activation`.
    temperature : float
        Temperature ``T`` of the activation and of the logistic loss.
    explicit_labels : list of str or None
        Pauli strings when ``term_set`` is ``"explicit"``; must be None
        otherwise. Default value is None.
    """

    def __init__(
        self,
        term_set: str,
        n_qubits: int,
        include_identity: bool,
        activation: str,
        temperature: float,
        explicit_labels: list[str] | None = None,
    ) -> None:
        self.term_set = term_set
        self.labels, self.terms = build_terms(
            term_set, n_qubits, include_identity, explicit_labels
        )
        self.temperature = float(temperature)
        self.activation: Activation = make_activation(activation, temperature)
        self._classifier_activation = make_activation("tanh", temperature)
        self._logistic = {
            1.0: make_activation("logistic_pos", temperature),
            -1.0: make_activation("logistic_neg", temperature),
        }
        self.commuting = is_commuting(self.terms)

    @property
    def n_parameters(self) -> int:
        """Number of trainable coefficients ``J``."""
        return len(self.labels)

    def _spectrum(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return np.linalg.eigh(np.einsum("j,jab->ab", theta, self.terms))

    def _trace_gradient(
        self,
        activation: Activation,
        eigenvalues: np.ndarray,
        eigenvectors: np.ndarray,
        weighted_state: np.ndarray,
    ) -> np.ndarray:
        """Return ``d/dtheta_j Tr[phi(H) R]`` for every j (Daleckii-Krein).

        The per-term formula ``Re sum_ab F_ab (V^dag H_j V)_ab (V^dag R V)_ba``
        equals ``Re Tr[H_j V (F o V^dag R V) V^dag]`` because the
        divided-difference matrix F is symmetric. One basis change per call
        replaces J of them (O(d^3) instead of O(J d^3)), which made the
        9-qubit E11 runs about 9x faster.
        """
        divided = divided_differences(activation, eigenvalues)
        rotated_state = eigenvectors.conj().T @ weighted_state @ eigenvectors
        gradient_operator = (
            eigenvectors @ (divided * rotated_state) @ eigenvectors.conj().T
        )
        return np.real(np.einsum("jab,ba->j", self.terms, gradient_operator))

    def outputs(self, theta: np.ndarray, data: Dataset) -> np.ndarray:
        """Return ``Tr[phi(H(theta)) rho_m]`` for every state."""
        eigenvalues, eigenvectors = self._spectrum(theta)
        observable = matrix_function(self.activation, eigenvalues, eigenvectors)
        return expectations(observable, data.states)

    def loss_and_gradient(
        self, theta: np.ndarray, data: Dataset, loss: str
    ) -> tuple[float, np.ndarray]:
        """Return the loss and its exact gradient at ``theta``.

        Raises
        ------
        ValueError
            If ``loss`` is not ``"squared"`` or ``"logistic"``.
        """
        eigenvalues, eigenvectors = self._spectrum(theta)
        n_states = len(data.targets)
        if loss == "squared":
            observable = matrix_function(self.activation, eigenvalues, eigenvectors)
            predictions = expectations(observable, data.states)
            residuals = predictions - data.targets
            # The gradient only needs the residual-weighted state sum (Eq. (48)).
            weighted_state = weighted_state_sum(2.0 * residuals / n_states, data.states)
            gradient = self._trace_gradient(
                self.activation, eigenvalues, eigenvectors, weighted_state
            )
            return float(np.mean(residuals**2)), gradient
        if loss == "logistic":
            value = 0.0
            gradient = np.zeros(self.n_parameters)
            for label, observable_fn in self._logistic.items():
                mask = data.targets == label
                if not np.any(mask):
                    continue
                # Eq. (60)-(62): the loss is linear in the labelled state, so
                # states with the same label can be summed first.
                class_state = data.states[mask].sum(axis=0) / n_states
                observable = matrix_function(observable_fn, eigenvalues, eigenvectors)
                value += float(np.real(np.einsum("ab,ba->", observable, class_state)))
                gradient += self._trace_gradient(
                    observable_fn, eigenvalues, eigenvectors, class_state
                )
            return value, gradient
        raise ValueError(f"loss must be one of {LOSSES}, got {loss!r}")

    def finite_shot_loss_and_gradient(
        self,
        theta: np.ndarray,
        data: Dataset,
        shots: int,
        rng: np.random.Generator,
    ) -> tuple[float, np.ndarray]:
        """Squared loss (exact) and a finite-shot estimate of its gradient (E9b).

        ``g_j = (2/M) sum_m (f_m - y_m) d_j f_m`` is estimated with
        independent single-shot outcomes per (state, parameter): Algorithm 1
        (p. 8) outputs +-1/T with mean ``d_j f_m``; Algorithm 2 outputs
        +-|theta|_1/T with mean ``f_m``. Both are two-point random variables,
        so ``shots`` outcomes are drawn as a binomial count with the exact
        probability, which has the same distribution as simulating each
        circuit (verified in E0). The product of independent unbiased
        estimates keeps the gradient estimate unbiased.

        Raises
        ------
        ValueError
            If the activation is not tanh (the algorithms estimate
            ``g_T = tanh(x / T)``), the register is larger than 6 qubits (the
            per-state gradients are computed densely), or an outcome
            probability falls outside [0, 1].
        """
        if self.activation.name != "tanh":
            raise ValueError(
                "finite-shot gradients implement Algorithms 1 and 2, which estimate tanh(H/T)"
            )
        if self.terms.shape[-1] > 64:
            raise ValueError("finite-shot gradients are limited to n <= 6")
        eigenvalues, eigenvectors = self._spectrum(theta)
        observable = matrix_function(self.activation, eigenvalues, eigenvectors)
        outputs = expectations(observable, data.states)
        divided = divided_differences(self.activation, eigenvalues)
        rotated_terms = eigenvectors.conj().T @ self.terms @ eigenvectors
        rotated_states = eigenvectors.conj().T @ data.states @ eigenvectors
        per_state_gradients = np.real(
            np.einsum("jab,mba->mj", divided * rotated_terms, rotated_states)
        )
        temperature = self.temperature
        output_scale = float(np.sum(np.abs(theta))) / temperature
        gradient_probability = (1.0 + temperature * per_state_gradients) / 2.0
        output_probability = (1.0 + outputs / output_scale) / 2.0
        for name, probability in (
            ("gradient", gradient_probability),
            ("output", output_probability),
        ):
            if np.any(probability < -1e-12) or np.any(probability > 1 + 1e-12):
                raise ValueError(
                    f"{name} outcome probability outside [0, 1]: the estimator scale is wrong"
                )
        gradient_hits = rng.binomial(shots, np.clip(gradient_probability, 0.0, 1.0))
        output_hits = rng.binomial(shots, np.clip(output_probability, 0.0, 1.0))
        gradient_estimates = (2.0 * gradient_hits / shots - 1.0) / temperature
        output_estimates = (2.0 * output_hits / shots - 1.0) * output_scale
        residual_estimates = output_estimates - data.targets
        gradient = 2.0 * residual_estimates @ gradient_estimates / len(data.targets)
        loss = float(np.mean((outputs - data.targets) ** 2))
        return loss, gradient

    def classify(self, theta: np.ndarray, data: Dataset, rule: str) -> np.ndarray:
        """Return predicted labels in {-1, +1} under a prediction rule.

        Raises
        ------
        ValueError
            If ``rule`` is unknown.
        """
        eigenvalues, eigenvectors = self._spectrum(theta)
        if rule == "activation":
            observable = matrix_function(
                self._classifier_activation, eigenvalues, eigenvectors
            )
        elif rule == "linear":
            observable = (eigenvectors * eigenvalues) @ eigenvectors.conj().T
        else:
            raise ValueError(f"rule must be one of {PREDICTION_RULES}, got {rule!r}")
        scores = expectations(observable, data.states)
        return _sign_with_positive_zero(scores)


class PauliFeatureNeuron:
    """Classical neuron on Pauli expectation features (baseline B2/B3).

    Parameters are as for :class:`HamiltonianNeuron`; ``term_set`` selects the
    Pauli strings whose expectation values form the features.
    """

    def __init__(
        self,
        term_set: str,
        n_qubits: int,
        include_identity: bool,
        activation: str,
        temperature: float,
    ) -> None:
        self.term_set = term_set
        self.labels, self.terms = build_terms(term_set, n_qubits, include_identity)
        self.temperature = float(temperature)
        self.activation = make_activation(activation, temperature)
        self._classifier_activation = make_activation("tanh", temperature)

    @property
    def n_parameters(self) -> int:
        """Number of trainable weights ``J``."""
        return len(self.labels)

    def features(self, data: Dataset) -> np.ndarray:
        """Return ``<H_j>_rho_m`` as a real array of shape ``(M, J)``."""
        return data.pauli_features(self.labels, self.terms)

    def outputs(self, theta: np.ndarray, data: Dataset) -> np.ndarray:
        """Return ``phi(sum_j theta_j <H_j>_rho)`` for every state."""
        return self.activation.value(self.features(data) @ theta)

    def loss_and_gradient(
        self, theta: np.ndarray, data: Dataset, loss: str
    ) -> tuple[float, np.ndarray]:
        """Return the loss and its exact gradient at ``theta``.

        Raises
        ------
        ValueError
            If ``loss`` is unknown.
        """
        features = self.features(data)
        pre_activation = features @ theta
        n_states = len(data.targets)
        if loss == "squared":
            residuals = self.activation.value(pre_activation) - data.targets
            weights = 2.0 * residuals * self.activation.derivative(pre_activation)
            return float(np.mean(residuals**2)), features.T @ weights / n_states
        if loss == "logistic":
            T = self.temperature
            margins = data.targets * pre_activation / T
            value = float(np.mean(T * np.logaddexp(0.0, -margins)))
            weights = -data.targets / (1.0 + np.exp(margins))
            return value, features.T @ weights / n_states
        raise ValueError(f"loss must be one of {LOSSES}, got {loss!r}")

    def classify(self, theta: np.ndarray, data: Dataset, rule: str) -> np.ndarray:
        """Return predicted labels; both rules reduce to sign(theta . v) for odd phi."""
        if rule not in PREDICTION_RULES:
            raise ValueError(f"rule must be one of {PREDICTION_RULES}, got {rule!r}")
        pre_activation = self.features(data) @ theta
        scores = (
            self._classifier_activation.value(pre_activation)
            if rule == "activation"
            else pre_activation
        )
        return _sign_with_positive_zero(scores)


MODEL_KINDS = {"hamiltonian": HamiltonianNeuron, "pauli_features": PauliFeatureNeuron}


def build_model(model_cfg: dict, n_qubits: int, temperature: float):
    """Instantiate one model from its config entry.

    Parameters
    ----------
    model_cfg : dict
        Keys ``kind``, ``term_set``, ``include_identity``, ``activation``,
        and ``term_labels`` when ``term_set`` is ``"explicit"``.
    n_qubits : int
        Number of qubits.
    temperature : float
        Temperature ``T``.

    Raises
    ------
    ValueError
        If ``kind`` is unknown.
    """
    kind = model_cfg["kind"]
    if kind not in MODEL_KINDS:
        raise ValueError(
            f"model kind must be one of {sorted(MODEL_KINDS)}, got {kind!r}"
        )
    arguments = (
        model_cfg["term_set"],
        n_qubits,
        bool(model_cfg["include_identity"]),
        model_cfg["activation"],
        temperature,
    )
    if model_cfg["term_set"] == "explicit":
        if kind != "hamiltonian":
            raise ValueError(
                "explicit term labels are only supported for hamiltonian models"
            )
        return HamiltonianNeuron(*arguments, explicit_labels=model_cfg["term_labels"])
    return MODEL_KINDS[kind](*arguments)
