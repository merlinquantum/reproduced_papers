"""Synthetic quantum data of Sec. VI A and the authors' reference outputs.

The paper's data are generated, not downloaded: a random target Hamiltonian
``H* = sum_j zeta_j P_j`` with ``zeta_j ~ U[low, high]`` (Sec. VI D1: [-2, 2])
and a set of input density matrices ``rho_m``. Regression targets are
``y_m = Tr[phi(H*) rho_m]`` (Eq. (120)); classification labels are a sign
(Sec. VI A, LOG.md D3):

* ``sign_activation``: ``y_m = sgn Tr[g_T(H*) rho_m]`` (paper text, p. 19),
* ``sign_linear``: ``y_m = sgn Tr[H* rho_m]`` (``logloss_nqubits.ipynb``).

State sets (LOG.md D6):

* ``code_squared_loss``: ``mse_nqubits.ipynb`` set, 2**(n+1) + 5 states:
  computational and |+-> product states, GHZ, I/2**n, 3 Ginibre mixed states.
* ``paper_squared_loss``: paper text p. 21: computational, Hadamard, Bell,
  GHZ, sqrt(p)|0..0> + sqrt(1-p)|1..1> for ``p_values``, and I/2**n. For
  n > 2 the four Bell states sit on qubits 0, 1 with the rest in |0>
  (assumption: the paper only writes two-qubit Bell states).
* ``pauli_basis``: paper text pp. 21-22: every product eigenstate of X, Y
  and Z on all qubits (3 * 2**n states, the three bases in equal parts).
* ``haar``: ``n_states`` Haar-random pure states (normalized complex Gaussian
  vectors, as in the notebook). The notebook uses 1000 training and 500
  validation states for Fig. 8.

Random streams: every quantity draws from
``numpy.random.default_rng([seed, stream])`` with a fixed stream index, so the
target and the states depend only on ``seed`` and the data config. Configs
that share a seed therefore see identical data (paired comparisons).

Split names: ``train`` and ``validation``. For regression the ``validation``
split is a held-out Haar set added by this reproduction (the paper reports
training loss only).
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .activations import make_activation, matrix_function
from .models import Dataset, expectations
from .pauli import build_terms

logger = logging.getLogger(__name__)

STATE_SETS = (
    "code_squared_loss",
    "paper_squared_loss",
    "pauli_basis",
    "computational_basis",
    "haar",
)
LABEL_RULES = ("regression", "sign_activation", "sign_linear")
SPLITS = ("train", "validation")
PAPER_NAME = "fermi_dirac_neurons"

STREAM_TARGET = 0
STREAM_TRAIN_STATES = 1
STREAM_VALIDATION_STATES = 2
STREAM_INITIALIZATION = 3
# Finite-shot gradient outcomes (E9b): one stream per model, 50 + model index.
STREAM_SHOTS = 50
# Phase 5: permutation that carves the learning-rate selection split out of
# the training states.
STREAM_SELECTION = 60

_KET_0 = np.array([1, 0], dtype=np.complex128)
_KET_1 = np.array([0, 1], dtype=np.complex128)
_KET_PLUS = np.array([1, 1], dtype=np.complex128) / np.sqrt(2)
_KET_MINUS = np.array([1, -1], dtype=np.complex128) / np.sqrt(2)
_KET_PLUS_I = np.array([1, 1j], dtype=np.complex128) / np.sqrt(2)
_KET_MINUS_I = np.array([1, -1j], dtype=np.complex128) / np.sqrt(2)


def random_stream(seed: int, stream: int) -> np.random.Generator:
    """Return the NumPy generator of one named random stream."""
    return np.random.default_rng([int(seed), int(stream)])


def selection_split(
    n_states: int, fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Split training-state indices into (train, selection) for learning-rate selection.

    ``round(fraction * n_states)`` states, at least one, go to the selection
    split, chosen by a permutation from stream :data:`STREAM_SELECTION`; both
    index arrays are sorted so state order is preserved.

    Raises
    ------
    ValueError
        If ``fraction`` is not in (0, 1) or fewer than 2 states are given.
    """
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"selection fraction must be in (0, 1), got {fraction}")
    if n_states < 2:
        raise ValueError(f"need at least 2 training states to split, got {n_states}")
    n_selection = min(n_states - 1, max(1, round(fraction * n_states)))
    order = random_stream(seed, STREAM_SELECTION).permutation(n_states)
    return np.sort(order[n_selection:]), np.sort(order[:n_selection])


def _density(vector: np.ndarray) -> np.ndarray:
    return np.outer(vector, vector.conj())


def _product(kets) -> np.ndarray:
    vector = np.ones(1, dtype=np.complex128)
    for ket in kets:
        vector = np.kron(vector, ket)
    return vector


def _product_basis(single_qubit_basis, n_qubits: int):
    """Yield the product-basis density matrices one at a time."""
    for kets in itertools.product(single_qubit_basis, repeat=n_qubits):
        yield _density(_product(kets))


def _ghz(n_qubits: int) -> np.ndarray:
    return (_product([_KET_0] * n_qubits) + _product([_KET_1] * n_qubits)) / np.sqrt(2)


def haar_pure_states(
    n_qubits: int, n_states: int, rng: np.random.Generator
) -> np.ndarray:
    """Return ``n_states`` Haar-random pure density matrices."""
    dim = 2**n_qubits
    vectors = rng.standard_normal((n_states, dim)) + 1j * rng.standard_normal(
        (n_states, dim)
    )
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.einsum("ma,mb->mab", vectors, vectors.conj())


def ginibre_mixed_states(
    n_qubits: int, n_states: int, rng: np.random.Generator
) -> np.ndarray:
    """Return mixed states ``A A^dag / Tr(A A^dag)`` with complex Gaussian ``A``."""
    dim = 2**n_qubits
    states = []
    for _ in range(n_states):
        matrix = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
        rho = matrix @ matrix.conj().T
        states.append(rho / np.trace(rho))
    return np.array(states)


def state_count(state_cfg: dict, n_qubits: int) -> int:
    """Return how many states :func:`build_states` produces, without building them.

    Used by ``utils/run_sweep.py`` to estimate a run's memory before launch
    (each state is a dense ``2**n x 2**n`` complex matrix).

    Raises
    ------
    ValueError
        If the set name is unknown.
    """
    name = state_cfg["set"]
    if name == "code_squared_loss":
        return 2 * 2**n_qubits + 2 + 3
    if name == "paper_squared_loss":
        return 2 * 2**n_qubits + 4 + 1 + len(state_cfg["p_values"]) + 1
    if name == "pauli_basis":
        return 3 * 2**n_qubits
    if name == "computational_basis":
        return 2**n_qubits
    if name == "haar":
        return int(state_cfg["n_states"])
    raise ValueError(f"state set must be one of {STATE_SETS}, got {name!r}")


def build_states(
    state_cfg: dict, n_qubits: int, rng: np.random.Generator
) -> np.ndarray:
    """Return the density matrices of one state set.

    Parameters
    ----------
    state_cfg : dict
        Key ``set`` (one of ``STATE_SETS``) plus ``n_states`` for ``haar`` and
        ``p_values`` for ``paper_squared_loss``.
    n_qubits : int
        Number of qubits.
    rng : numpy.random.Generator
        Stream for random states.

    Returns
    -------
    numpy.ndarray
        Complex array of shape ``(M, 2**n, 2**n)``.

    Raises
    ------
    ValueError
        If the set name is unknown.
    """
    name = state_cfg["set"]
    if name == "haar":
        return haar_pure_states(n_qubits, int(state_cfg["n_states"]), rng)
    # Fill a preallocated array instead of stacking a list: a list of 4 MiB
    # matrices plus its stacked copy doubled the peak memory at n = 9.
    states = np.empty(
        (state_count(state_cfg, n_qubits), 2**n_qubits, 2**n_qubits),
        dtype=np.complex128,
    )
    count = 0
    for count, state in enumerate(_iter_states(state_cfg, n_qubits, rng), start=1):
        states[count - 1] = state
    if count != len(states):
        raise RuntimeError(
            f"state_count disagrees with the {name!r} generator: {len(states)} vs {count}"
        )
    return states


def _iter_states(state_cfg: dict, n_qubits: int, rng: np.random.Generator):
    """Yield the density matrices of a non-random state set in their fixed order."""
    name = state_cfg["set"]
    dim = 2**n_qubits
    maximally_mixed = np.eye(dim, dtype=np.complex128) / dim
    if name == "code_squared_loss":
        yield from _product_basis([_KET_0, _KET_1], n_qubits)
        yield from _product_basis([_KET_PLUS, _KET_MINUS], n_qubits)
        yield _density(_ghz(n_qubits))
        yield maximally_mixed
        yield from ginibre_mixed_states(n_qubits, 3, rng)
        return
    if name == "paper_squared_loss":
        yield from _product_basis([_KET_0, _KET_1], n_qubits)
        yield from _product_basis([_KET_PLUS, _KET_MINUS], n_qubits)
        rest = _product([_KET_0] * (n_qubits - 2))
        bell_pairs = [
            (_product([_KET_0, _KET_0]) + _product([_KET_1, _KET_1])) / np.sqrt(2),
            (_product([_KET_0, _KET_0]) - _product([_KET_1, _KET_1])) / np.sqrt(2),
            (_product([_KET_0, _KET_1]) + _product([_KET_1, _KET_0])) / np.sqrt(2),
            (_product([_KET_0, _KET_1]) - _product([_KET_1, _KET_0])) / np.sqrt(2),
        ]
        for pair in bell_pairs:
            yield _density(np.kron(pair, rest))
        yield _density(_ghz(n_qubits))
        all_zero = _product([_KET_0] * n_qubits)
        all_one = _product([_KET_1] * n_qubits)
        for p in state_cfg["p_values"]:
            yield _density(np.sqrt(p) * all_zero + np.sqrt(1 - p) * all_one)
        yield maximally_mixed
        return
    if name == "computational_basis":
        # Classical data (E8): the 2^n diagonal basis states.
        yield from _product_basis([_KET_0, _KET_1], n_qubits)
        return
    if name == "pauli_basis":
        for basis in (
            [_KET_PLUS, _KET_MINUS],
            [_KET_PLUS_I, _KET_MINUS_I],
            [_KET_0, _KET_1],
        ):
            yield from _product_basis(basis, n_qubits)
        return
    raise ValueError(f"state set must be one of {STATE_SETS}, got {name!r}")


@dataclass
class Target:
    """Random target Hamiltonian ``H*(zeta)`` and its observable.

    Parameters
    ----------
    labels : list of str
        Pauli-string labels of the target terms.
    zeta : numpy.ndarray
        Target coefficients.
    observable : numpy.ndarray
        ``phi(H*)`` for regression or the labelling observable for
        classification.
    hamiltonian : numpy.ndarray
        ``H*`` itself.
    """

    labels: list[str]
    zeta: np.ndarray
    observable: np.ndarray
    hamiltonian: np.ndarray


def build_target(cfg: dict) -> Target:
    """Return the seeded random target of one configuration.

    Raises
    ------
    ValueError
        If ``data.label_rule`` is unknown.
    """
    data_cfg = cfg["data"]
    target_cfg = data_cfg["target"]
    n_qubits = int(data_cfg["n_qubits"])
    temperature = float(cfg["physics"]["temperature"])
    explicit_labels = (
        target_cfg["term_labels"] if target_cfg["term_set"] == "explicit" else None
    )
    labels, terms = build_terms(
        target_cfg["term_set"],
        n_qubits,
        bool(target_cfg["include_identity"]),
        explicit_labels,
    )
    rng = random_stream(cfg["seed"], STREAM_TARGET)
    # Same draw as the notebooks: (U[0, 1) - 0.5) * 4 = U[-2, 2).
    zeta = rng.uniform(
        float(target_cfg["low"]), float(target_cfg["high"]), size=len(labels)
    )
    hamiltonian = np.einsum("j,jab->ab", zeta, terms)
    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
    rule = data_cfg["label_rule"]
    if rule == "regression":
        function = make_activation(target_cfg["function"], temperature)
    elif rule == "sign_activation":
        function = make_activation("tanh", temperature)
    elif rule == "sign_linear":
        function = make_activation("linear", temperature)
    else:
        raise ValueError(f"label_rule must be one of {LABEL_RULES}, got {rule!r}")
    observable = matrix_function(function, eigenvalues, eigenvectors)
    return Target(labels, zeta, observable, hamiltonian)


def _targets_for(target: Target, states: np.ndarray, label_rule: str) -> np.ndarray:
    values = expectations(target.observable, states)
    if label_rule == "regression":
        return values
    return np.where(values >= 0.0, 1.0, -1.0)


def build_split(cfg: dict, split: str, target: Target | None = None) -> Dataset:
    """Return one split as a full-batch :class:`~lib.models.Dataset`.

    Raises
    ------
    ValueError
        If ``split`` is unknown.
    """
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    data_cfg = cfg["data"]
    n_qubits = int(data_cfg["n_qubits"])
    stream = STREAM_TRAIN_STATES if split == "train" else STREAM_VALIDATION_STATES
    states = build_states(
        data_cfg[f"{split}_states"], n_qubits, random_stream(cfg["seed"], stream)
    )
    target = target if target is not None else build_target(cfg)
    return Dataset(states, _targets_for(target, states, data_cfg["label_rule"]))


class FullBatchLoader:
    """Iterable that yields the whole split once per pass, as ``(states, targets)``.

    The paper trains with full-batch gradient descent (Eq. (119)), so one
    batch is the full split.
    """

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def __iter__(self):
        yield self.dataset.states, self.dataset.targets

    def __len__(self) -> int:
        return 1


def build_dataloader(cfg: dict, split: str) -> FullBatchLoader:
    """Return the full-batch loader of one split."""
    return FullBatchLoader(build_split(cfg, split))


def describe_dataset(cfg: dict, target: Target, splits: dict[str, Dataset]) -> dict:
    """Return the ``dataset`` block of ``run_status.json``."""
    data_cfg = cfg["data"]
    description = {
        "name": "synthetic quantum states, Sec. VI A (paper's own data model)",
        "splits": {
            name: {
                # The Phase 5 selection split is carved out of the training states.
                "state_set": data_cfg[
                    f"{'train' if name == 'selection' else name}_states"
                ]["set"],
                "n_states": int(len(data.targets)),
            }
            for name, data in splits.items()
        },
        "preprocessing": (
            f"target {data_cfg['target']['term_set']} n={data_cfg['n_qubits']}, "
            f"zeta~U[{data_cfg['target']['low']},{data_cfg['target']['high']}], "
            f"label_rule={data_cfg['label_rule']}"
        ),
        "subset": "full",
    }
    if data_cfg["label_rule"] != "regression":
        description["positive_label_fraction"] = {
            name: float(np.mean(data.targets > 0)) for name, data in splits.items()
        }
    return description


def reference_outputs_dir(cfg: dict) -> Path:
    """Return the cached directory of the authors' released ``outputs/`` CSVs.

    Raises
    ------
    FileNotFoundError
        If the directory is missing (see LOG.md Data Acquisition Log for the
        acquisition command).
    """
    path = Path(cfg["data_root"]) / PAPER_NAME / "reference_outputs" / "outputs"
    if not path.is_dir():
        raise FileNotFoundError(
            f"Reference outputs not found at {path}; run the acquisition command in LOG.md"
        )
    return path


def load_reference_curve(cfg: dict, filename: str) -> dict[str, np.ndarray]:
    """Read one of the authors' per-epoch CSVs into column arrays.

    Empty cells (accuracy is logged every 20 epochs) become NaN.
    """
    import csv

    path = reference_outputs_dir(cfg) / filename
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {
        column: np.array(
            [float(row[column]) if row[column] else np.nan for row in rows]
        )
        for column in rows[0]
    }
