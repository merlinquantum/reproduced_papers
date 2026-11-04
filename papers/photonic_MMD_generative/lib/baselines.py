"""Generative baselines and model samplers (LOG.md "Fair Baseline Plan").

Paper baselines (§6): RBM (``qml_benchmarks``, the original notebook's
settings), uniform fixed-weight random model, test-to-test MMD (two halves of
the test set). Added: B4 independent-marginal fixed-weight sampler.

Samplers of trained circuits: indistinguishable photons with Perceval's
Clifford2017 backend (exact boson sampling, the model's deployment), and
distinguishable photons (B5) by sampling each photon independently from its
single-photon output distribution |U[:, s]|^2.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def uniform_fixed_weight(m: int, n: int, num_samples: int, seed: int) -> np.ndarray:
    """Uniform samples over {x in {0,1}^m : |x| = n} (paper's random model)."""
    rng = np.random.default_rng(seed)
    X = np.zeros((num_samples, m), dtype=np.int64)
    X[
        np.arange(num_samples)[:, None],
        np.argsort(rng.random((num_samples, m)), axis=1)[:, :n],
    ] = 1
    return X


def independent_marginal_fixed_weight(
    X_train: np.ndarray, num_samples: int, seed: int
) -> np.ndarray:
    """B4: draw n distinct modes sequentially, each with probability ∝ training marginal.

    Uses the Gumbel top-n trick, which is equivalent to sequential sampling
    without replacement with weights mu_j (mode occupation frequencies).
    """
    rng = np.random.default_rng(seed)
    m, n = X_train.shape[1], int(X_train[0].sum())
    log_weights = np.log(X_train.mean(axis=0) + 1e-12)
    keys = log_weights[None, :] + rng.gumbel(size=(num_samples, m))
    X = np.zeros((num_samples, m), dtype=np.int64)
    X[np.arange(num_samples)[:, None], np.argsort(-keys, axis=1)[:, :n]] = 1
    return X


def test_halves(X_test: np.ndarray, seed: int):
    """Shuffle the test set and split it into two halves (test-to-test reference)."""
    perm = np.random.default_rng(seed).permutation(len(X_test))
    half = len(X_test) // 2
    return X_test[perm[:half]], X_test[perm[half : 2 * half]]


def train_rbm(X_train: np.ndarray, rbm_cfg: dict, seed: int):
    """Fit the paper's RBM benchmark (qml_benchmarks RestrictedBoltzmannMachine).

    ``rbm_cfg`` keys: ``n_components``, ``learning_rate``, ``batch_size``,
    ``n_iter``, ``mmd_samples``, ``gibbs_steps`` (original notebook: 64, 0.01,
    64, 2000, 1000, 10), and ``score_sigma`` (bandwidth of the RBM's internal
    MMD score, used only by its ``score`` method, not by our evaluation).
    """
    from qml_benchmarks.models.energy_based_model import RestrictedBoltzmannMachine

    rbm = RestrictedBoltzmannMachine(
        n_components=rbm_cfg["n_components"],
        learning_rate=rbm_cfg["learning_rate"],
        batch_size=rbm_cfg["batch_size"],
        n_iter=rbm_cfg["n_iter"],
        mmd_kwargs={
            "n_samples": rbm_cfg["mmd_samples"],
            "n_steps": rbm_cfg["gibbs_steps"],
            "sigma": [rbm_cfg["score_sigma"]],
        },
        score_fn="mmd",
        random_state=seed,
    )
    rbm.fit(np.asarray(X_train))
    return rbm


def rbm_samples(rbm, num_samples: int, gibbs_steps: int) -> np.ndarray:
    """Samples from a fitted RBM (not constrained to Hamming weight n)."""
    return np.asarray(rbm.sample(num_samples, gibbs_steps), dtype=np.int64)


def boson_samples(U: np.ndarray, modes, num_samples: int, seed: int) -> np.ndarray:
    """Exact boson samples of the trained interferometer (Perceval Clifford2017, CPU)."""
    import perceval as pcvl

    m = U.shape[0]
    backend = pcvl.Clifford2017Backend()
    backend.set_circuit(pcvl.components.Unitary(pcvl.Matrix(np.asarray(U))))
    occupied = {int(s) for s in modes}
    backend.set_input_state(
        pcvl.BasicState([1 if mode in occupied else 0 for mode in range(m)])
    )
    pcvl.random_seed(seed)
    return np.array([list(s) for s in backend.samples(num_samples)], dtype=np.int64)


def distinguishable_samples(
    U: np.ndarray, modes, num_samples: int, seed: int
) -> np.ndarray:
    """B5 samples: each photon s lands in mode j with probability |U[j, s]|^2, independently."""
    rng = np.random.default_rng(seed)
    U = np.asarray(U)
    X = np.zeros((num_samples, U.shape[0]), dtype=np.int64)
    for s in modes:
        probabilities = np.abs(U[:, int(s)]) ** 2
        probabilities = probabilities / probabilities.sum()
        landing = rng.choice(U.shape[0], size=num_samples, p=probabilities)
        np.add.at(X, (np.arange(num_samples), landing), 1)
    return X
