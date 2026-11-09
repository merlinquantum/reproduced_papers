"""B6: tuned RBM baseline for the Fig. 3a comparison (LOG.md "Phase 5 Comparison Plan").

Independent control, not part of the paper reproduction. The paper's RBM
(``baselines.train_rbm``) is kept unchanged; this module adds the fair
version declared in Phase 5:

- trained on binarized data, since Bernoulli visible units cannot represent
  the photon counts >= 2 present in boson-sampling samples;
- grid over hidden units (64, 256 and the iso-parameter 512 at m = 256) and
  learning rate, with Gibbs chain length read at several points of the same
  chains;
- selection per kernel bandwidth on a validation split carved from the
  training set, then a refit of the selected setting on the full training set.

Training drives sklearn's ``BernoulliRBM.partial_fit`` over the same minibatch
slices as ``fit`` so that every epoch can be logged; the resulting weights are
identical to ``fit`` (tested).
"""

from __future__ import annotations

import logging
import time

import numpy as np
from sklearn.utils import gen_even_slices

from . import estimator

logger = logging.getLogger(__name__)


def num_parameters(m: int, hidden: int) -> int:
    """Weights plus visible and hidden biases of an m x hidden RBM."""
    return m * hidden + m + hidden


def validation_split(X_train: np.ndarray, validation_size: int, split_seed: int):
    """Fixed split of the training set into (fit, validation) parts.

    The permutation depends only on ``split_seed``, so every seed of the sweep
    tunes on the same validation set. The test set is never touched.
    """
    if not 0 < validation_size < len(X_train):
        raise ValueError(
            f"validation_size must be in (0, {len(X_train)}), got {validation_size}."
        )
    perm = np.random.default_rng(split_seed).permutation(len(X_train))
    return X_train[perm[validation_size:]], X_train[perm[:validation_size]]


def train_rbm_logged(
    X: np.ndarray, hidden: int, learning_rate: float, rbm_cfg: dict, seed: int, tag: str
):
    """Fit a ``qml_benchmarks`` RBM epoch by epoch, logging each epoch.

    Parameters
    ----------
    X : numpy.ndarray
        Binary training data, shape (num_samples, m).
    hidden : int
        Number of hidden units.
    learning_rate : float
        Contrastive-divergence learning rate.
    rbm_cfg : dict
        ``batch_size`` and ``n_iter`` (epochs).
    seed : int
        ``random_state`` of the RBM (weights init, Gibbs chains, sampling).
    tag : str
        Label used in log lines.

    Returns
    -------
    tuple
        Fitted RBM and training seconds.
    """
    from qml_benchmarks.models.energy_based_model import RestrictedBoltzmannMachine

    X = np.asarray(X, dtype=np.float64)
    if X.max() > 1:
        raise ValueError("B6 trains on binarized data; got values > 1.")
    batch_size, epochs = rbm_cfg["batch_size"], rbm_cfg["n_iter"]
    rbm = RestrictedBoltzmannMachine(
        n_components=hidden,
        learning_rate=learning_rate,
        batch_size=batch_size,
        n_iter=epochs,
        random_state=seed,
    )
    rbm.initialize(X)
    # Same minibatch slicing as sklearn's BernoulliRBM.fit (no shuffling).
    n_batches = int(np.ceil(len(X) / batch_size))
    slices = list(gen_even_slices(n_batches * batch_size, n_batches, n_samples=len(X)))
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        for batch in slices:
            rbm.partial_fit(X[batch])
        logger.info(
            "TRAIN_EPOCH_COMPLETED | model=%s | epoch=%d/%d | seconds=%.1f",
            tag,
            epoch,
            epochs,
            time.perf_counter() - start,
        )
    return rbm, time.perf_counter() - start


def gibbs_chain_samples(rbm, num_samples: int, read_steps) -> dict:
    """Samples of one set of Gibbs chains read after each step count in ``read_steps``.

    Chains start from uniform random bits drawn with the RBM's own generator,
    as ``RestrictedBoltzmannMachine.sample`` does.
    """
    read_steps = sorted(int(s) for s in read_steps)
    v = rbm.rng.choice([0, 1], size=(num_samples, rbm.dim))
    out, done = {}, 0
    for target in read_steps:
        for _ in range(target - done):
            v = rbm.gibbs(v)
        done = target
        out[target] = np.asarray(v, dtype=np.int64)
    return out


def tune_and_refit(
    X_train: np.ndarray, tuning_cfg: dict, sigmas, seed: int
) -> tuple[dict, dict]:
    """Run the declared B6 protocol for one seed.

    Parameters
    ----------
    X_train : numpy.ndarray
        Training split (raw occupations); binarized here.
    tuning_cfg : dict
        ``hidden_units``, ``learning_rates``, ``gibbs_steps``,
        ``validation_size``, ``split_seed``, ``num_samples``, ``batch_size``,
        ``n_iter``.
    sigmas : sequence of float
        Gaussian bandwidths; one selection per bandwidth.
    seed : int
        Run seed (RBM ``random_state``).

    Returns
    -------
    tuple of dict
        ``record`` (grid scores, selections, timings, all JSON-serializable)
        and ``samples`` mapping ``"sigma<sigma>"`` to the refit model's
        samples with the selected Gibbs steps.
    """
    X_bin = np.minimum(np.asarray(X_train), 1)
    m = X_bin.shape[1]
    X_fit, X_val = validation_split(
        X_bin, tuning_cfg["validation_size"], tuning_cfg["split_seed"]
    )
    num = tuning_cfg["num_samples"]
    grid = []
    for hidden in tuning_cfg["hidden_units"]:
        for lr in tuning_cfg["learning_rates"]:
            tag = f"rbm_h{hidden}_lr{lr:g}_fit"
            rbm, seconds = train_rbm_logged(X_fit, hidden, lr, tuning_cfg, seed, tag)
            start = time.perf_counter()
            chains = gibbs_chain_samples(rbm, num, tuning_cfg["gibbs_steps"])
            validation = {
                str(steps): {
                    str(float(s)): estimator.gaussian_mmd_unbiased(
                        samples, X_val, float(s)
                    )
                    for s in sigmas
                }
                for steps, samples in chains.items()
            }
            grid.append(
                {
                    "hidden": hidden,
                    "learning_rate": lr,
                    "num_parameters": num_parameters(m, hidden),
                    "train_seconds": seconds,
                    "sample_and_score_seconds": time.perf_counter() - start,
                    "validation_gaussian_mmd_binarized": validation,
                }
            )
            logger.info(
                "EVALUATION_COMPLETED | split=validation | model=%s | %s",
                tag,
                " | ".join(
                    f"gibbs{k}:" + ",".join(f"{v:.4g}" for v in d.values())
                    for k, d in validation.items()
                ),
            )
    selections = {}
    for s in sigmas:
        key = str(float(s))
        candidates = [
            (
                entry["validation_gaussian_mmd_binarized"][str(steps)][key],
                entry["num_parameters"],
                int(steps),
                entry,
            )
            for entry in grid
            for steps in sorted(tuning_cfg["gibbs_steps"])
        ]
        best = min(candidates, key=lambda c: c[:3])
        selections[key] = {
            "hidden": best[3]["hidden"],
            "learning_rate": best[3]["learning_rate"],
            "gibbs_steps": best[2],
            "num_parameters": best[1],
            "validation_mmd": best[0],
        }
        logger.info(
            "SELECTION_COMPLETED | sigma=%s | hidden=%d | lr=%g | gibbs=%d | validation_mmd=%.6g",
            key,
            best[3]["hidden"],
            best[3]["learning_rate"],
            best[2],
            best[0],
        )
    refits, samples = {}, {}
    for key, sel in selections.items():
        setting = (sel["hidden"], sel["learning_rate"])
        if setting not in refits:
            tag = f"rbm_h{setting[0]}_lr{setting[1]:g}_refit"
            rbm, seconds = train_rbm_logged(X_bin, *setting, tuning_cfg, seed, tag)
            refits[setting] = {
                "rbm": rbm,
                "seconds": seconds,
                "chains": gibbs_chain_samples(rbm, num, tuning_cfg["gibbs_steps"]),
            }
        sel["refit_seconds"] = refits[setting]["seconds"]
        samples[f"sigma{key}"] = refits[setting]["chains"][sel["gibbs_steps"]]
    record = {
        "fit_size": int(len(X_fit)),
        "validation_size": int(len(X_val)),
        "split_seed": tuning_cfg["split_seed"],
        "grid": grid,
        "selected": selections,
        "num_refits": len(refits),
        "total_train_seconds": float(
            sum(e["train_seconds"] for e in grid)
            + sum(r["seconds"] for r in refits.values())
        ),
    }
    return record, samples
