"""Per-shot Aer checks of the authors' circuits (App. A; Sec. 6.3).

For random encoded pairs on ``n`` qubits per party:

1. Noiseless: ancilla-zero counts of the distributed and centralised circuits
   against ``Binomial(shots, 1/2 + 1/2 |<a|b>|^2)``. A chi-square over pairs
   tests the law used by the ``accuracy`` experiment for both columns.
2. Noisy: counts of the distributed circuit under the authors' depolarizing
   model against the exact noisy probability of :mod:`lib.noise`; the same
   statistic against the noiseless probability shows that the check can tell
   them apart.
3. Client messages: fraction of 1s in each classical bit sent by Alice and
   Bob for different inputs; teleportation makes them uniform (0.5).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy import stats

from .noise import noisy_zero_probability
from .protocol import (
    centralised_circuit,
    client_bit_counts,
    distributed_circuit,
    independent_seed,
    run_ancilla_zeros,
)

logger = logging.getLogger(__name__)


def _random_states(
    rng: np.random.Generator, n_states: int, n_qubits: int
) -> np.ndarray:
    states = rng.normal(size=(n_states, 2**n_qubits))
    return states / np.linalg.norm(states, axis=1, keepdims=True)


def _chi_square(
    zeros: np.ndarray, probability: np.ndarray, shots: int
) -> dict[str, Any]:
    expected = shots * probability
    variance = shots * probability * (1 - probability)
    z_scores = (zeros - expected) / np.sqrt(variance)
    statistic = float(np.sum(z_scores**2))
    return {
        "z_scores": z_scores.tolist(),
        "chi_square": statistic,
        "dof": int(len(zeros)),
        "p_value": float(stats.chi2.sf(statistic, len(zeros))),
    }


def check_protocol(cfg: dict[str, Any]) -> dict[str, Any]:
    """Run the three checks; settings from ``cfg["experiment"]``."""

    experiment = cfg["experiment"]
    rng = np.random.default_rng(np.random.SeedSequence([int(cfg["seed"]), 3]))
    n_qubits, n_pairs, shots = (
        int(experiment["n_qubits"]),
        int(experiment["n_pairs"]),
        int(experiment["shots"]),
    )
    alice, bob = (
        _random_states(rng, n_pairs, n_qubits),
        _random_states(rng, n_pairs, n_qubits),
    )
    ideal = 0.5 + 0.5 * np.einsum("pi,pi->p", alice, bob) ** 2
    logger.info(
        "TRAINING_STARTED | checks=noiseless,noisy,client_bits | n_qubits=%d | pairs=%d | shots=%d | no training",
        n_qubits,
        n_pairs,
        shots,
    )

    distributed, centralised = np.zeros(n_pairs), np.zeros(n_pairs)
    for pair in range(n_pairs):
        distributed[pair] = run_ancilla_zeros(
            distributed_circuit(alice[pair], bob[pair]),
            shots,
            0.0,
            seed=independent_seed(1, pair),
        )
        centralised[pair] = run_ancilla_zeros(
            centralised_circuit(alice[pair], bob[pair]),
            shots,
            0.0,
            seed=independent_seed(2, pair),
        )
        logger.info(
            "TRAIN_PAIR_COMPLETED | check=noiseless | pair=%d/%d | p0_ideal=%.4f | distributed=%.4f | centralised=%.4f",
            pair + 1,
            n_pairs,
            ideal[pair],
            distributed[pair] / shots,
            centralised[pair] / shots,
        )
    noiseless = {
        "p0_ideal": ideal.tolist(),
        "distributed_zero_fraction": (distributed / shots).tolist(),
        "centralised_zero_fraction": (centralised / shots).tolist(),
        "distributed_vs_ideal": _chi_square(distributed, ideal, shots),
        "centralised_vs_ideal": _chi_square(centralised, ideal, shots),
    }
    logger.info(
        "EVALUATION_COMPLETED | check=noiseless | distributed p=%.3f | centralised p=%.3f",
        noiseless["distributed_vs_ideal"]["p_value"],
        noiseless["centralised_vs_ideal"]["p_value"],
    )

    noisy: dict[str, Any] = {}
    for rate_index, rate in enumerate(experiment["noise_rates"]):
        predicted = np.array(
            [
                noisy_zero_probability(alice[p : p + 1], bob[p : p + 1], float(rate))[
                    0, 0
                ]
                for p in range(n_pairs)
            ]
        )
        observed = np.zeros(n_pairs)
        for pair in range(n_pairs):
            observed[pair] = run_ancilla_zeros(
                distributed_circuit(alice[pair], bob[pair]),
                shots,
                float(rate),
                seed=independent_seed(3 + rate_index, pair),
            )
            logger.info(
                "TRAIN_PAIR_COMPLETED | check=noisy | rate=%s | pair=%d/%d | predicted=%.4f | aer=%.4f",
                rate,
                pair + 1,
                n_pairs,
                predicted[pair],
                observed[pair] / shots,
            )
        noisy[str(rate)] = {
            "p0_predicted": predicted.tolist(),
            "aer_zero_fraction": (observed / shots).tolist(),
            "aer_vs_predicted": _chi_square(observed, predicted, shots),
            "aer_vs_noiseless": _chi_square(observed, ideal, shots),
        }
        logger.info(
            "EVALUATION_COMPLETED | check=noisy | rate=%s | vs predicted p=%.3f | vs noiseless p=%.3g",
            rate,
            noisy[str(rate)]["aer_vs_predicted"]["p_value"],
            noisy[str(rate)]["aer_vs_noiseless"]["p_value"],
        )

    bits: dict[str, Any] = {}
    for pair in range(min(2, n_pairs)):
        fractions = client_bit_counts(
            distributed_circuit(alice[pair], bob[pair]),
            shots,
            seed=independent_seed(9, pair),
        )
        bits[str(pair)] = {
            party: values.tolist() for party, values in fractions.items()
        }
    all_fractions = np.concatenate(
        [np.concatenate(list(map(np.asarray, b.values()))) for b in bits.values()]
    )
    # Binomial standard error of a fraction at 0.5.
    bits["max_deviation_from_half"] = float(np.max(np.abs(all_fractions - 0.5)))
    bits["standard_error"] = float(0.5 / np.sqrt(shots))
    logger.info(
        "EVALUATION_COMPLETED | check=client_bits | max |fraction - 0.5|=%.4f | standard error=%.4f",
        bits["max_deviation_from_half"],
        bits["standard_error"],
    )
    return {"noiseless": noiseless, "noisy": noisy, "client_bits": bits}
