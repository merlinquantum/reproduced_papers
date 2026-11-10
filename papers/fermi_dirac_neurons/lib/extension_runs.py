"""Runner tasks of the user-requested extensions E10 (gradient variance) and E12 (contextuality).

Both tasks are evaluation-only (no training loop). Each returns
``(metrics, dataset_description)`` for :func:`lib.runner.train_and_evaluate`
and logs every evaluated cell as ``EVALUATION_STARTED`` /
``EVALUATION_COMPLETED`` events.

Config blocks (see ``utils/generate_configs.py``)::

    "gradient_variance": {
        "term_set", "include_identity", "n_qubits", "init_range",
        "temperatures": {"fixed": [...], "sqrt_n_factors": [...]},
        "inputs": [...], "parameter_positions": ["first", "middle", "last"],
        "draws", "target_range",
        "data_variance_check": null | {"temperature", "parameter_position", "n_states"}
    }

    "contextuality": {
        "temperatures": {"low", "high", "count"},          # log-spaced grid
        "invariance": null | {"n_qubits": [...], "sets_per_kind", "sizes": [...],
                              "locality", "activations": [...], "epsilon", "theta_range"},
        "graded": null | {"families": [...], "n_qubits": [...], "draws", "theta_range",
                          "epsilons": [...], "activation", "graded_max_support"}
    }
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from . import contextuality as cx
from . import trainability as tr
from .pauli import term_labels

logger = logging.getLogger(__name__)

STREAM_GRADIENT_VARIANCE = 200
STREAM_CONTEXTUALITY = 300
STREAM_COMPRESSIBILITY = 400


def parameter_indices(n_terms: int, positions: list[str]) -> list[int]:
    """Map ``first`` / ``middle`` / ``last`` to term indices.

    Raises
    ------
    ValueError
        If a position name is unknown.
    """
    lookup = {"first": 0, "middle": n_terms // 2, "last": n_terms - 1}
    unknown = set(positions) - set(lookup)
    if unknown:
        raise ValueError(
            f"Unknown parameter positions {sorted(unknown)}; use {sorted(lookup)}"
        )
    return sorted({lookup[position] for position in positions})


def run_gradient_variance(cfg: dict, run_dir: Path) -> tuple[dict, dict]:
    """E10: variance of d_j f and d_j L over random parameters for one Hamiltonian family."""
    block = cfg["gradient_variance"]
    n_qubits = int(block["n_qubits"])
    explicit_labels = block["term_labels"] if block["term_set"] == "explicit" else None
    labels = term_labels(
        block["term_set"], n_qubits, bool(block["include_identity"]), explicit_labels
    )
    indices = parameter_indices(len(labels), block["parameter_positions"])
    rng = np.random.default_rng([int(cfg["seed"]), STREAM_GRADIENT_VARIANCE])
    draws = int(block["draws"])
    logger.info(
        "DATASET_READY | random parameters theta ~ U[-%g, %g] and inputs %s (no dataset) | term_set=%s | n_qubits=%d | J=%d | parameters=%s | draws=%d",
        block["init_range"],
        block["init_range"],
        ",".join(block["inputs"]),
        block["term_set"],
        n_qubits,
        len(labels),
        ",".join(labels[index] for index in indices),
        draws,
    )
    logger.info("EVALUATION_STARTED | quantity=gradient_variance | draws=%d", draws)

    def log_draw(draw: int) -> None:
        logger.info("DRAW_COMPLETED | draw=%d/%d", draw, draws)

    records = tr.variance_cell_sweep(
        block["term_set"],
        bool(block["include_identity"]),
        n_qubits,
        float(block["init_range"]),
        block["temperatures"],
        list(block["inputs"]),
        indices,
        draws,
        rng,
        target_range=float(block["target_range"]),
        on_draw=log_draw,
        explicit_labels=explicit_labels,
    )
    for record in records:
        logger.info(
            "EVALUATION_COMPLETED | quantity=%s | T=%s (%s) | input=%s | parameter=%s | variance=%.6e | ci95=[%.3e, %.3e]",
            record["quantity"],
            f"{record['temperature']:g}",
            record["temperature_rule"],
            record["input"],
            record["parameter_label"],
            record["variance"],
            *record["variance_ci95"],
        )
    data_check = None
    if block["data_variance_check"] is not None:
        check_cfg = block["data_variance_check"]
        index = parameter_indices(len(labels), [check_cfg["parameter_position"]])[0]
        logger.info(
            "EVALUATION_STARTED | quantity=haar_data_variance | n_states=%d",
            check_cfg["n_states"],
        )
        data_check = tr.data_variance_check(
            block["term_set"],
            bool(block["include_identity"]),
            n_qubits,
            float(block["init_range"]),
            float(check_cfg["temperature"]),
            index,
            int(check_cfg["n_states"]),
            rng,
            explicit_labels,
        )
        logger.info(
            "EVALUATION_COMPLETED | quantity=haar_data_variance | predicted=%.6e | empirical=%.6e",
            data_check["predicted_variance"],
            data_check["empirical_variance"],
        )
    metrics = {
        "task": "gradient_variance",
        "experiment": cfg["experiment"]["id"],
        "seed": cfg["seed"],
        "term_labels": labels,
        "records": records,
        "haar_data_variance_check": data_check,
    }
    dataset = {
        "name": "none (random parameters and input ensembles)",
        "splits": "n/a",
        "preprocessing": "none",
        "subset": f"inputs {','.join(block['inputs'])}",
    }
    return metrics, dataset


def _temperature_grid(grid_cfg: dict) -> list[float]:
    return np.geomspace(
        float(grid_cfg["low"]), float(grid_cfg["high"]), int(grid_cfg["count"])
    ).tolist()


def _invariance(
    block: dict, temperatures: list[float], rng: np.random.Generator
) -> list[dict]:
    results = []
    for n_qubits in block["n_qubits"]:
        for kind, sampler in (
            ("noncontextual", cx.random_noncontextual_labels),
            ("contextual", cx.random_contextual_labels),
        ):
            for index in range(int(block["sets_per_kind"])):
                size = int(block["sizes"][index % len(block["sizes"])])
                labels = sampler(int(n_qubits), size, int(block["locality"]), rng)
                theta = rng.uniform(
                    -block["theta_range"], block["theta_range"], size=len(labels)
                )
                for activation in block["activations"]:
                    records = cx.temperature_scan(
                        labels,
                        theta,
                        temperatures,
                        [float(block["epsilon"])],
                        activation=activation,
                        graded_max_support=0,
                    )
                    flags = [record["noncontextual"] for record in records]
                    results.append(
                        {
                            "n_qubits": int(n_qubits),
                            "set_kind": kind,
                            "set_index": index,
                            "labels": labels,
                            "activation": activation,
                            "term_set_noncontextual": cx.is_noncontextual(labels),
                            "flag_changed_at_some_T": len(set(flags)) > 1
                            or flags[0] != cx.is_noncontextual(labels),
                            "support_sizes": [
                                record["support_size"] for record in records
                            ],
                        }
                    )
                    logger.info(
                        "EVALUATION_COMPLETED | quantity=invariance | n=%d | set=%s#%d | activation=%s | flag_changed=%s | support=%d..%d",
                        n_qubits,
                        kind,
                        index,
                        activation,
                        results[-1]["flag_changed_at_some_T"],
                        min(results[-1]["support_sizes"]),
                        max(results[-1]["support_sizes"]),
                    )
    return results


def _graded(
    block: dict, temperatures: list[float], rng: np.random.Generator
) -> list[dict]:
    results = []
    for family in block["families"]:
        for n_qubits in block["n_qubits"]:
            labels = term_labels(family, int(n_qubits), False)
            for draw in range(int(block["draws"])):
                theta = rng.uniform(
                    -block["theta_range"], block["theta_range"], size=len(labels)
                )
                records = cx.temperature_scan(
                    labels,
                    theta,
                    temperatures,
                    [float(value) for value in block["epsilons"]],
                    activation=block["activation"],
                    graded_max_support=int(block["graded_max_support"]),
                )
                for record in records:
                    results.append(
                        {
                            "family": family,
                            "n_qubits": int(n_qubits),
                            "draw": draw,
                            **record,
                        }
                    )
                logger.info(
                    "EVALUATION_COMPLETED | quantity=graded | family=%s | n=%d | draw=%d | max_support=%d | always_contextual=%s",
                    family,
                    n_qubits,
                    draw,
                    max(record["support_size"] for record in records),
                    not any(record["noncontextual"] for record in records),
                )
    return results


def run_contextuality_scan(cfg: dict, run_dir: Path) -> tuple[dict, dict]:
    """E12 (amended): Kirby-Love invariance test and graded contextual fraction vs T."""
    block = cfg["contextuality"]
    temperatures = _temperature_grid(block["temperatures"])
    rng = np.random.default_rng([int(cfg["seed"]), STREAM_CONTEXTUALITY])
    logger.info(
        "DATASET_READY | random Pauli term sets and coefficients (no dataset) | temperatures=%d in [%g, %g]",
        len(temperatures),
        temperatures[0],
        temperatures[-1],
    )
    metrics = {
        "task": "contextuality_scan",
        "experiment": cfg["experiment"]["id"],
        "seed": cfg["seed"],
        "temperatures": temperatures,
        "invariance": None,
        "graded": None,
    }
    if block["invariance"] is not None:
        logger.info("EVALUATION_STARTED | quantity=invariance")
        metrics["invariance"] = _invariance(block["invariance"], temperatures, rng)
    if block["graded"] is not None:
        logger.info("EVALUATION_STARTED | quantity=graded")
        metrics["graded"] = _graded(block["graded"], temperatures, rng)
    dataset = {
        "name": "none (random Pauli term sets and coefficients)",
        "splits": "n/a",
        "preprocessing": "none",
        "subset": "n/a",
    }
    return metrics, dataset


def run_pauli_compressibility(cfg: dict, run_dir: Path) -> tuple[dict, dict]:
    """E15 part 3: Pauli strings needed to compute Tr[g_T(H) rho] within epsilon.

    For each draw of theta ~ U[-a, a] the activation observable is expanded
    in all 4^n Pauli strings (fast transform) and K_eps is the smallest
    number of non-identity strings whose dropped coefficients sum to at most
    eps, which bounds the error on the output for every input state.
    """
    from .activations import make_activation, matrix_function
    from .pauli import build_terms

    block = cfg["pauli_compressibility"]
    n_qubits = int(block["n_qubits"])
    explicit_labels = block["term_labels"] if block["term_set"] == "explicit" else None
    labels, terms = build_terms(block["term_set"], n_qubits, False, explicit_labels)
    activation = make_activation(block["activation"], float(block["temperature"]))
    rng = np.random.default_rng([int(cfg["seed"]), STREAM_COMPRESSIBILITY])
    logger.info(
        "DATASET_READY | random parameters (no dataset) | term_set=%s | n_qubits=%d | J=%d | kirby_love_noncontextual=%s | commuting=%s",
        block["term_set"],
        n_qubits,
        len(labels),
        cx.is_noncontextual(labels),
        bool(cx.commutation_matrix(labels).all()),
    )
    records = []
    for theta_range in block["theta_ranges"]:
        logger.info(
            "EVALUATION_STARTED | quantity=paulis_needed | theta_range=%g", theta_range
        )
        for draw in range(int(block["draws"])):
            theta = rng.uniform(-theta_range, theta_range, size=len(labels))
            eigenvalues, eigenvectors = np.linalg.eigh(
                np.einsum("j,jab->ab", theta, terms)
            )
            coefficients = cx.fast_pauli_expansion(
                matrix_function(activation, eigenvalues, eigenvectors)
            )
            record = {
                "theta_range": float(theta_range),
                "draw": draw,
                "nonzero_strings": int(np.sum(np.abs(coefficients[1:]) > 1e-12)),
                "paulis_needed": {
                    f"{epsilon:g}": cx.paulis_needed(coefficients, float(epsilon))
                    for epsilon in block["epsilons"]
                },
                "l1_norm": float(np.sum(np.abs(coefficients[1:]))),
            }
            records.append(record)
            logger.info(
                "EVALUATION_COMPLETED | quantity=paulis_needed | theta_range=%g | draw=%d | nonzero=%d | needed=%s",
                theta_range,
                draw,
                record["nonzero_strings"],
                record["paulis_needed"],
            )
    metrics = {
        "task": "pauli_compressibility",
        "experiment": cfg["experiment"]["id"],
        "seed": cfg["seed"],
        "term_labels": labels,
        "kirby_love_noncontextual": cx.is_noncontextual(labels),
        "commuting": bool(cx.commutation_matrix(labels).all()),
        "total_strings": 4**n_qubits - 1,
        "records": records,
    }
    dataset = {
        "name": "none (random parameters)",
        "splits": "n/a",
        "preprocessing": "none",
        "subset": "n/a",
    }
    return metrics, dataset
