"""Model registry: ABS quantum kernels and classical baselines as SVM inputs.

A model spec in the config expands into one or more *instances* (for example one
per Haar ``U_0`` seed or one per alternative outcome assignment). Every instance
exposes ``fit_one(train, test, arm, partition_seed)`` so all models run through
``evaluation.evaluate_over_partitions`` on identical partitions.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import cache, partial
from pathlib import Path
from typing import Any

import numpy as np

from .abs_merlin import simulated_state_table
from .circuits import get_platform
from .data import OutcomeDataset, digitized_kernels_path
from .evaluation import fit_feature_model, fit_precomputed
from .kernels import (
    gaussian_kernel,
    load_digitized_kernel,
    median_heuristic_gamma,
    quantum_kernel,
    random_rank_d_kernel,
    states_for_assignment,
    u0_for_config,
)

N_OUTCOMES = 15


@dataclass
class ModelInstance:
    """One concrete model evaluated over all partitions.

    Attributes
    ----------
    label : str
        Model label from the config.
    instance : dict
        Instance identity (seed, assignment, ...), recorded in metrics.
    fit_one : Callable
        ``(train, test, arm, partition_seed) -> (train_acc, test_acc, params)``.
    outcome_kernel : numpy.ndarray | None
        15x15 outcome kernel when the model is a kernel over outcomes.
    details : dict
        Extra provenance (U_0 selection, post-selection probabilities).
    """

    label: str
    instance: dict[str, Any]
    fit_one: Callable[[np.ndarray, np.ndarray, str, int], tuple[float, float, dict]]
    outcome_kernel: np.ndarray | None = None
    details: dict[str, Any] = field(default_factory=dict)


def _precomputed_instance(
    label: str,
    instance: dict[str, Any],
    outcome_kernel: np.ndarray,
    dataset: OutcomeDataset,
    evaluation_cfg: dict[str, Any],
    details: dict[str, Any] | None = None,
) -> ModelInstance:
    if outcome_kernel.shape != (N_OUTCOMES, N_OUTCOMES):
        raise ValueError(f"{label}: outcome kernel must be 15x15")
    point_kernel = outcome_kernel[np.ix_(dataset.outcome_index, dataset.outcome_index)]
    fit = partial(
        fit_precomputed,
        point_kernel,
        dataset.labels,
        evaluation_cfg=evaluation_cfg,
    )

    def fit_one(train, test, arm, partition_seed):
        return fit(train, test, arm, partition_seed=partition_seed)

    return ModelInstance(label, instance, fit_one, outcome_kernel, details or {})


def _assignments(spec: dict[str, Any]) -> list[tuple[str, np.ndarray]]:
    """Return (name, assignment) pairs: identity, or seeded random permutations."""
    if "assignment" not in spec:
        raise ValueError(
            f"{spec['label']}: 'assignment' is required ('identity' or "
            "{'n_assignments': N, 'seed': S})"
        )
    assignment = spec["assignment"]
    if assignment == "identity":
        return [("identity", np.arange(N_OUTCOMES))]
    rng = np.random.default_rng(assignment["seed"])
    return [
        (f"random_{index}", rng.permutation(N_OUTCOMES))
        for index in range(assignment["n_assignments"])
    ]


@cache
def _platform_states(
    platform_name: str,
    seed: int | None,
    u0_json: str,
    digitized_path: str,
    simulation_json: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return the cached state table of a platform for one U_0 choice.

    U_0 selection does not depend on the dataset, so moons runs with several
    dataset variants and arms reuse it instead of re-searching the Haar pool.
    """
    platform = get_platform(platform_name)
    u0_cfg = json.loads(u0_json)
    simulation = json.loads(simulation_json)
    if u0_cfg["strategy"] == "fixed_mesh":
        states, probabilities = simulated_state_table(platform, None, simulation)
        return states, probabilities, {"strategy": "fixed_mesh"}
    u0, record = u0_for_config(platform, seed, u0_cfg, Path(digitized_path))
    states, probabilities = simulated_state_table(platform, u0, simulation)
    return states, probabilities, record


def build_abs_simulated(
    spec: dict[str, Any],
    dataset: OutcomeDataset,
    evaluation_cfg: dict[str, Any],
    digitized_path: str,
) -> list[ModelInstance]:
    """Expand a simulated ABS kernel spec over U_0 seeds and assignments.

    ``digitized_path`` is read only by the ``fit_digitized`` U_0 strategy.
    """
    platform = get_platform(spec["platform"])
    if len(platform.outcomes) != N_OUTCOMES:
        raise ValueError(
            f"platform {platform.name} encodes {len(platform.outcomes)} "
            "outcomes; classification needs 15"
        )
    if platform.needs_random_u0:
        u0_cfg = spec["u0"]
        seeds = u0_cfg["seeds"]
    else:
        seeds, u0_cfg = [None], {"strategy": "fixed_mesh"}

    # Phase 3 specs have no "simulation" key and use the numpy reference.
    simulation = spec.get("simulation", {"backend": "numpy"})
    instances = []
    for seed in seeds:
        states, probabilities, record = _platform_states(
            platform.name,
            seed,
            json.dumps(u0_cfg, sort_keys=True),
            digitized_path,
            json.dumps(simulation, sort_keys=True),
        )
        for assignment_name, assignment in _assignments(spec):
            selected = states_for_assignment(states, assignment)
            kernel = quantum_kernel(selected, spec["kernel_kind"])
            instances.append(
                _precomputed_instance(
                    spec["label"],
                    {"u0_seed": seed, "assignment": assignment_name},
                    kernel,
                    dataset,
                    evaluation_cfg,
                    {
                        "u0_selection": record,
                        "simulation": simulation,
                        "assignment": assignment.tolist(),
                        "postselection_probability": [
                            float(probabilities[observed, circuit])
                            for circuit, observed in enumerate(assignment)
                        ],
                    },
                )
            )
    return instances


def build_instances(
    spec: dict[str, Any],
    dataset: OutcomeDataset,
    cfg: dict[str, Any],
) -> list[ModelInstance]:
    """Expand one model spec into instances.

    Raises
    ------
    ValueError
        For an unknown ``kind`` or a spec that does not fit the dataset.
    """
    kind = spec["kind"]
    evaluation_cfg = cfg["evaluation"]
    label = spec["label"]
    if kind == "abs_simulated":
        return build_abs_simulated(
            spec, dataset, evaluation_cfg, str(digitized_kernels_path(cfg))
        )
    if kind == "abs_digitized":
        kernel = load_digitized_kernel(digitized_kernels_path(cfg), spec["key"])
        return [
            _precomputed_instance(
                label, {"key": spec["key"]}, kernel, dataset, evaluation_cfg,
                {"source": "digitised from paper figure (substitute-data)"},
            )
        ]  # fmt: skip
    if kind == "index_gaussian":
        # Classical Gaussian kernel over the 15 outcomes, on the outcome index or
        # on the cluster centroid each outcome represents (moons).
        if spec["source"] == "outcome_index":
            outcome_features = np.arange(N_OUTCOMES, dtype=float)[:, None]
        elif spec["source"] == "centroid":
            if "centroids" not in dataset.extras:
                raise ValueError(f"{label}: centroid source needs moons_kmeans")
            centroids = dataset.extras["centroids"]
            outcome_features = np.empty_like(centroids)
            outcome_features[dataset.extras["assignment"]] = centroids
        else:
            raise ValueError(f"{label}: unknown source {spec['source']!r}")
        gamma = median_heuristic_gamma(outcome_features)
        base = gaussian_kernel(outcome_features, gamma)
        instances = []
        for assignment_name, assignment in _assignments(spec):
            permuted = base[np.ix_(assignment, assignment)]
            instances.append(
                _precomputed_instance(
                    label,
                    {"assignment": assignment_name},
                    permuted,
                    dataset,
                    evaluation_cfg,
                    {"gamma": gamma, "assignment": assignment.tolist()},
                )
            )
        return instances
    if kind == "random_rank_d":
        return [
            _precomputed_instance(
                label,
                {"seed": seed, "dimension": spec["dimension"]},
                random_rank_d_kernel(N_OUTCOMES, spec["dimension"], seed),
                dataset,
                evaluation_cfg,
            )
            for seed in spec["seeds"]
        ]
    if kind == "feature":
        fit = partial(
            fit_feature_model,
            spec,
            dataset.features,
            dataset.labels,
            evaluation_cfg=evaluation_cfg,
        )

        def fit_one(train, test, arm, partition_seed):
            return fit(train, test, arm, partition_seed=partition_seed)

        return [ModelInstance(label, {"estimator": spec["estimator"]}, fit_one)]
    raise ValueError(f"unknown model kind {kind!r}")
