"""Synthetic datasets of Hoch et al. (2025), regenerated as `substitute-data`.

The authors' data are available only on request. The generators follow the
"Synthetic Data Specification" in LOG.md:

- ``nonlinear_1d`` (S1): 15 points x = 0..14, one per post-selected outcome; labels
  read from Fig. 5a, +1 for {0-3, 11-14} and -1 for {4-10}.
- ``moons_kmeans`` (S2): ``make_moons(200)`` discretised by K-means into 15
  clusters; each cluster is mapped to one ABS outcome index by an ``assignment``.

Every point carries an outcome index in 0..14, so a dataset is classified with
``K_points = K_outcome[index_i, index_j]``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_moons

from runtime_lib.data_paths import paper_data_dir

PAPER_NAME = "adaptive_boson_sampling"
N_OUTCOMES = 15
NONLINEAR_1D_POSITIVE = (0, 1, 2, 3, 11, 12, 13, 14)

# Scalar of a centroid (x, y) used to rank clusters for "centroid_rank"; the
# angle is taken about (0.5, 0.25), the centre of the make_moons point cloud.
CENTROID_RANK_RULES = {
    "x": lambda c: c[:, 0],
    "-x": lambda c: -c[:, 0],
    "y": lambda c: c[:, 1],
    "-y": lambda c: -c[:, 1],
    "x+y": lambda c: c[:, 0] + c[:, 1],
    "x-y": lambda c: c[:, 0] - c[:, 1],
    "angle": lambda c: np.arctan2(c[:, 1] - 0.25, c[:, 0] - 0.5),
}

_REQUIRED_KEYS = {
    "nonlinear_1d": ("name", "n_points"),
    "moons_kmeans": (
        "name",
        "n_samples",
        "noise",
        "data_seed",
        "n_clusters",
        "kmeans_seed",
        "assignment",
        "assignment_seed",
    ),
}


@dataclass
class OutcomeDataset:
    """Labelled points mapped to ABS outcome indices.

    Attributes
    ----------
    name : str
        Dataset name.
    outcome_index : numpy.ndarray
        int64 ``(N,)``; ABS outcome (kernel row) of each point.
    labels : numpy.ndarray
        int64 ``(N,)`` in {-1, +1}.
    features : numpy.ndarray
        float64 ``(N, f)`` classical features for baselines: x for 1D, the raw 2D
        point for moons.
    extras : dict
        Generator byproducts (cluster ids, centroids, assignment, purity).
    """

    name: str
    outcome_index: np.ndarray
    labels: np.ndarray
    features: np.ndarray
    extras: dict[str, Any] = field(default_factory=dict)


def validate_dataset_config(dataset_cfg: dict[str, Any]) -> None:
    """Fail clearly when a required dataset key is missing or unknown.

    Raises
    ------
    ValueError
        For an unknown dataset name or a missing key.
    """
    name = dataset_cfg.get("name")
    if name not in _REQUIRED_KEYS:
        raise ValueError(
            f"unknown dataset {name!r}; expected one of {sorted(_REQUIRED_KEYS)}"
        )
    missing = [key for key in _REQUIRED_KEYS[name] if key not in dataset_cfg]
    if missing:
        raise ValueError(f"dataset {name!r} config is missing keys {missing}")


def make_nonlinear_1d(n_points: int) -> OutcomeDataset:
    """Return the 15-point 1D dataset of Fig. 5a (not linearly separable)."""
    if n_points != N_OUTCOMES:
        raise ValueError(f"nonlinear_1d has exactly {N_OUTCOMES} points (Fig. 5a)")
    positions = np.arange(n_points, dtype=np.int64)
    labels = np.where(np.isin(positions, NONLINEAR_1D_POSITIVE), 1, -1).astype(np.int64)
    return OutcomeDataset(
        name="nonlinear_1d",
        outcome_index=positions.copy(),
        labels=labels,
        features=positions.astype(np.float64)[:, None],
    )


def arc_order_assignment(
    centroids: np.ndarray, centroid_labels: np.ndarray
) -> np.ndarray:
    """Order clusters along the two moon arcs: upper moon left to right, then lower.

    Neighbouring outcome indices then map to neighbouring clusters, matching the
    Gaussian-like (banded) structure of the B2/B3 kernels. Clusters are split by
    majority label; within each moon they are sorted by the angle of the centroid
    around that moon's centre (make_moons: upper centre (0, 0), lower (1, 0.5)).

    Returns
    -------
    numpy.ndarray
        ``assignment[c]`` = outcome index of cluster ``c``.
    """
    upper = np.flatnonzero(centroid_labels == -1)
    lower = np.flatnonzero(centroid_labels == 1)
    upper_angle = np.arctan2(centroids[upper, 1] - 0.0, centroids[upper, 0] - 0.0)
    lower_angle = np.arctan2(centroids[lower, 1] - 0.5, centroids[lower, 0] - 1.0)
    # Upper moon runs from angle pi (left) to 0 (right); lower from pi to 2 pi.
    upper_sorted = upper[np.argsort(-upper_angle)]
    lower_sorted = lower[np.argsort(np.mod(lower_angle, 2 * np.pi))]
    ordered = np.concatenate([upper_sorted, lower_sorted])
    assignment = np.empty(len(centroids), dtype=np.int64)
    assignment[ordered] = np.arange(len(centroids))
    return assignment


def make_moons_kmeans(dataset_cfg: dict[str, Any]) -> OutcomeDataset:
    """Return make_moons points discretised into ABS outcome indices (Fig. 5d-e).

    ``assignment`` is ``"arc_order"``, ``"x_order"`` (clusters sorted by centroid
    x-coordinate), ``"centroid_rank"`` (sorted by ``CENTROID_RANK_RULES[rank_rule]``
    of the centroid), ``"kmeans_label"`` (identity on the K-means label) or
    ``"random"`` (permutation seeded by ``assignment_seed``).
    """
    n_clusters = dataset_cfg["n_clusters"]
    if n_clusters != N_OUTCOMES:
        raise ValueError(f"moons_kmeans needs {N_OUTCOMES} clusters, one per outcome")
    points, raw_labels = make_moons(
        n_samples=dataset_cfg["n_samples"],
        noise=dataset_cfg["noise"],
        random_state=dataset_cfg["data_seed"],
    )
    # make_moons label 0 is the upper moon; store it as -1.
    labels = np.where(raw_labels == 1, 1, -1).astype(np.int64)
    kmeans = KMeans(
        n_clusters=n_clusters, n_init=10, random_state=dataset_cfg["kmeans_seed"]
    ).fit(points)
    cluster = kmeans.labels_.astype(np.int64)
    centroids = kmeans.cluster_centers_.astype(np.float64)

    majority = np.empty(n_clusters, dtype=np.int64)
    purity = np.empty(n_clusters)
    for index in range(n_clusters):
        members = labels[cluster == index]
        if members.size == 0:
            raise ValueError(f"K-means cluster {index} is empty")
        positive_fraction = float(np.mean(members == 1))
        majority[index] = 1 if positive_fraction >= 0.5 else -1
        purity[index] = max(positive_fraction, 1 - positive_fraction)

    strategy = dataset_cfg["assignment"]
    if strategy == "arc_order":
        assignment = arc_order_assignment(centroids, majority)
    elif strategy in ("x_order", "centroid_rank"):
        # Outcome index increases with a scalar of the cluster centroid: a
        # parameter-free 2D -> 1D ordering. The paper does not state its rule;
        # these variants were added in Phase 03 as diagnosis candidates.
        # "x_order" is "centroid_rank" with rank_rule "x".
        rule = "x" if strategy == "x_order" else dataset_cfg["rank_rule"]
        if rule not in CENTROID_RANK_RULES:
            raise ValueError(
                f"unknown rank_rule {rule!r}; expected one of "
                f"{sorted(CENTROID_RANK_RULES)}"
            )
        scores = CENTROID_RANK_RULES[rule](centroids)
        assignment = np.empty(n_clusters, dtype=np.int64)
        assignment[np.argsort(scores, kind="stable")] = np.arange(n_clusters)
    elif strategy == "kmeans_label":
        assignment = np.arange(n_clusters, dtype=np.int64)
    elif strategy == "random":
        rng = np.random.default_rng(dataset_cfg["assignment_seed"])
        assignment = rng.permutation(n_clusters).astype(np.int64)
    else:
        raise ValueError(f"unknown assignment {strategy!r}")

    return OutcomeDataset(
        name="moons_kmeans",
        outcome_index=assignment[cluster],
        labels=labels,
        features=points.astype(np.float64),
        extras={
            "cluster": cluster,
            "centroids": centroids,
            "assignment": assignment,
            "cluster_majority": majority,
            "cluster_purity": purity,
            # Weighted purity bounds the accuracy of any cluster-index classifier.
            "weighted_purity": float(
                np.sum(purity * np.bincount(cluster, minlength=n_clusters))
                / len(cluster)
            ),
        },
    )


def build_dataset(dataset_cfg: dict[str, Any]) -> OutcomeDataset:
    """Validate a dataset config and generate the dataset."""
    validate_dataset_config(dataset_cfg)
    if dataset_cfg["name"] == "nonlinear_1d":
        return make_nonlinear_1d(dataset_cfg["n_points"])
    return make_moons_kmeans(dataset_cfg)


def partition_indices(
    n_points: int, test_fraction: float, partition_seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return (train, test) indices of one random unstratified partition.

    The same ``partition_seed`` gives the same split for every model, so all
    kernels in a comparison see identical partitions. Test size is
    ``round(test_fraction * n_points)``: 3 of 15 and 40 of 200 for 0.2.
    """
    n_test = int(round(test_fraction * n_points))
    if not 0 < n_test < n_points:
        raise ValueError(f"test_fraction {test_fraction} leaves an empty split")
    order = np.random.default_rng(partition_seed).permutation(n_points)
    return np.sort(order[n_test:]), np.sort(order[:n_test])


def build_dataloader(
    cfg: dict[str, Any], split: str, partition_seed: int = 0
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Return one full batch ``(outcome_index, labels, features)`` for a split.

    Kernel SVMs train on the whole split at once, so the loader yields a single
    batch. ``split`` is ``"train"``, ``"test"`` or ``"all"``.
    """
    dataset = build_dataset(cfg["dataset"])
    if split == "all":
        indices = np.arange(len(dataset.labels))
    else:
        train, test = partition_indices(
            len(dataset.labels), cfg["evaluation"]["test_fraction"], partition_seed
        )
        if split == "train":
            indices = train
        elif split == "test":
            indices = test
        else:
            raise ValueError(f"unknown split {split!r}")
    return [
        (
            dataset.outcome_index[indices],
            dataset.labels[indices],
            dataset.features[indices],
        )
    ]


def digitized_kernels_path(cfg: dict[str, Any]) -> Path:
    """Return ``<data_root>/adaptive_boson_sampling/digitized/figure_kernels.npz``.

    The data root comes from ``--data-root``, then ``DATA_DIR``, then the
    repository ``data/`` folder (shared runtime convention).
    """
    return (
        paper_data_dir(PAPER_NAME, data_root=cfg.get("data_root"))
        / "digitized"
        / "figure_kernels.npz"
    )
