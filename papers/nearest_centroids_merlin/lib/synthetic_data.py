"""
Synthetic data generation for paper reproduction.

Generates synthetic datasets matching Section III.B.a of the paper:
- k clusters (k = 2 or 4)
- d dimensions (d = 4 or 8)
- n = 10 data points per centroid
- Minimum distance between centroids: 0.3
- Gaussian variance: 0.05
- Points distributed within sphere of radius 1
"""

from typing import Optional

import numpy as np


def generate_synthetic_data(
    n_clusters: int = 4,
    n_dimensions: int = 8,
    n_points_per_cluster: int = 10,
    min_centroid_distance: float = 0.3,
    gaussian_variance: float = 0.05,
    sphere_radius: float = 1.0,
    seed: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic clustered data as described in the paper.

    Parameters
    ----------
    n_clusters : int
        Number of clusters (k in paper). Default 4.
    n_dimensions : int
        Dimensionality of data points (d in paper). Default 8.
    n_points_per_cluster : int
        Points generated per cluster (n in paper). Default 10.
    min_centroid_distance : float
        Minimum Euclidean distance between any two centroids. Default 0.3.
    gaussian_variance : float
        Variance of Gaussian noise added to generate points. Default 0.05.
    sphere_radius : float
        Points are distributed within a sphere of this radius. Default 1.0.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    X : np.ndarray
        Data points of shape (n_clusters * n_points_per_cluster, n_dimensions)
    y : np.ndarray
        Labels of shape (n_clusters * n_points_per_cluster,)
    centroids : np.ndarray
        Generated centroids of shape (n_clusters, n_dimensions)
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate centroids with minimum distance constraint
    centroids = _generate_centroids(
        n_clusters, n_dimensions, min_centroid_distance, sphere_radius
    )

    # Generate points around each centroid
    X_list = []
    y_list = []

    for cluster_idx in range(n_clusters):
        centroid = centroids[cluster_idx]

        # Add Gaussian noise to centroid to generate points
        noise = np.random.normal(
            loc=0.0,
            scale=np.sqrt(gaussian_variance),
            size=(n_points_per_cluster, n_dimensions),
        )
        points = centroid + noise

        # Clip points to remain within sphere (optional, paper doesn't explicitly say)
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        mask = norms > sphere_radius
        if mask.any():
            points = np.where(mask, points * sphere_radius / norms, points)

        X_list.append(points)
        y_list.append(np.full(n_points_per_cluster, cluster_idx))

    X = np.vstack(X_list)
    y = np.hstack(y_list)

    return X, y, centroids


def _generate_centroids(
    n_clusters: int,
    n_dimensions: int,
    min_distance: float,
    sphere_radius: float,
    max_attempts: int = 10000,
) -> np.ndarray:
    """Generate centroids with minimum pairwise distance constraint."""
    centroids: list[np.ndarray] = []
    attempts = 0

    while len(centroids) < n_clusters and attempts < max_attempts:
        # Generate random point within sphere
        point = np.random.randn(n_dimensions)
        point = point / np.linalg.norm(point)  # Unit vector
        r = np.random.uniform(0, sphere_radius) ** (
            1.0 / n_dimensions
        )  # Uniform in volume
        point = point * r * sphere_radius

        # Check minimum distance to existing centroids
        valid = True
        for existing in centroids:
            if np.linalg.norm(point - existing) < min_distance:
                valid = False
                break

        if valid:
            centroids.append(point)

        attempts += 1

    if len(centroids) < n_clusters:
        raise ValueError(
            f"Could not generate {n_clusters} centroids with min distance "
            f"{min_distance} in {max_attempts} attempts. Try reducing min_distance."
        )

    return np.array(centroids)


def generate_paper_datasets():
    """
    Generate all synthetic datasets used in the paper.

    Returns dict with keys like 'Nq4_Nc2', 'Nq4_Nc4', 'Nq8_Nc2', 'Nq8_Nc4'
    """
    datasets = {}

    configs = [
        {"n_dimensions": 4, "n_clusters": 2, "label": "Nq4_Nc2"},
        {"n_dimensions": 4, "n_clusters": 4, "label": "Nq4_Nc4"},
        {"n_dimensions": 8, "n_clusters": 2, "label": "Nq8_Nc2"},
        {"n_dimensions": 8, "n_clusters": 4, "label": "Nq8_Nc4"},
    ]

    for cfg in configs:
        X, y, centroids = generate_synthetic_data(
            n_clusters=cfg["n_clusters"],
            n_dimensions=cfg["n_dimensions"],
            n_points_per_cluster=10,
            min_centroid_distance=0.3,
            gaussian_variance=0.05,
            sphere_radius=1.0,
            seed=123,
        )
        datasets[cfg["label"]] = {
            "X": X,
            "y": y,
            "centroids": centroids,
            "n_dimensions": cfg["n_dimensions"],
            "n_clusters": cfg["n_clusters"],
        }

    return datasets
