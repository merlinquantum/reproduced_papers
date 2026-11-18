"""Classical node features for the Phase 5 fair-baseline controls.

These features are not part of the paper. They exist to test claim C3 (App.
D.2) against a confounder found in this reproduction: in the ladder-concat
dataset as constructed here, the two classes differ in their counts of
degree-2 and degree-4 nodes, so node degree alone carries the class signal.
"""

from __future__ import annotations

import numpy as np


def degree_one_hot_features(adjacency: np.ndarray, num_features: int) -> np.ndarray:
    """Encode each node's degree as a one-hot row.

    Column ``d`` is 1 for a node of degree ``d``. The width matches the
    paper's 20-channel positional encodings so that the degree control uses
    exactly the quantum model's GCN and parameter count.

    Parameters
    ----------
    adjacency : numpy.ndarray
        Symmetric 0/1 adjacency matrix of shape ``(N, N)``.
    num_features : int
        Number of one-hot columns; every degree must be below it.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(N, num_features)`` with one 1.0 per row.

    Raises
    ------
    ValueError
        If a node's degree does not fit in ``num_features`` columns.
    """
    degrees = np.rint(adjacency.sum(axis=1)).astype(int)
    if degrees.size and degrees.max() >= num_features:
        raise ValueError(
            f"node degree {degrees.max()} does not fit in {num_features} one-hot columns"
        )
    features = np.zeros((adjacency.shape[0], num_features), dtype=np.float64)
    features[np.arange(adjacency.shape[0]), degrees] = 1.0
    return features
