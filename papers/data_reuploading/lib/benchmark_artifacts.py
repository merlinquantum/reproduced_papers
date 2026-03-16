from __future__ import annotations

from typing import NamedTuple

import numpy as np


class GridSpec(NamedTuple):
    xlim: tuple[float, float]
    ylim: tuple[float, float]
    resolution: int


def compute_grid_spec(
    X_tr: np.ndarray, X_te: np.ndarray | None, resolution: int, margin: float = 0.15
) -> GridSpec:
    """Compute shared grid bounds for decision maps."""
    all_pts = np.vstack([X_tr, X_te]) if X_te is not None else X_tr
    span = all_pts.max(0) - all_pts.min(0)
    pad = margin * span
    xlim = (float(all_pts[:, 0].min() - pad[0]), float(all_pts[:, 0].max() + pad[0]))
    ylim = (float(all_pts[:, 1].min() - pad[1]), float(all_pts[:, 1].max() + pad[1]))
    return GridSpec(xlim=xlim, ylim=ylim, resolution=resolution)


def compute_decision_grid(model, grid_spec: GridSpec) -> np.ndarray:
    """Return the class-probability difference grid for contour plotting."""
    x = np.linspace(grid_spec.xlim[0], grid_spec.xlim[1], grid_spec.resolution)
    y = np.linspace(grid_spec.ylim[0], grid_spec.ylim[1], grid_spec.resolution)
    xx, yy = np.meshgrid(x, y)
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    prob = model.predict_proba(grid)
    diff = (prob[:, 1] - prob[:, 0]).reshape(xx.shape)
    return diff


def compute_probability_features(model, X_tr: np.ndarray, X_te: np.ndarray | None):
    """Return the first quantum feature axis for train/test points."""
    feat_tr = model.get_quantum_features(X_tr)[:, 0]
    feat_te = None
    if X_te is not None:
        feat_te = model.get_quantum_features(X_te)[:, 0]
    return feat_tr, feat_te
