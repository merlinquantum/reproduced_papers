"""Figures of merit reported in the paper's Appendix (main.tex).

- MRE = mean(|psi_NN - psi_SEM| / median(|psi_SEM|)), per time snapshot.
- PPMCC computed per spatial grid point over the time dimension, summarized
  with the median over all grid points.
"""

from __future__ import annotations

import numpy as np


def mre_per_time_snapshot(psi_pred: np.ndarray, psi_ref: np.ndarray) -> np.ndarray:
    mre_values = []
    for k in range(psi_pred.shape[0]):
        psi_nn = psi_pred[k]
        psi_ref_k = psi_ref[k]
        ref_median_abs = np.median(np.abs(psi_ref_k))
        mre_values.append(np.mean(np.abs(psi_nn - psi_ref_k) / ref_median_abs))
    return np.array(mre_values)


def ppmcc_per_grid_point(psi_pred: np.ndarray, psi_ref: np.ndarray) -> np.ndarray:
    n_time = psi_pred.shape[0]
    psi_pred_flat = psi_pred.reshape(n_time, -1)
    psi_ref_flat = psi_ref.reshape(n_time, -1)

    ppmcc_values = []
    for j in range(psi_pred_flat.shape[1]):
        nn_series = psi_pred_flat[:, j]
        ref_series = psi_ref_flat[:, j]
        if np.std(nn_series) == 0 or np.std(ref_series) == 0:
            ppmcc_values.append(np.nan)
        else:
            ppmcc_values.append(np.corrcoef(nn_series, ref_series)[0, 1])
    return np.array(ppmcc_values)


def compute_figures_of_merit(
    psi_pred: np.ndarray, psi_ref: np.ndarray
) -> dict[str, object]:
    mre = mre_per_time_snapshot(psi_pred, psi_ref)
    ppmcc = ppmcc_per_grid_point(psi_pred, psi_ref)
    return {
        "mre_per_time": mre,
        "ppmcc_per_grid_point": ppmcc,
        "median_mre_percent": float(100 * np.median(mre)),
        "median_ppmcc": float(np.nanmedian(ppmcc)),
    }


__all__ = ["mre_per_time_snapshot", "ppmcc_per_grid_point", "compute_figures_of_merit"]
