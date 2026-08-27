"""Build the committed smoke-test subset from the full SEM dataset.

The full file ``data/bve_qnn/sem_supervised_dataset.npz`` is not stored in
git (Copernicus CDS regeneration, see ``generate_dataset.py``). Smoke tests
and the paper notebook use a tiny spatially/time-subsampled copy that keeps
the same key layout so ``lib/runner.py`` can still reshape predictions onto
``psi_qcl_training``.

Usage (from papers/bve_qnn, with the full dataset already generated)::

    python utils/make_subset.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
FULL_PATH = REPO_ROOT / "data" / "bve_qnn" / "sem_supervised_dataset.npz"
SUBSET_PATH = REPO_ROOT / "data" / "bve_qnn" / "sem_supervised_subset.npz"

# Two training hours and an 8 x 8 spatial grid -> 128 supervised points.
TIME_INDICES = (0, 7)
N_LAT = 8
N_LON = 8


def build_subset(full_path: Path = FULL_PATH, subset_path: Path = SUBSET_PATH) -> Path:
    """Write a reshape-compatible smoke subset of the full SEM dataset.

    Parameters
    ----------
    full_path : Path
        Path to the full ``sem_supervised_dataset.npz``. Default value is
        ``data/bve_qnn/sem_supervised_dataset.npz`` at the repository root.
    subset_path : Path
        Output path for the subset. Default value is
        ``data/bve_qnn/sem_supervised_subset.npz``.

    Returns
    -------
    Path
        Path of the written subset file.

    Raises
    ------
    FileNotFoundError
        If ``full_path`` does not exist.
    """
    if not full_path.exists():
        raise FileNotFoundError(
            f"Full SEM dataset not found at {full_path}. Generate it with "
            "notebooks/neutral_atom/quantum_bve_step_by_step.ipynb first."
        )

    data = np.load(full_path)
    n_time, n_lat, n_lon = data["psi_qcl_training"].shape
    features = data["supervised_features"].reshape(n_time, n_lat, n_lon, 4)
    targets = data["supervised_targets"].reshape(n_time, n_lat, n_lon)
    targets_norm = data["supervised_targets_normalized"].reshape(n_time, n_lat, n_lon)

    time_idx = np.array(TIME_INDICES)
    lat_idx = np.linspace(0, n_lat - 1, N_LAT, dtype=int)
    lon_idx = np.linspace(0, n_lon - 1, N_LON, dtype=int)

    psi = data["psi_qcl_training"][np.ix_(time_idx, lat_idx, lon_idx)]
    feat = features[np.ix_(time_idx, lat_idx, lon_idx)]
    tgt = targets[np.ix_(time_idx, lat_idx, lon_idx)]
    tgt_norm = targets_norm[np.ix_(time_idx, lat_idx, lon_idx)]

    subset_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        subset_path,
        supervised_features=feat.reshape(-1, 4),
        supervised_targets=tgt.reshape(-1),
        supervised_targets_normalized=tgt_norm.reshape(-1),
        supervised_target_mean=data["supervised_target_mean"],
        supervised_target_std=data["supervised_target_std"],
        training_hours=data["training_hours"][time_idx],
        training_time_normalized=data["training_time_normalized"][time_idx],
        lat_downsampled=data["lat_downsampled"][lat_idx],
        lon_downsampled=data["lon_downsampled"][lon_idx],
        psi_qcl_training=psi,
    )
    return subset_path


def main() -> None:
    path = build_subset()
    packed = np.load(path)
    print(
        f"wrote {path} ({path.stat().st_size} bytes)\n"
        f"  features {packed['supervised_features'].shape}\n"
        f"  psi_qcl_training {packed['psi_qcl_training'].shape}"
    )


if __name__ == "__main__":
    main()
