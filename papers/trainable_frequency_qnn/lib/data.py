"""Datasets for the trainable-frequency reproduction.

Two data sources sit behind ``build_dataloader(cfg, split)``:

- ``cosine_series`` (paper Sec. V, Eq. 4): synthetic target
  ``y(x) = mean_{w in Omega_d} cos(w x)`` on ``[-4 pi, 4 pi]`` with
  ``n_d = ceil(2 |D| max(Omega_d))`` equally spaced training points.
- ``cylinder_wake`` (paper Sec. VI): the Raissi et al. NekTar solution of the
  Re = 100 wake behind a cylinder, ``cylinder_nektar_wake.mat`` under
  ``<data_root>/trainable_frequency_qnn/``.

The data root follows the shared runtime: ``--data-root`` / ``cfg["data_root"]``,
then the ``DATA_DIR`` environment variable, then ``<repo>/data``.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from runtime_lib.data_paths import resolve_data_root
from runtime_lib.dtypes import dtype_torch

PAPER_DATA_DIRNAME = "trainable_frequency_qnn"


def resolve_torch_dtype(cfg: dict[str, Any]) -> torch.dtype:
    """Return the real torch dtype of the run, failing if it is not set."""
    dtype = dtype_torch(cfg.get("dtype"))
    if dtype not in (torch.float32, torch.float64):
        raise ValueError(
            f"Config 'dtype' must be float32 or float64, got {cfg.get('dtype')!r}."
        )
    return dtype


# Grid of cylinder_nektar_wake.mat: X_star rows are ordered x-fastest,
# row = iy * N_X + ix, with 100 x-values in [1, 8] and 50 y-values in [-2, 2].
N_X = 100
N_Y = 50


def cosine_series_points(
    frequencies: list[float], domain_pi: list[float], n_points: int
):
    """Sample the paper's cosine series on ``n_points`` equally spaced points.

    Parameters
    ----------
    frequencies : list of float
        Data frequencies Omega_d.
    domain_pi : list of float
        Domain bounds in units of pi, ``[-4, 4]`` in the paper.
    n_points : int
        Number of points, both endpoints included.

    Returns
    -------
    tuple of numpy.ndarray
        ``(x, y)`` arrays of shape ``(n_points,)``.
    """
    x = np.linspace(domain_pi[0] * math.pi, domain_pi[1] * math.pi, n_points)
    y = np.mean([np.cos(w * x) for w in frequencies], axis=0)
    return x, y


def nyquist_point_count(frequencies: list[float], domain_pi: list[float]) -> int:
    """Return ``n_d = ceil(2 |D| max(Omega_d))`` from paper Sec. V."""
    domain_length = (domain_pi[1] - domain_pi[0]) * math.pi
    return math.ceil(2 * domain_length * max(frequencies))


def _cosine_series_split(dataset_cfg: dict[str, Any], split: str):
    frequencies = [float(w) for w in dataset_cfg["frequencies"]]
    domain_pi = dataset_cfg["domain_pi"]
    n_train = nyquist_point_count(frequencies, domain_pi)
    if split == "train":
        return cosine_series_points(frequencies, domain_pi, n_train)
    if split == "dense":
        return cosine_series_points(
            frequencies, domain_pi, dataset_cfg["dense_factor"] * n_train
        )
    raise ValueError(f"cosine_series has splits 'train' and 'dense', not {split!r}.")


def cylinder_wake_path(cfg: dict[str, Any]) -> Path:
    """Return the verified local path of ``cylinder_nektar_wake.mat``."""
    data_root = resolve_data_root(cfg.get("data_root"))
    path = data_root / PAPER_DATA_DIRNAME / cfg["dataset"]["file"]
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Re-download it with the command in README.md (Data section)."
        )
    return path


def load_cylinder_wake(cfg: dict[str, Any]) -> dict[str, np.ndarray]:
    """Load the wake solution restricted to the configured time window.

    Returns
    -------
    dict of numpy.ndarray
        ``x`` and ``y`` of shape ``(5000,)``, ``t`` of shape ``(T,)`` and
        ``u``, ``v``, ``p`` of shape ``(5000, T)``.
    """
    from scipy.io import loadmat

    raw = loadmat(cylinder_wake_path(cfg))
    times = raw["t"].ravel()
    t_min, t_max = cfg["dataset"]["time_window"]
    keep = (times >= t_min - 1e-9) & (times <= t_max + 1e-9)
    return {
        "x": raw["X_star"][:, 0],
        "y": raw["X_star"][:, 1],
        "t": times[keep],
        "u": raw["U_star"][:, 0, keep],
        "v": raw["U_star"][:, 1, keep],
        "p": raw["p_star"][:, keep],
    }


def coordinate_bounds(wake: dict[str, np.ndarray]) -> np.ndarray:
    """Return ``[[min, max]]`` for x, y and t over the loaded window."""
    return np.array(
        [
            [wake["x"].min(), wake["x"].max()],
            [wake["y"].min(), wake["y"].max()],
            [wake["t"].min(), wake["t"].max()],
        ]
    )


def transform_coordinates(
    coords: np.ndarray, transform: str, bounds: np.ndarray
) -> np.ndarray:
    """Map physical (x, y, t) to the model inputs.

    ``raw`` feeds physical coordinates, as written in the paper's circuit
    ``R_y(theta_F x)``. ``minmax`` maps each coordinate linearly to [-1, 1]. The
    paper does not say which one it used (LOG.md, A7). With ``minmax`` the PDE
    residual must be computed in physical units, so ``NavierStokesTask`` applies
    the chain-rule factors from :func:`input_scale_factors`.
    """
    if transform == "raw":
        return coords
    if transform == "minmax":
        return 2 * (coords - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0]) - 1
    raise ValueError(
        f"Unknown coordinate_transform {transform!r}; expected 'raw' or 'minmax'."
    )


def input_scale_factors(transform: str, bounds: np.ndarray) -> np.ndarray:
    """Return d(model input)/d(physical coordinate) for x, y and t."""
    if transform == "raw":
        return np.ones(3)
    return 2 / (bounds[:, 1] - bounds[:, 0])


def data_subgrid_rows(stride: int) -> np.ndarray:
    """Return X_star rows of the sparse supervised sub-grid.

    The paper gives a 10 x 5 grid of cells (1 % of the 100 x 50 grid) without
    their positions (LOG.md, A8). We take every ``stride``-th grid line centred
    in the domain: ix = 5, 15, ..., 95 and iy = 5, 15, ..., 45 for stride 10.
    """
    x_index = np.arange(stride // 2, N_X, stride)
    y_index = np.arange(stride // 2, N_Y, stride)
    return (y_index[:, None] * N_X + x_index[None, :]).ravel()


def _wake_split(cfg: dict[str, Any], split: str) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(inputs, targets)`` with columns (x, y, t) and (u, v, p)."""
    dataset_cfg = cfg["dataset"]
    wake = load_cylinder_wake(cfg)
    bounds = coordinate_bounds(wake)
    n_points, n_times = wake["u"].shape

    if split == "collocation":
        rows = np.arange(n_points)
        time_index = np.arange(n_times)
    elif split == "data":
        rows = data_subgrid_rows(dataset_cfg["data_stride"])
        time_index = np.arange(n_times)
    elif split == "evaluation":
        rows = np.arange(n_points)
        eval_times = np.asarray(dataset_cfg["evaluation_times"])
        time_index = np.array(
            [int(np.argmin(np.abs(wake["t"] - t))) for t in eval_times]
        )
        if not np.allclose(wake["t"][time_index], eval_times):
            raise ValueError(
                f"Evaluation times {eval_times} are not on the data time grid."
            )
    else:
        raise ValueError(
            f"cylinder_wake has splits 'collocation', 'data', 'evaluation', not {split!r}."
        )

    # Time-major ordering keeps each evaluation snapshot contiguous.
    grid_rows = np.tile(rows, len(time_index))
    grid_times = np.repeat(time_index, len(rows))
    coords = np.stack(
        [wake["x"][grid_rows], wake["y"][grid_rows], wake["t"][grid_times]], axis=1
    )
    targets = np.stack(
        [
            wake["u"][grid_rows, grid_times],
            wake["v"][grid_rows, grid_times],
            wake["p"][grid_rows, grid_times],
        ],
        axis=1,
    )
    inputs = transform_coordinates(coords, dataset_cfg["coordinate_transform"], bounds)
    return inputs, targets


def build_dataset(cfg: dict[str, Any], split: str) -> TensorDataset:
    """Build the tensors of one split.

    Parameters
    ----------
    cfg : dict
        Resolved configuration with ``dataset.name`` and ``dtype``.
    split : str
        ``train`` / ``dense`` for ``cosine_series``; ``collocation`` / ``data`` /
        ``evaluation`` for ``cylinder_wake``.

    Returns
    -------
    torch.utils.data.TensorDataset
        ``(inputs, targets)`` with inputs of shape ``(n, n_inputs)`` and
        targets of shape ``(n,)`` (cosine series) or ``(n, 3)`` (u, v, p).
    """
    dtype = resolve_torch_dtype(cfg)
    name = cfg["dataset"]["name"]
    if name == "cosine_series":
        x, y = _cosine_series_split(cfg["dataset"], split)
        inputs, targets = x[:, None], y
    elif name == "cylinder_wake":
        inputs, targets = _wake_split(cfg, split)
    else:
        raise ValueError(
            f"Unknown dataset {name!r}; expected 'cosine_series' or 'cylinder_wake'."
        )
    return TensorDataset(
        torch.as_tensor(inputs, dtype=dtype), torch.as_tensor(targets, dtype=dtype)
    )


def build_dataloader(cfg: dict[str, Any], split: str) -> DataLoader:
    """Return a DataLoader for one split.

    Training splits (``train``, ``collocation``, ``data``) are shuffled with a
    seeded generator and batched with ``training.batch_size``, so each epoch
    draws points without replacement (LOG.md, A5). Evaluation splits are
    returned unshuffled in one or more fixed batches of ``evaluation.batch_size``.
    """
    dataset = build_dataset(cfg, split)
    if split in ("train", "collocation", "data"):
        generator = torch.Generator().manual_seed(
            int(cfg["seed"]) + _SPLIT_SEED_OFFSET[split]
        )
        return DataLoader(
            dataset,
            batch_size=cfg["training"]["batch_size"],
            shuffle=True,
            generator=generator,
            drop_last=False,
        )
    return DataLoader(
        dataset, batch_size=cfg["evaluation"]["batch_size"], shuffle=False
    )


# Distinct sampling streams per split, independent of the ansatz initialisation.
_SPLIT_SEED_OFFSET = {"train": 1, "collocation": 2, "data": 3}
