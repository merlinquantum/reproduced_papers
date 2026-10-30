"""Synthetic beta-mixing VARMA windows and label functionals (paper Sec. 4.1, App. B).

The generator replays the random draws of the authors' code (commit ``8f02ff1``,
``src/data/generate/varma.py``, ``beta_mixing.py`` and ``label/functionals.py``) so
that seed 12345 reproduces their released dataset exactly. The order of the draws
from one ``numpy.random.default_rng(seed)`` stream is:

1. ``p`` AR directions (d x d normals), then ``p`` AR weights (uniform);
2. ``q`` MA directions (d x d normals);
3. the innovations, shape ``(T, d)``;
4. at the first window, the label vectors: ``u`` (current-point readout),
   ``u`` (exponential fading), ``u`` then ``v`` orthogonal to it (Volterra).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from runtime_lib.data_paths import resolve_data_root

# Labels in the authors' order, followed by our variant of F1 as the paper text
# describes it (u . X_{t+1}, a genuine one-step forecast).
TASK_NAMES = (
    "current_point_readout",
    "exp_fading",
    "volterra",
    "one_step_forecast",
)


@dataclass
class WindowsDataset:
    """Windows, labels and the fixed train/test split.

    Attributes
    ----------
    windows : numpy.ndarray
        Input windows of shape ``(N, w, d)``; ``windows[i, -1]`` is the most recent point.
    labels : numpy.ndarray
        Labels of shape ``(L, N)`` in the order of ``task_names``.
    task_names : tuple[str, ...]
        Name of each label row.
    train_idx, test_idx : numpy.ndarray
        Index arrays of the train/test split (random permutation, as in the code).
    meta : dict
        Generation parameters and label vectors.
    """

    windows: np.ndarray
    labels: np.ndarray
    task_names: tuple[str, ...]
    train_idx: np.ndarray
    test_idx: np.ndarray
    meta: dict


def _unit_spectral_direction(rng: np.random.Generator, dim: int) -> np.ndarray:
    matrix = rng.normal(size=(dim, dim))
    return matrix / np.linalg.svd(matrix, compute_uv=False)[0]


def _unit_vector(rng: np.random.Generator, dim: int) -> np.ndarray:
    vector = rng.normal(size=dim)
    return vector / np.linalg.norm(vector)


def _unit_vector_orthogonal_to(rng: np.random.Generator, u: np.ndarray) -> np.ndarray:
    vector = rng.normal(size=u.shape[0])
    vector = vector - np.dot(u, vector) * u
    return vector / np.linalg.norm(vector)


def draw_varma_parameters(rng, *, input_dim, ar_order, ma_order, gamma, eta, rho):
    """Draw a stable VARMA(p, q) parameter set (App. B.1).

    Returns
    -------
    tuple[list[numpy.ndarray], list[numpy.ndarray]]
        AR matrices ``Phi_1..Phi_p`` with ``sum ||Phi_i||_2 = gamma`` and MA matrices
        ``Theta_1..Theta_q`` with ``||Theta_j||_2 = eta * rho**(j-1)``.
    """
    directions = [_unit_spectral_direction(rng, input_dim) for _ in range(ar_order)]
    weights = rng.uniform(0.0, 1.0, size=ar_order)
    ar_weights = gamma * weights / weights.sum()
    ar = [ar_weights[i] * directions[i] for i in range(ar_order)]
    ma = [
        eta * rho ** (j - 1) * _unit_spectral_direction(rng, input_dim)
        for j in range(1, ma_order + 1)
    ]
    return ar, ma


def varma_step(ar, ma, latent, innovations, t):
    """Return ``Z_t`` from past latents and innovations (missing past terms are zero)."""
    value = innovations[t].copy()
    for i, phi in enumerate(ar, start=1):
        if t - i >= 0:
            value += phi @ latent[t - i]
    for j, theta in enumerate(ma, start=1):
        if t - j >= 0:
            value += theta @ innovations[t - j]
    return value


def generate_dataset(data_cfg: dict, seed: int) -> WindowsDataset:
    """Generate the strided windows dataset and its labels.

    Parameters
    ----------
    data_cfg : dict
        The ``data`` section of the resolved config.
    seed : int
        Seed of the data stream (12345 reproduces the authors' release).

    Returns
    -------
    WindowsDataset
        Windows, labels for ``TASK_NAMES`` and the train/test split.
    """
    num_windows = int(data_cfg["num_windows"])
    window = int(data_cfg["window"])
    input_dim = int(data_cfg["input_dim"])
    gap = int(data_cfg["gap"])
    burn_in = int(data_cfg["burn_in"])
    varma = data_cfg["varma"]
    alpha = float(data_cfg["alpha"])

    rng = np.random.default_rng(seed)
    ar, ma = draw_varma_parameters(
        rng,
        input_dim=input_dim,
        ar_order=int(varma["ar_order"]),
        ma_order=int(varma["ma_order"]),
        gamma=float(varma["gamma"]),
        eta=float(varma["eta"]),
        rho=float(varma["rho"]),
    )
    # The authors' code calls the gap "s": windows are w long with a stride of w + gap.
    stride = window + gap
    length = burn_in + window + (num_windows - 1) * stride
    innovations = rng.normal(
        0.0, float(varma["innovation_scale"]), size=(length, input_dim)
    )
    latent = np.zeros((length + 1, input_dim))
    for t in range(length):
        latent[t] = varma_step(ar, ma, latent, innovations, t)

    # One extra step for the one-step-forecast label of the last window. Its
    # innovation comes from a separate stream so the authors' draws stay unchanged.
    extra_innovation = np.random.default_rng([seed, 1]).normal(
        0.0, float(varma["innovation_scale"]), size=(1, input_dim)
    )
    latent[length] = varma_step(
        ar, ma, latent, np.vstack([innovations, extra_innovation]), length
    )
    path = np.tanh(latent)

    ends = burn_in + (window - 1) + stride * np.arange(num_windows)
    windows = np.stack([path[end - window + 1 : end + 1] for end in ends])

    u_readout = _unit_vector(rng, input_dim)
    u_fading = _unit_vector(rng, input_dim)
    u_volterra = _unit_vector(rng, input_dim)
    v_volterra = _unit_vector_orthogonal_to(rng, u_volterra)

    decay = alpha ** np.arange(window)
    newest_first = windows[:, ::-1, :]
    fading_u = (newest_first @ u_fading) @ decay
    fading_uv = (newest_first @ u_volterra) @ decay
    fading_v = (newest_first @ v_volterra) @ decay
    labels = np.stack(
        [
            windows[:, -1, :] @ u_readout,
            fading_u,
            fading_uv + 0.5 * fading_v**2,
            path[ends + 1] @ u_readout,
        ]
    )

    split_rng = np.random.default_rng(int(data_cfg["split_seed"]))
    permutation = split_rng.permutation(num_windows)
    num_test = int(float(data_cfg["test_ratio"]) * num_windows)
    meta = {
        "seed": seed,
        "stride": stride,
        "length": length,
        "u_readout": u_readout.tolist(),
        "u_fading": u_fading.tolist(),
        "u_volterra": u_volterra.tolist(),
        "v_volterra": v_volterra.tolist(),
    }
    return WindowsDataset(
        windows=windows,
        labels=labels,
        task_names=TASK_NAMES,
        train_idx=permutation[num_test:],
        test_idx=permutation[:num_test],
        meta=meta,
    )


def release_dataset_path(data_cfg: dict, data_root: str | None) -> Path:
    """Path of the authors' released dataset under the shared data root.

    ``data_root`` is the runtime's resolved ``--data-root`` (or ``DATA_DIR``); when
    ``None``, :func:`runtime_lib.data_paths.resolve_data_root` applies its defaults.
    """
    return resolve_data_root(data_root) / data_cfg["release_relpath"]


def check_against_release(
    dataset: WindowsDataset, data_cfg: dict, data_root: str | None
) -> dict:
    """Compare a seed-12345 dataset with the authors' released ``.npz``.

    Returns
    -------
    dict
        Maximum absolute differences of windows and of the three released labels.

    Raises
    ------
    FileNotFoundError
        If the release file is not under the data root.
    """
    path = release_dataset_path(data_cfg, data_root)
    if not path.is_file():
        raise FileNotFoundError(f"Authors' release dataset not found: {path}")
    release = np.load(path, allow_pickle=False)
    return {
        "release_path": str(path),
        "max_abs_diff_windows": float(np.max(np.abs(release["X"] - dataset.windows))),
        "max_abs_diff_labels": float(np.max(np.abs(release["y"] - dataset.labels[:3]))),
    }


def build_dataloader(cfg: dict, split: str) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield ``(windows, labels)`` batches of one split.

    Parameters
    ----------
    cfg : dict
        Resolved config; uses ``seed``, ``data`` and ``data.batch_size``.
    split : str
        ``"train"`` or ``"test"``.

    Yields
    ------
    tuple[numpy.ndarray, numpy.ndarray]
        Windows of shape ``(B, w, d)`` and labels of shape ``(L, B)``.
    """
    dataset = generate_dataset(cfg["data"], int(cfg["seed"]))
    if split == "train":
        indices = dataset.train_idx
    elif split == "test":
        indices = dataset.test_idx
    else:
        raise ValueError(f"Unknown split {split!r}; expected 'train' or 'test'.")
    batch_size = int(cfg["data"]["batch_size"])
    for start in range(0, len(indices), batch_size):
        batch = indices[start : start + batch_size]
        yield dataset.windows[batch], dataset.labels[:, batch]
