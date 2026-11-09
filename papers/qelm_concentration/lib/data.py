"""Datasets of the reproduction: Statlog Landsat (real) and NARMA-n (synthetic by definition).

Both sit behind :func:`build_dataloader`, selected by ``cfg["dataset"]["name"]``.

Data root: ``cfg["data_root"]``, which the shared runtime takes from ``--data-root``, else
the config (``defaults.json`` sets ``../../data``, i.e. ``<repo>/data``), else the ``DATA_DIR``
environment variable when the config value is null. Landsat files are read from
``<data_root>/qelm_concentration/<dataset.landsat.subdir>/sat.{trn,tst}``. Dataset settings
live under ``cfg["dataset"]["landsat"]`` and ``cfg["dataset"]["narma"]``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from runtime_lib.data_paths import resolve_data_root

logger = logging.getLogger(__name__)

PAPER_DIR = Path(__file__).resolve().parents[1]
PAPER_DATA_DIRNAME = "qelm_concentration"
LANDSAT_FILES = ("sat.trn", "sat.tst")
LANDSAT_N_FEATURES = 36
NARMA_INPUT_LOW, NARMA_INPUT_HIGH = 0.0, 0.5
# The paper writes "u~ = 2u - 1 ~ Unif(-1, 1)" with u ~ Unif(0, 0.5), which is inconsistent:
# 2u - 1 lies in [-1, 0]. "unit_interval" maps [0, 0.5] affinely onto [-1, 1] (the stated
# distribution); "two_u_minus_one" applies the stated formula literally.
NARMA_INPUT_ENCODINGS = {
    "unit_interval": lambda u: 4 * u - 1,
    "two_u_minus_one": lambda u: 2 * u - 1,
}


# --------------------------------------------------------------------------- Landsat


@dataclass(frozen=True)
class LandsatSplit:
    """One seeded subsample and train/test split of Landsat.

    Attributes
    ----------
    train_x, test_x : numpy.ndarray
        Encoder inputs, min-max scaled to ``[-1, 1]`` with training statistics and
        tiled ``n_copies`` times (72 = 2 x 36 in the paper).
    train_raw, test_raw : numpy.ndarray
        Unscaled 36 raw features (for the classical baselines, which apply their own
        StandardScaler as in paper Table A2).
    train_y, test_y : numpy.ndarray
        Integer labels in {1, 2, 3, 4, 5, 7}.
    subset_indices : numpy.ndarray
        Row indices of the pooled 6435-row table kept by the subsample.
    """

    train_x: np.ndarray
    test_x: np.ndarray
    train_raw: np.ndarray
    test_raw: np.ndarray
    train_y: np.ndarray
    test_y: np.ndarray
    subset_indices: np.ndarray


def landsat_directory(cfg: dict) -> Path:
    """Directory holding ``sat.trn`` and ``sat.tst``.

    A relative ``cfg["data_root"]`` is resolved against this paper folder (the shared
    runtime passes an absolute path).
    """
    data_root = resolve_data_root(cfg.get("data_root"), PAPER_DIR)
    return data_root / PAPER_DATA_DIRNAME / cfg["dataset"]["landsat"]["subdir"]


def load_landsat_table(directory: Path) -> tuple[np.ndarray, np.ndarray]:
    """Pool the UCI train and test files into one ``(6435, 36)`` table and labels.

    Raises
    ------
    FileNotFoundError
        If a Landsat file is missing (see README "Data" for the download command).
    """
    tables = []
    for name in LANDSAT_FILES:
        path = directory / name
        if not path.exists():
            raise FileNotFoundError(
                f"Missing Landsat file {path}. Download it as described in the README Data section."
            )
        tables.append(np.loadtxt(path, dtype=np.int64))
    table = np.concatenate(tables, axis=0)
    if table.shape[1] != LANDSAT_N_FEATURES + 1:
        raise ValueError(f"Unexpected Landsat table shape {table.shape}")
    return table[:, :LANDSAT_N_FEATURES].astype(float), table[:, -1]


def minmax_to_unit_interval(
    train: np.ndarray, test: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Scale each feature to ``[-1, 1]`` with training min/max (test may leave the range)."""
    low = train.min(axis=0)
    span = train.max(axis=0) - low
    span = np.where(span > 0, span, 1.0)
    return 2 * (train - low) / span - 1, 2 * (test - low) / span - 1


def make_landsat_split(cfg: dict, split_seed: int) -> LandsatSplit:
    """Draw the stratified subsample and stratified train/test split for ``split_seed``.

    The paper's 860-row subsample and its split are unpublished; this subsample is our
    own (reduced-scope relative to the paper's selection).
    """
    dataset_cfg = cfg["dataset"]["landsat"]
    features, labels = load_landsat_table(landsat_directory(cfg))
    n_subsample = int(dataset_cfg["n_subsample"])
    all_indices = np.arange(len(labels))
    if n_subsample < len(labels):
        subset_indices, _ = train_test_split(
            all_indices,
            train_size=n_subsample,
            stratify=labels,
            random_state=split_seed,
        )
    else:
        subset_indices = all_indices
    subset_indices = np.sort(subset_indices)
    train_idx, test_idx = train_test_split(
        subset_indices,
        test_size=float(dataset_cfg["test_fraction"]),
        stratify=labels[subset_indices],
        random_state=split_seed,
    )
    train_raw, test_raw = features[train_idx], features[test_idx]
    scaling = dataset_cfg["input_scaling"]
    if scaling == "minmax":
        train_scaled, test_scaled = minmax_to_unit_interval(train_raw, test_raw)
    elif scaling == "standard":
        # z-scores with training statistics (the paper's classical pipeline standardises;
        # its QELM input scaling is unstated).
        mean, std = train_raw.mean(axis=0), train_raw.std(axis=0)
        std = np.where(std > 0, std, 1.0)
        train_scaled, test_scaled = (train_raw - mean) / std, (test_raw - mean) / std
    else:
        raise ValueError(
            f"Unknown input_scaling {scaling!r}; expected 'minmax' or 'standard'"
        )
    n_copies = int(dataset_cfg["n_copies"])
    return LandsatSplit(
        train_x=np.tile(train_scaled, (1, n_copies)),
        test_x=np.tile(test_scaled, (1, n_copies)),
        train_raw=train_raw,
        test_raw=test_raw,
        train_y=labels[train_idx],
        test_y=labels[test_idx],
        subset_indices=subset_indices,
    )


# --------------------------------------------------------------------------- NARMA


@dataclass(frozen=True)
class NarmaSequence:
    """Windows and NARMA-n targets of one input sequence (paper App. "Methods").

    Attributes
    ----------
    train_windows, test_windows : numpy.ndarray
        Shape ``(T, L)``; row ``t`` is ``(u~_t, u~_{t-1}, ..., u~_{t-L+1})`` with ``u~``
        the configured input encoding of ``u`` (see ``NARMA_INPUT_ENCODINGS``).
    train_targets, test_targets : numpy.ndarray
        Shape ``(T, n_orders)``; column ``j`` is ``y_{t+1}`` of NARMA-``orders[j]``.
    orders : tuple of int
        Task orders.
    raw_inputs : numpy.ndarray
        Full input sequence ``u`` in ``[0, 0.5]`` (used by the ESN baseline).
    first_window_index : int
        Index in ``raw_inputs`` of the time step of the first window.
    """

    train_windows: np.ndarray
    test_windows: np.ndarray
    train_targets: np.ndarray
    test_targets: np.ndarray
    orders: tuple[int, ...]
    raw_inputs: np.ndarray
    first_window_index: int


def narma_series(inputs: np.ndarray, order: int) -> np.ndarray:
    """NARMA-n reference outputs (paper App. "Methods").

    ``y_{t+1} = 0.2 y_t + 0.04 sum_{i=0}^{n-1} y_{t-i} + 1.5 u_{t-n+1} u_t + 0.001``,
    with ``y_t = 0`` for ``t < n``.
    """
    outputs = np.zeros_like(inputs)
    for t in range(order - 1, len(inputs) - 1):
        recent = outputs[t - order + 1 : t + 1].sum()
        outputs[t + 1] = (
            0.2 * outputs[t]
            + 0.04 * recent
            + 1.5 * inputs[t - order + 1] * inputs[t]
            + 0.001
        )
    return outputs


def make_narma_sequence(
    window: int,
    orders: list[int],
    t_init: int,
    t_train: int,
    t_test: int,
    input_encoding: str,
    rng: np.random.Generator,
) -> NarmaSequence:
    """Sample one input sequence and build its windows and multi-order targets.

    ``T_tot = T_init + T_train + T_test + (L - 1)`` inputs are drawn; the last
    ``T_train + T_test + L - 1`` form the windows. The target of the window ending at
    ``u_t`` is ``y_{t+1}``, so one extra input step is drawn to define the last target.
    """
    total = t_init + t_train + t_test + (window - 1) + 1
    if input_encoding not in NARMA_INPUT_ENCODINGS:
        raise ValueError(
            f"Unknown NARMA input_encoding {input_encoding!r}; expected one of "
            f"{sorted(NARMA_INPUT_ENCODINGS)}"
        )
    raw = rng.uniform(NARMA_INPUT_LOW, NARMA_INPUT_HIGH, size=total)
    encoded = NARMA_INPUT_ENCODINGS[input_encoding](raw)
    first = t_init + window - 1
    steps = np.arange(first, first + t_train + t_test)
    windows = np.stack([encoded[steps - lag] for lag in range(window)], axis=1)
    targets = np.stack(
        [narma_series(raw, order)[steps + 1] for order in orders], axis=1
    )
    return NarmaSequence(
        train_windows=windows[:t_train],
        test_windows=windows[t_train:],
        train_targets=targets[:t_train],
        test_targets=targets[t_train:],
        orders=tuple(int(order) for order in orders),
        raw_inputs=raw,
        first_window_index=first,
    )


def narma_sequence_from_config(
    cfg: dict, window: int, rng: np.random.Generator
) -> NarmaSequence:
    """Build a :class:`NarmaSequence` from ``cfg["dataset"]["narma"]`` (requires every key)."""
    dataset_cfg = cfg["dataset"]["narma"]
    return make_narma_sequence(
        window=window,
        orders=list(dataset_cfg["orders"]),
        t_init=int(dataset_cfg["t_init"]),
        t_train=int(dataset_cfg["t_train"]),
        t_test=int(dataset_cfg["t_test"]),
        input_encoding=dataset_cfg["input_encoding"],
        rng=rng,
    )


# --------------------------------------------------------------------------- loader


def build_dataloader(cfg: dict, split: str, seed: int | None = None) -> DataLoader:
    """Batches of ``(encoder_inputs, targets)`` for the configured dataset.

    Parameters
    ----------
    cfg : dict
        Resolved config; ``cfg["dataset"]["name"]`` is ``"landsat"`` or ``"narma"``.
    split : str
        ``"train"`` or ``"test"``.
    seed : int or None
        Split seed (Landsat) or sequence seed (NARMA). Default value is None, which uses
        ``cfg["seed"]``.

    Returns
    -------
    torch.utils.data.DataLoader
        Unshuffled batches of size ``cfg["dataset"]["batch_size"]``. Landsat yields
        ``(B, 36 * n_copies)`` inputs with integer labels; NARMA yields ``(B, L)`` windows
        with ``(B, n_orders)`` targets, ``L = N/2``.
    """
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")
    seed = int(cfg["seed"] if seed is None else seed)
    name = cfg["dataset"]["name"]
    if name == "landsat":
        data = make_landsat_split(cfg, seed)
        inputs = data.train_x if split == "train" else data.test_x
        targets = torch.as_tensor(data.train_y if split == "train" else data.test_y)
    elif name == "narma":
        window = int(cfg["reservoir"]["n_qubits"]) // 2
        sequence = narma_sequence_from_config(cfg, window, np.random.default_rng(seed))
        inputs = sequence.train_windows if split == "train" else sequence.test_windows
        targets = torch.as_tensor(
            sequence.train_targets if split == "train" else sequence.test_targets
        )
    else:
        raise ValueError(
            f"Unknown dataset name {name!r}; expected 'landsat' or 'narma'"
        )
    dataset = TensorDataset(torch.as_tensor(inputs), targets)
    return DataLoader(
        dataset, batch_size=int(cfg["dataset"]["batch_size"]), shuffle=False
    )
