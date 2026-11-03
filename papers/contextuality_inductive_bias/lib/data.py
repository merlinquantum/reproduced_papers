"""RPS payoff data sets (Sec. 5.2, App. F) and the feature map of Eq. (93).

Splits
------
``train``
    1500 labelled strategies. ``data.train.source = "author"`` loads the
    authors' ``Xdata.npy`` / ``Ydata.npy`` from
    ``<data_root>/contextuality_inductive_bias/zero_sum_game_payoff/`` after a
    SHA-256 check; ``"generated"`` draws them with the App. F generator from
    ``data.train.seed`` (seed 999 reproduces the author arrays exactly).
``validation``
    Optional labelled split from the same generator (``data.validation``),
    used only for hyperparameter selection in the fair comparison.
``test``
    ``data.test.n`` unlabelled strategies with their exact win probabilities
    (the paper evaluates the KL divergence to the true distribution).

Labels follow the author code: 0 = positive payoff (+1), 1 = negative (-1).

Input scaling protocols (discrepancy D12 in LOG.md): the paper scales the
compressed inputs by pi/2; the author code does so for training but feeds the
test strategies unscaled.

==========  =======================  ===========
protocol    train/validation scale   test scale
==========  =======================  ===========
``paper``   pi/2                     pi/2
``code``    pi/2                     1
``fork``    1                        1
==========  =======================  ===========
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from runtime_lib.data_paths import (
    DEFAULT_DATA_DIRNAME,
    ENV_DATA_ROOT,
    resolve_data_root,
)
from runtime_lib.dtypes import dtype_torch

from .rps_game import win_probabilities

PAPER_NAME = "contextuality_inductive_bias"
DATASET_DIRNAME = "zero_sum_game_payoff"
AUTHOR_FILES = ("Xdata.npy", "Ydata.npy")
PROTOCOLS = {
    "paper": {"train": math.pi / 2, "test": math.pi / 2},
    "code": {"train": math.pi / 2, "test": 1.0},
    "fork": {"train": 1.0, "test": 1.0},
}
SPLITS = ("train", "validation", "test")
_PROJECT_DIR = Path(__file__).resolve().parents[1]
_REPO_ROOT = Path(__file__).resolve().parents[3]

logger = logging.getLogger(__name__)


@dataclass
class SplitData:
    """One data split, ready for training or evaluation.

    Attributes
    ----------
    name : str
        Split name.
    strategies : numpy.ndarray
        Raw strategy matrices, shape ``(N, 3, 3)``.
    features : torch.Tensor
        Scaled compressed features (Eq. (93)), shape ``(N, 6)``.
    win_probabilities : torch.Tensor
        Exact P(Y_k = +1 | x), shape ``(N, 3)``.
    labels : torch.Tensor or None
        Sampled labels (0 = win, 1 = lose), shape ``(N, 3)``; ``None`` for test.
    input_scale : float
        Scale applied to the compressed features.
    provenance : dict
        Source, seed and array hashes recorded in ``run_status.json``.
    """

    name: str
    strategies: np.ndarray
    features: torch.Tensor
    win_probabilities: torch.Tensor
    labels: torch.Tensor | None
    input_scale: float
    provenance: dict = field(default_factory=dict)

    @property
    def size(self) -> int:
        return int(self.strategies.shape[0])


def compress_strategies(strategies: np.ndarray, scale: float) -> np.ndarray:
    """Map strategies to the six inputs of Eq. (93), times ``scale``.

    ``x1 = P(A1=R), x2 = P(A2=P), x3 = P(A3=S)`` (each player's special action)
    and ``x4 = P(A1=P) - P(A1=S), x5 = P(A2=S) - P(A2=R),
    x6 = P(A3=R) - P(A3=P)``.

    Parameters
    ----------
    strategies : numpy.ndarray
        Shape ``(N, 3, 3)``.
    scale : float
        Multiplicative input scale (pi/2 in the paper).

    Returns
    -------
    numpy.ndarray
        Shape ``(N, 6)``.
    """
    s = np.asarray(strategies, dtype=np.float64)
    columns = [
        s[:, 0, 0],
        s[:, 1, 1],
        s[:, 2, 2],
        s[:, 0, 1] - s[:, 0, 2],
        s[:, 1, 2] - s[:, 1, 0],
        s[:, 2, 0] - s[:, 2, 1],
    ]
    return scale * np.stack(columns, axis=1)


def generate_strategies(rng: np.random.RandomState, n: int) -> np.ndarray:
    """App. F: each row is (r1, r2, r3) / sum(r) with r_i ~ U[0, 1].

    Uses the legacy ``RandomState`` stream so that seed 999 reproduces the
    author arrays (the notebook calls ``np.random.seed(999)`` then
    ``np.random.rand(N, 3, 3)``).
    """
    raw = rng.rand(n, 3, 3)
    return raw / raw.sum(axis=2, keepdims=True)


def sample_labels(rng: np.random.RandomState, probabilities: np.ndarray) -> np.ndarray:
    """Sample payoffs as in the author code: label 0 (win) iff P > U[0, 1]."""
    threshold = rng.rand(*probabilities.shape)
    return np.where(probabilities > threshold, 0, 1).astype(np.int64)


def generate_labelled(seed: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Draw ``n`` strategies and their sampled labels from one seed."""
    rng = np.random.RandomState(seed)
    strategies = generate_strategies(rng, n)
    labels = sample_labels(rng, win_probabilities(strategies))
    return strategies, labels


def author_data_dir(cfg: dict) -> Path:
    """Directory holding the author arrays under the shared data root.

    Precedence matches the shared runtime: ``cfg["data_root"]`` (set by the
    runtime from ``--data-root``), then ``DATA_DIR``, then ``<repo>/data``.
    """
    preferred = cfg.get("data_root")
    if preferred is None and ENV_DATA_ROOT not in os.environ:
        preferred = _REPO_ROOT / DEFAULT_DATA_DIRNAME
    root = resolve_data_root(preferred)
    # TODO: remove once runtime_lib.find_repo_root no longer stops at the paper
    # folder's own implementation.py. The shared runtime chdirs into the paper
    # before resolving its default, so it sets data_root (and DATA_DIR) to
    # papers/<paper>/data instead of the repository data root of AGENTS.md.
    # Only that exact, non-existent folder is redirected; any other explicit
    # --data-root / DATA_DIR is used as given.
    mis_resolved_default = _PROJECT_DIR / DEFAULT_DATA_DIRNAME
    if (
        root.resolve() == mis_resolved_default.resolve()
        and not mis_resolved_default.exists()
    ):
        logger.warning(
            "DATA_ROOT_REDIRECTED | from=%s | to=%s | reason=runtime default resolved inside the paper folder",
            root,
            _REPO_ROOT / DEFAULT_DATA_DIRNAME,
        )
        root = _REPO_ROOT / DEFAULT_DATA_DIRNAME
    return root / PAPER_NAME / DATASET_DIRNAME


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_author_training(directory: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load and verify ``Xdata.npy`` / ``Ydata.npy`` against ``SHA256SUMS``.

    Raises
    ------
    FileNotFoundError
        If the directory, a data file, or ``SHA256SUMS`` is missing. The
        re-acquisition command is in LOG.md (Data Acquisition Log).
    ValueError
        If a checksum or array shape does not match.
    """
    sums_path = directory / "SHA256SUMS"
    if not sums_path.is_file():
        raise FileNotFoundError(
            f"Missing {sums_path}; re-acquire the author data as described in LOG.md"
        )
    expected = {}
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        digest, name = line.split()
        expected[name] = digest
    hashes = {}
    for name in AUTHOR_FILES:
        path = directory / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing author data file {path}")
        hashes[name] = _sha256(path)
        if hashes[name] != expected.get(name):
            raise ValueError(
                f"SHA-256 mismatch for {path}: {hashes[name]} != {expected.get(name)}"
            )
    strategies = np.load(directory / "Xdata.npy", allow_pickle=False)
    labels = np.load(directory / "Ydata.npy", allow_pickle=False)
    if strategies.shape[1:] != (3, 3) or labels.shape != (strategies.shape[0], 3):
        raise ValueError(
            f"Unexpected author array shapes {strategies.shape}, {labels.shape}"
        )
    return strategies, labels.astype(np.int64), {"source": "author", "sha256": hashes}


def protocol_scales(cfg: dict) -> dict:
    """Return the train and test input scales of ``data.protocol``."""
    protocol = cfg["data"]["protocol"]
    if protocol not in PROTOCOLS:
        raise ValueError(
            f"data.protocol must be one of {sorted(PROTOCOLS)}, got {protocol!r}"
        )
    return PROTOCOLS[protocol]


def _array_digest(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _real_dtype(cfg: dict) -> torch.dtype:
    dtype = dtype_torch(cfg["dtype"])
    if dtype not in (torch.float32, torch.float64):
        raise ValueError(f"dtype must be float32 or float64, got {cfg['dtype']!r}")
    return dtype


def load_split(cfg: dict, split: str) -> SplitData:
    """Build one split from the configuration.

    Parameters
    ----------
    cfg : dict
        Resolved configuration with ``data``, ``dtype`` and ``device`` keys.
    split : str
        ``"train"``, ``"validation"`` or ``"test"``.

    Returns
    -------
    SplitData
        Arrays and tensors of the split on ``cfg["device"]``.
    """
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    data_cfg = cfg["data"]
    scales = protocol_scales(cfg)
    labels = None
    if split == "train":
        source = data_cfg["train"]["source"]
        if source == "author":
            strategies, labels, provenance = load_author_training(author_data_dir(cfg))
        elif source == "generated":
            if data_cfg["train"]["seed"] is None:
                raise ValueError(
                    "data.train.seed is required when data.train.source = 'generated' (use --train-seed)"
                )
            seed = int(data_cfg["train"]["seed"])
            strategies, labels = generate_labelled(seed, int(data_cfg["train"]["n"]))
            provenance = {"source": "generated", "seed": seed}
        else:
            raise ValueError(
                f"data.train.source must be 'author' or 'generated', got {source!r}"
            )
        scale = scales["train"]
    elif split == "validation":
        seed = int(data_cfg["validation"]["seed"])
        strategies, labels = generate_labelled(seed, int(data_cfg["validation"]["n"]))
        provenance = {"source": "generated", "seed": seed}
        scale = scales["train"]
    else:
        seed = int(data_cfg["test"]["seed"])
        strategies = generate_strategies(
            np.random.RandomState(seed), int(data_cfg["test"]["n"])
        )
        provenance = {"source": "generated", "seed": seed}
        scale = scales["test"]
    arrays = (strategies,) if labels is None else (strategies, labels)
    provenance["array_sha256"] = _array_digest(*arrays)
    provenance["n"] = int(strategies.shape[0])
    provenance["input_scale"] = scale
    device = torch.device(cfg["device"])
    dtype = _real_dtype(cfg)
    return SplitData(
        name=split,
        strategies=strategies,
        features=torch.as_tensor(
            compress_strategies(strategies, scale), dtype=dtype, device=device
        ),
        win_probabilities=torch.as_tensor(
            win_probabilities(strategies), dtype=dtype, device=device
        ),
        labels=None
        if labels is None
        else torch.as_tensor(labels, dtype=dtype, device=device),
        input_scale=scale,
        provenance=provenance,
    )


def build_dataloader(cfg: dict, split: str) -> DataLoader:
    """Full-batch ``DataLoader`` of ``(features, labels, win_probabilities)``.

    The paper trains with full-batch gradient descent (App. F), so one batch
    holds the whole split. For the unlabelled test split the label tensor is a
    ``(N, 0)`` placeholder.
    """
    data = load_split(cfg, split)
    labels = (
        data.labels
        if data.labels is not None
        else data.features.new_zeros((data.size, 0))
    )
    dataset = TensorDataset(data.features, labels, data.win_probabilities)
    return DataLoader(dataset, batch_size=data.size, shuffle=False)


def describe_dataset(cfg: dict) -> dict:
    """Dataset identity recorded in ``run_status.json`` before data construction."""
    data_cfg = cfg["data"]
    scales = protocol_scales(cfg)
    train = data_cfg["train"]
    train_desc = (
        "author arrays (XanaduAI/Contextual-ML cb070ff240)"
        if train["source"] == "author"
        else f"generated seed={train['seed']}"
    )
    return {
        "name": "rps_zero_sum_game_payoff",
        "splits": {
            "train": train_desc,
            "validation": data_cfg["validation"]
            if data_cfg["validation"]["enabled"]
            else None,
            "test": {"n": data_cfg["test"]["n"], "seed": data_cfg["test"]["seed"]},
        },
        "preprocessing": (
            f"Eq. (93) compression; protocol={data_cfg['protocol']} "
            f"(train scale {scales['train']:.6f}, test scale {scales['test']:.6f})"
        ),
        "subset": "full",
    }
