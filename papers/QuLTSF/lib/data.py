"""Paper-local entry point to the shared LTSF data pipeline.

The data root comes from ``--data-root`` (``cfg["data_root"]``) or the
``DATA_DIR`` environment variable, resolved by the shared runtime; the Weather
file then lives at ``<data_root>/QuLTSF/<dataset.path>``.
"""

from __future__ import annotations

import torch

from papers.shared.QuLTSF.data import (
    SeriesSplits,
    build_dataset,
    load_series,
)
from papers.shared.QuLTSF.data import build_dataloader as _build_split_loader
from runtime_lib.dtypes import dtype_torch


def model_dtype(cfg: dict) -> torch.dtype:
    """Return the floating dtype requested by ``cfg["dtype"]``.

    Parameters
    ----------
    cfg : dict
        Resolved configuration.

    Returns
    -------
    torch.dtype
        The configured dtype.

    Raises
    ------
    KeyError
        If ``dtype`` is missing from the configuration.
    """
    if cfg.get("dtype") is None:
        raise KeyError("Missing required config key dtype")
    return dtype_torch(cfg["dtype"])


def build_dataloader(cfg: dict, split: str, series: SeriesSplits | None = None):
    """Return the loader of one split.

    Parameters
    ----------
    cfg : dict
        Resolved configuration.
    split : str
        ``"train"``, ``"val"`` or ``"test"``.
    series : SeriesSplits | None
        Already loaded series, to avoid re-reading the CSV for each split.
        Default value is None (load it here).

    Returns
    -------
    papers.shared.QuLTSF.data.WindowBatches
        Batches ``(input [B, L, C], target [B, T, C])`` on ``cfg["device"]``.
    """
    series = series if series is not None else load_series(cfg)
    if "device" not in cfg:
        raise KeyError("Missing required config key device")
    return _build_split_loader(series, cfg, split, model_dtype(cfg), cfg["device"])


__all__ = [
    "SeriesSplits",
    "build_dataloader",
    "build_dataset",
    "load_series",
    "model_dtype",
]
