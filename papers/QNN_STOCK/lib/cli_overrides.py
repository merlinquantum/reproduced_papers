"""Custom CLI override hooks referenced from ``cli.json``."""

from __future__ import annotations

from typing import Any


def set_tickers(
    cfg: dict[str, Any], value: Any, arg_def: dict[str, Any]
) -> dict[str, Any]:
    """``--tickers AAPL,GOOGL`` -> ``dataset.tickers = ["AAPL", "GOOGL"]``."""

    tickers = [
        chunk.strip().upper() for chunk in str(value).split(",") if chunk.strip()
    ]
    if not tickers:
        raise ValueError("--tickers needs at least one ticker")
    cfg.setdefault("dataset", {})["tickers"] = tickers
    return cfg
