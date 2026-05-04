from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class DtypeSpec:
    label: str
    torch_dtype: torch.dtype


_DTYPE_ALIASES: dict[str, torch.dtype] = {
    "float16": torch.float16,
    "half": torch.float16,
    "float32": torch.float32,
    "float": torch.float32,
    "single": torch.float32,
    "float64": torch.float64,
    "double": torch.float64,
    "bfloat16": torch.bfloat16,
    "complex64": torch.complex64,
    "cfloat": torch.complex64,
    "complex128": torch.complex128,
    "cdouble": torch.complex128,
}

_CANONICAL_LABELS: dict[torch.dtype, str] = {
    torch.float16: "float16",
    torch.float32: "float32",
    torch.float64: "float64",
    torch.bfloat16: "bfloat16",
    torch.complex64: "complex64",
    torch.complex128: "complex128",
}


def _normalize_label(label: str) -> str:
    normalized = label.strip().lower()
    if normalized not in _DTYPE_ALIASES:
        supported = ", ".join(sorted(_DTYPE_ALIASES))
        raise ValueError(
            f"Unsupported dtype alias '{label}'. Expected one of: {supported}"
        )
    return normalized


def coerce_dtype_spec(value: Any) -> DtypeSpec:
    if isinstance(value, DtypeSpec):
        return value
    if isinstance(value, torch.dtype):
        return DtypeSpec(
            label=_CANONICAL_LABELS.get(value, str(value).removeprefix("torch.")),
            torch_dtype=value,
        )
    if isinstance(value, str):
        normalized = _normalize_label(value)
        torch_dtype = _DTYPE_ALIASES[normalized]
        return DtypeSpec(
            label=_CANONICAL_LABELS.get(torch_dtype, normalized),
            torch_dtype=torch_dtype,
        )
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return coerce_dtype_spec(
            value[1] if isinstance(value[1], torch.dtype) else value[0]
        )
    if isinstance(value, Mapping):
        if "label" in value:
            return coerce_dtype_spec(value["label"])
        if "dtype" in value:
            return coerce_dtype_spec(value["dtype"])
    raise TypeError(
        "dtype values must be a string alias, torch.dtype, or DtypeSpec-compatible pair"
    )


def dtype_label(value: Any) -> str:
    return coerce_dtype_spec(value).label


def dtype_torch(value: Any) -> torch.dtype:
    return coerce_dtype_spec(value).torch_dtype
