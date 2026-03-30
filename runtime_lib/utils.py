from __future__ import annotations

import importlib
import re
from collections.abc import Iterable
from typing import Callable


def import_callable(path_spec: str) -> Callable[..., object]:
    module_name, attr = path_spec.rsplit(".", 1)
    module = importlib.import_module(module_name)
    fn = getattr(module, attr)
    if not callable(fn):  # pragma: no cover - defensive
        raise TypeError(f"Callable required: {path_spec}")
    return fn


def _iter_placeholders(obj: object, path: str = "") -> Iterable[tuple[str, str]]:
    """Yield (path, value) for any string containing <<...>> placeholders."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            next_path = f"{path}.{key}" if path else str(key)
            yield from _iter_placeholders(value, next_path)
    elif isinstance(obj, (list, tuple)):
        for idx, value in enumerate(obj):
            next_path = f"{path}[{idx}]" if path else f"[{idx}]"
            yield from _iter_placeholders(value, next_path)
    elif isinstance(obj, str):
        if re.search(r"<<[^<>]+>>", obj):
            yield path, obj


def ensure_no_placeholders(cfg: dict | list | tuple) -> None:
    """Raise if the config contains unresolved <<PLACEHOLDER>> strings."""
    placeholders = list(_iter_placeholders(cfg))
    if not placeholders:
        return
    details = "; ".join(f"{path}={value}" for path, value in placeholders)
    raise ValueError(
        "Configuration contains unresolved placeholders. Use CLI parameters to define"
        " these or replace <<...>> in configuration with real values before running: "
        f"{details}"
    )


__all__ = ["import_callable", "ensure_no_placeholders"]
