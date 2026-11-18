"""Strict access to nested configuration values.

Missing keys raise immediately instead of falling back to invented defaults,
as required by the repository ``AGENTS.md``.
"""

from __future__ import annotations

from typing import Any

_MISSING = object()


def require(cfg: dict[str, Any], dotted_key: str) -> Any:
    """Return a nested config value or fail with a clear error.

    Parameters
    ----------
    cfg : dict
        Resolved configuration.
    dotted_key : str
        Path such as ``"training.batch_size"``.

    Returns
    -------
    Any
        The stored value (``None`` is a valid stored value).

    Raises
    ------
    KeyError
        If any component of the path is missing.
    """
    node: Any = cfg
    walked: list[str] = []
    for part in dotted_key.split("."):
        walked.append(part)
        if not isinstance(node, dict) or part not in node:
            raise KeyError(f"Missing required config key '{'.'.join(walked)}'")
        node = node[part]
    return node


def require_choice(
    cfg: dict[str, Any], dotted_key: str, choices: tuple[str, ...]
) -> str:
    """Return a string config value that must be one of ``choices``.

    Parameters
    ----------
    cfg : dict
        Resolved configuration.
    dotted_key : str
        Path to the value.
    choices : tuple of str
        Allowed values.

    Returns
    -------
    str
        The validated value.

    Raises
    ------
    ValueError
        If the value is not one of ``choices``.
    """
    value = require(cfg, dotted_key)
    if value not in choices:
        raise ValueError(f"Config '{dotted_key}'={value!r} must be one of {choices}")
    return value
