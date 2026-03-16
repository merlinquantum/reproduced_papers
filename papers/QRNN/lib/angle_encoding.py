from __future__ import annotations

import torch


def apply_input_encoding(
    x: torch.Tensor,
    kind: str | None,
    *,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Map input features to angle parameters.

    Supported kinds:
    - "identity": no transform
    - "arccos": acos(clamp(x, -1, 1)) -> [0, pi]

    Notes
    -----
    The paper circuit is often described as R_y(arccos(x_t)). In practice, this
    requires x_t in [-1, 1]. When inputs are standardized, consider using
    dataset.feature_normalization="minmax_-1_1".
    """

    k = str(kind or "identity").strip().lower()
    if k in {"identity", "none"}:
        return x
    if k in {"arccos", "acos"}:
        x_clipped = x.clamp(-1.0 + eps, 1.0 - eps)
        return torch.acos(x_clipped)
    raise ValueError(
        f"Unsupported input_encoding {kind!r} (expected: identity, arccos)"
    )
