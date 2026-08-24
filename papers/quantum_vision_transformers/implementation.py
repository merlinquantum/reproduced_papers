#!/usr/bin/env python3
"""Compatibility entry point for the QVT paper folder.

The shared repo-level runner is the primary entrypoint now. This shim keeps the
paper-local invocation working while the migration settles.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _print_models() -> None:
    # Keep this in the shim so `--list-models` is fast and does not pull heavy deps.
    models = {
        "A": "Orthogonal Patch-wise (no attention)",
        "B": "Quantum Orthogonal Transformer (overlap attention)",
        "C": "Direct Quantum Attention (pragmatic hybrid)",
        "D": "Compound Transformer (2-photon compound matrix)",
        "E": "Multi-sector Attention (shared circuit, 1ph+2ph)",
        "F": "Hierarchical Compound (3-photon, region+patch+feature)",
        "VisionTransformer": "Classical baseline from the paper appendix",
        "OrthoFNN": "Quantum fully connected baseline from the paper",
    }
    for key, value in models.items():
        print(f"  {key}  {value}")


def resolve_runtime_dtype(cfg: dict, dtype_override: str | None = None):
    # Used by paper-local tests; keep import lazy so the shim works from any CWD.
    from lib.runner import resolve_runtime_dtype as _resolve_runtime_dtype

    return _resolve_runtime_dtype(cfg, dtype_override)


def main(argv: list[str] | None = None) -> int:
    argv = [] if argv is None else list(argv)
    if "--list-models" in argv:
        _print_models()
        return 0

    project_dir = Path(__file__).resolve().parent
    from runtime_lib import run_from_project

    run_from_project(project_dir, argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
