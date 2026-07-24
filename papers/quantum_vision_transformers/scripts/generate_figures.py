#!/usr/bin/env python3
"""Backwards-compatible wrapper for figure generation helpers.

The main implementation lives in `scripts/analysis/generate_figures.py`. Some
tests and downstream tooling expect this file path to exist.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_analysis_module():
    script_path = Path(__file__).resolve().parent / "analysis" / "generate_figures.py"
    spec = importlib.util.spec_from_file_location(
        "qvt_generate_figures_analysis", script_path
    )
    if spec is None or spec.loader is None:  # pragma: no cover
        raise ImportError(f"Could not load generate_figures module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_analysis = _load_analysis_module()

make_variant_key = _analysis.make_variant_key
pretty_model_label = _analysis.pretty_model_label
group_by = _analysis.group_by

__all__ = ["make_variant_key", "pretty_model_label", "group_by"]
