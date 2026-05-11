"""Project-level path helpers for benchmark-specific models and results."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parents[1]
MODELS_ROOT = PROJECT_ROOT / "models"
RESULTS_ROOT = PROJECT_ROOT / "results"
DATA_ROOT = Path(os.getenv("DATA_DIR", REPO_ROOT / "data")) / PROJECT_ROOT.name
KNOWN_BENCHMARKS = {"DHO", "SEE", "DEE", "TAF"}


def results_dir_for_model_dir(model_dir: str | Path) -> str:
    """
    Map a benchmark checkpoint directory to its corresponding results folder.

    The folder structure mirrors the paper's four benchmark families: DHO, SEE,
    DEE, and TAF.
    """
    model_path = Path(model_dir)
    benchmark = model_path.name.upper()
    if benchmark not in KNOWN_BENCHMARKS:
        raise ValueError(f"Cannot infer benchmark from model directory '{model_path}'")
    return str(RESULTS_ROOT / benchmark)


def results_case_dir_for_model_dir(model_dir: str | Path, case_prefix: str) -> str:
    """
    Return the benchmark results directory for one specific model configuration.

    Example:
      models/DEE + dee_cc_10-4 -> results/DEE/dee_cc_10-4
    """
    return str(Path(results_dir_for_model_dir(model_dir)) / case_prefix)


def model_dir_for_benchmark(benchmark: str) -> str:
    """Return the curated checkpoint directory for a benchmark family."""
    benchmark_name = benchmark.upper()
    if benchmark_name not in KNOWN_BENCHMARKS:
        raise ValueError(f"Unknown benchmark: {benchmark}")
    return str(MODELS_ROOT / benchmark_name)


def results_dir_for_benchmark(benchmark: str) -> str:
    """Return the curated results directory for a benchmark family."""
    benchmark_name = benchmark.upper()
    if benchmark_name not in KNOWN_BENCHMARKS:
        raise ValueError(f"Unknown benchmark: {benchmark}")
    return str(RESULTS_ROOT / benchmark_name)


def data_dir_for_benchmark(benchmark: str) -> Path:
    """Return the shared data directory for a benchmark family."""
    return DATA_ROOT / benchmark
