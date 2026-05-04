"""Project-level path helpers for benchmark-specific models and results."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_ROOT = PROJECT_ROOT / "models"
RESULTS_ROOT = PROJECT_ROOT / "results"
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
