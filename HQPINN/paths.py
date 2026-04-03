from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_ROOT = PROJECT_ROOT / "models"
RESULTS_ROOT = PROJECT_ROOT / "results"
KNOWN_BENCHMARKS = {"DHO", "SEE", "DEE", "TAF"}


def results_dir_for_model_dir(model_dir: str | Path) -> str:
    model_path = Path(model_dir)
    benchmark = model_path.name.upper()
    if benchmark not in KNOWN_BENCHMARKS:
        raise ValueError(f"Cannot infer benchmark from model directory '{model_path}'")
    return str(RESULTS_ROOT / benchmark)
