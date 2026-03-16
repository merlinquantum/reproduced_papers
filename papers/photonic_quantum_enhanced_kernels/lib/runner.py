"""Runtime entrypoints for the photonic quantum-enhanced kernels project."""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Any

from runtime_lib.utils import import_callable

LOGGER = logging.getLogger(__name__)

EXPERIMENT_SPECS = {
    "accuracy_vs_input_state": (
        "lib.accuracy_vs_input_state.run_accuracy_vs_input_state"
    ),
    "accuracy_vs_kernel": "lib.accuracy_vs_kernel.run_accuracy_vs_kernel",
    "accuracy_vs_width": "lib.accuracy_vs_width.run_accuracy_vs_width",
    "accuracy_vs_geometric_difference": (
        "lib.accuracy_vs_geometric_difference.run_accuracy_vs_geometric_difference"
    ),
}
DEFAULT_EXPERIMENT = "accuracy_vs_kernel"


def _canonicalize_experiment(name: str) -> str:
    return name.strip().lower().replace("-", "_")


def _normalize_experiments(value: Any) -> list[str]:
    if value is None:
        return [DEFAULT_EXPERIMENT]

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return [DEFAULT_EXPERIMENT]
        if text.lower() == "all":
            return list(EXPERIMENT_SPECS)
        items = [item.strip() for item in text.split(",") if item.strip()]
        return [_canonicalize_experiment(item) for item in items]

    if isinstance(value, (list, tuple, set)):
        items = [str(item).strip() for item in value if str(item).strip()]
        if len(items) == 1 and items[0].lower() == "all":
            return list(EXPERIMENT_SPECS)
        return [_canonicalize_experiment(item) for item in items]

    return [_canonicalize_experiment(str(value))]


def _validate_experiments(names: list[str]) -> None:
    unknown = [name for name in names if name not in EXPERIMENT_SPECS]
    if unknown:
        choices = ", ".join(sorted(EXPERIMENT_SPECS))
        raise ValueError(
            f"Unknown experiment(s): {', '.join(unknown)}. Available: {choices}"
        )


def _merge_plotting(cfg: dict[str, Any], exp_cfg: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    base = cfg.get("plotting")
    if isinstance(base, dict):
        merged.update(base)
    override = exp_cfg.get("plotting")
    if isinstance(override, dict):
        merged.update(override)
    return merged


def _build_experiment_config(cfg: dict[str, Any], name: str) -> dict[str, Any]:
    exp_cfg = copy.deepcopy(cfg.get("experiments", {}).get(name, {}))
    exp_cfg.setdefault("seed", cfg.get("seed", 42))
    plotting = _merge_plotting(cfg, exp_cfg)
    if plotting:
        exp_cfg["plotting"] = plotting
    return exp_cfg


def _run_experiment(
    name: str, exp_cfg: dict[str, Any], output_dir: Path
) -> dict[str, Any]:
    callable_path = EXPERIMENT_SPECS[name]
    runner = import_callable(callable_path)
    return runner(exp_cfg, output_dir)


def _stringify(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _stringify(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_stringify(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def train_and_evaluate(cfg: dict[str, Any], run_dir: Path) -> None:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    experiments = _normalize_experiments(cfg.get("experiment"))
    _validate_experiments(experiments)

    results: dict[str, Any] = {}
    if cfg.get("dry_run"):
        for name in experiments:
            exp_dir = run_dir / name
            exp_dir.mkdir(parents=True, exist_ok=True)
            marker = exp_dir / "dry_run.txt"
            marker.write_text(
                "Dry run: experiment execution skipped.\n", encoding="utf-8"
            )
            results[name] = {"status": "dry_run", "output_dir": str(exp_dir)}
    else:
        for name in experiments:
            exp_dir = run_dir / name
            exp_dir.mkdir(parents=True, exist_ok=True)
            exp_cfg = _build_experiment_config(cfg, name)
            LOGGER.info("Running experiment: %s", name)
            results[name] = _run_experiment(name, exp_cfg, exp_dir)

    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(_stringify(results), indent=2), encoding="utf-8")
    LOGGER.info("Saved summary to %s", summary_path)


__all__ = ["train_and_evaluate", "EXPERIMENT_SPECS"]
