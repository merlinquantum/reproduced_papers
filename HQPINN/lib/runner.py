from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Mapping

from HQPINN.runner import run_from_project
from HQPINN.runtime import DEFAULT_CONFIG_PATH


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(merged.get(key), dict) and isinstance(value, dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _load_json_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _resolve_config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    inline_cfg = deepcopy(dict(cfg))
    config_path = inline_cfg.pop("config_path", None)

    resolved = _load_json_object(DEFAULT_CONFIG_PATH)
    if config_path:
        config_file = Path(config_path)
        if not config_file.is_absolute():
            config_file = Path.cwd() / config_file
        resolved = _merge_dicts(resolved, _load_json_object(config_file))

    return _merge_dicts(resolved, inline_cfg)


def train_and_evaluate(cfg: Mapping[str, Any], run_dir: str | Path) -> dict[str, Any]:
    """
    Shared-runner entrypoint for HQPINN.

    This wrapper normalizes config loading for the shared runner and ensures
    `run_dir` exists before delegating to benchmark code, which writes
    checkpoints and result artifacts directly into the top-level `models/`
    and `results/` folders.
    """
    resolved_run_dir = Path(run_dir).resolve()
    resolved_run_dir.mkdir(parents=True, exist_ok=True)

    config = _resolve_config(cfg)
    shared_runner_cfg = dict(config.get("shared_runner") or {})
    shared_runner_cfg["run_dir"] = str(resolved_run_dir)
    config["shared_runner"] = shared_runner_cfg

    run_from_project(config)

    return {
        "status": "completed",
        "run_dir": str(resolved_run_dir),
        "experiment": config.get("experiment"),
        "mode": config.get("mode"),
        "backend": config.get("backend"),
    }
