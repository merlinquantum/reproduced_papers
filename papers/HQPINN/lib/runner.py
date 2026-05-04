"""Shared-runtime dispatcher for HQPINN experiments."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .runtime import (
    DEFAULT_CONFIG_PATH,
    apply_runtime_config,
    require_file,
)

EXPERIMENT_ALIASES: dict[str, str] = {
    "dho-cc": "dho-cc",
    "dho-hy-pl": "dho-hy-pl",
    "dho-hy-m": "dho-hy-m",
    "dho-cp": "dho-hy-pl",
    "dho-ci": "dho-hy-m",
    "dho-cperc": "dho-hy-mp",
    "dho-hy-mp": "dho-hy-mp",
    "dho-pp": "dho-qq-pl",
    "dho-qq-pl": "dho-qq-pl",
    "dho-ii": "dho-qq-m",
    "dho-qq-m": "dho-qq-m",
    "dho-percperc": "dho-qq-mp",
    "dho-qq-mp": "dho-qq-mp",
    "see-cc": "see-cc",
    "see-cp": "see-hy-pl",
    "see-ci": "see-hy-m",
    "see-hy-pl": "see-hy-pl",
    "see-hy-m": "see-hy-m",
    "see-pp": "see-qq-pl",
    "see-qq-pl": "see-qq-pl",
    "see-ii": "see-qq-m",
    "see-qq-m": "see-qq-m",
    "dee-cc": "dee-cc",
    "dee-cp": "dee-hy-pl",
    "dee-ci": "dee-hy-m",
    "dee-hy-pl": "dee-hy-pl",
    "dee-hy-m": "dee-hy-m",
    "dee-pp": "dee-qq-pl",
    "dee-qq-pl": "dee-qq-pl",
    "dee-ii": "dee-qq-m",
    "dee-qq-m": "dee-qq-m",
    "taf-cc": "taf-cc",
    "taf-cp": "taf-hy-pl",
    "taf-ci": "taf-hy-m",
    "taf-hy-pl": "taf-hy-pl",
    "taf-hy-m": "taf-hy-m",
    "taf-pp": "taf-qq-pl",
    "taf-qq-pl": "taf-qq-pl",
    "taf-ii": "taf-qq-m",
    "taf-qq-m": "taf-qq-m",
}


def _canonical_experiment_name(experiment: str) -> str:
    return EXPERIMENT_ALIASES.get(experiment, experiment)


def _merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(merged.get(key), dict) and isinstance(value, dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_json(path: Path) -> dict[str, Any]:
    with require_file(path, label="runtime config file").open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return data


def _normalize_config(config: dict[str, Any]) -> dict[str, Any]:
    experiment = config.get("experiment")
    if not experiment:
        raise ValueError("Config must define 'experiment'")
    if not isinstance(experiment, str):
        raise ValueError("Config 'experiment' must be a string")
    config["experiment"] = _canonical_experiment_name(experiment)

    mode = config.get("mode")
    if mode not in {"train", "run", "remote"}:
        raise ValueError("Config 'mode' must be 'train', 'run', or 'remote'")

    backend = config.get("backend")
    if not isinstance(backend, str) or not backend:
        raise ValueError("Config must define a non-empty string 'backend'")

    return config


def _load_config(config_path: str) -> dict[str, Any]:
    defaults = _load_json(DEFAULT_CONFIG_PATH)

    path = Path(config_path)
    if not path.is_absolute():
        path = Path.cwd() / path
    config = _load_json(path)
    merged = _merge_dicts(defaults, config)

    return _normalize_config(merged)


def _require_model_int(
    model_config: dict[str, Any],
    key: str,
    experiment: str,
) -> int:
    value = model_config.get(key)
    if value is None:
        raise ValueError(f"{experiment} config requires model.{key}")
    return int(value)


def _build_model_size(*parts: int) -> str:
    return "-".join(str(part) for part in parts)


def run_from_project(config: dict[str, Any]) -> None:
    config = _normalize_config(apply_runtime_config(config))
    experiment = config["experiment"]
    mode = config["mode"]
    backend = config["backend"]
    model_config = config.get("model") or {}
    shared_runner_config = config.get("shared_runner") or {}
    force_retrain = bool(shared_runner_config.get("force_retrain", False))

    if experiment == "dho-cc":
        n_layers = model_config.get("n_layers")
        n_nodes = model_config.get("n_nodes")
        if n_layers is None or n_nodes is None:
            raise ValueError("dho-cc config requires model.n_layers and model.n_nodes")

        from .DHO.dho_cc import run

        run(
            mode=mode,
            backend=backend,
            n_layers=int(n_layers),
            n_nodes=int(n_nodes),
            **({"force_retrain": True} if force_retrain else {}),
        )
        return

    if experiment == "dho-hy-pl":
        n_layers = model_config.get("n_layers")
        n_nodes = model_config.get("n_nodes")
        n_qubits = model_config.get("n_qubits")
        if n_layers is None or n_nodes is None or n_qubits is None:
            raise ValueError(
                "dho-hy-pl config requires model.n_layers, model.n_nodes, and model.n_qubits"
            )

        from .DHO.dho_hy_pl import run

        run(
            mode=mode,
            backend=backend,
            n_layers=int(n_layers),
            n_nodes=int(n_nodes),
            n_qubits=int(n_qubits),
        )
        return

    if experiment == "dho-qq-pl":
        n_qubits = model_config.get("n_qubits")
        if n_qubits is None:
            raise ValueError("dho-qq-pl config requires model.n_qubits")

        from .DHO.dho_qq_pl import run

        run(
            mode=mode,
            backend=backend,
            n_qubits=int(n_qubits),
        )
        return

    if experiment == "dho-hy-m":
        n_layers = model_config.get("n_layers")
        n_nodes = model_config.get("n_nodes")
        n_photons = model_config.get("n_photons")
        if n_layers is None or n_nodes is None or n_photons is None:
            raise ValueError(
                "dho-hy-m config requires model.n_layers, model.n_nodes, and model.n_photons"
            )

        from .DHO.dho_hy_m import run

        run(
            mode=mode,
            backend=backend,
            n_layers=int(n_layers),
            n_nodes=int(n_nodes),
            n_photons=int(n_photons),
            **({"force_retrain": True} if force_retrain else {}),
        )
        return

    if experiment == "dho-hy-mp":
        n_layers = model_config.get("n_layers")
        n_nodes = model_config.get("n_nodes")
        if n_layers is None or n_nodes is None:
            raise ValueError(
                "dho-hy-mp config requires model.n_layers and model.n_nodes"
            )

        from .DHO.dho_hy_mp import run

        run(
            mode=mode,
            backend=backend,
            n_layers=int(n_layers),
            n_nodes=int(n_nodes),
        )
        return

    if experiment == "dho-qq-m":
        n_photons = model_config.get("n_photons")
        if n_photons is None:
            raise ValueError("dho-qq-m config requires model.n_photons")

        from .DHO.dho_qq_m import run

        run(
            mode=mode,
            backend=backend,
            n_photons=int(n_photons),
            **({"force_retrain": True} if force_retrain else {}),
        )
        return

    if experiment == "dho-qq-mp":
        from .DHO.dho_qq_mp import run

        run(
            mode=mode,
            backend=backend,
        )
        return

    if experiment == "see-cc":
        n_layers = model_config.get("n_layers")
        n_nodes = model_config.get("n_nodes")
        if n_layers is None or n_nodes is None:
            raise ValueError("see-cc config requires model.n_layers and model.n_nodes")

        from .SEE.see_cc import run

        run(
            mode=mode,
            backend=backend,
            n_layers=int(n_layers),
            n_nodes=int(n_nodes),
        )
        return

    if experiment == "see-hy-m":
        n_layers = model_config.get("n_layers")
        n_nodes = model_config.get("n_nodes")
        n_photons = model_config.get("n_photons")
        if n_layers is None or n_nodes is None or n_photons is None:
            raise ValueError(
                "see-hy-m config requires model.n_layers, model.n_nodes, and model.n_photons"
            )

        from .SEE.see_hy_m import run

        run(
            mode=mode,
            backend=backend,
            n_layers=int(n_layers),
            n_nodes=int(n_nodes),
            n_photons=int(n_photons),
        )
        return

    if experiment == "see-hy-pl":
        n_layers = model_config.get("n_layers")
        n_nodes = model_config.get("n_nodes")
        q_layers = model_config.get("q_layers")
        if n_layers is None or n_nodes is None or q_layers is None:
            raise ValueError(
                "see-hy-pl config requires model.n_layers, model.n_nodes, and model.q_layers"
            )

        from .SEE.see_hy_pl import run

        run(
            mode=mode,
            backend=backend,
            n_layers=int(n_layers),
            n_nodes=int(n_nodes),
            q_layers=int(q_layers),
        )
        return

    if experiment == "see-qq-m":
        n_photons = model_config.get("n_photons")
        if n_photons is None:
            raise ValueError("see-qq-m config requires model.n_photons")

        from .SEE.see_qq_m import run

        run(
            mode=mode,
            backend=backend,
            n_photons=int(n_photons),
        )
        return

    if experiment == "see-qq-pl":
        q_layers = model_config.get("q_layers")
        if q_layers is None:
            raise ValueError("see-qq-pl config requires model.q_layers")

        from .SEE.see_qq_pl import run

        run(
            mode=mode,
            backend=backend,
            q_layers=int(q_layers),
        )
        return

    if experiment == "dee-cc":
        n_layers = _require_model_int(model_config, "n_layers", experiment)
        n_nodes = _require_model_int(model_config, "n_nodes", experiment)

        from .DEE.dee_cc import run

        run(
            mode=mode,
            backend=backend,
            n_layers=n_layers,
            n_nodes=n_nodes,
        )
        return

    if experiment == "dee-hy-m":
        n_layers = _require_model_int(model_config, "n_layers", experiment)
        n_nodes = _require_model_int(model_config, "n_nodes", experiment)
        n_photons = _require_model_int(model_config, "n_photons", experiment)

        from .DEE.dee_hy_m import run

        run(
            mode=mode,
            backend=backend,
            model_size=_build_model_size(n_nodes, n_layers, n_photons),
        )
        return

    if experiment == "dee-hy-pl":
        n_layers = _require_model_int(model_config, "n_layers", experiment)
        n_nodes = _require_model_int(model_config, "n_nodes", experiment)
        q_layers = _require_model_int(model_config, "q_layers", experiment)

        from .DEE.dee_hy_pl import run

        run(
            mode=mode,
            backend=backend,
            model_size=_build_model_size(n_nodes, n_layers, q_layers),
        )
        return

    if experiment == "dee-qq-m":
        n_photons = _require_model_int(model_config, "n_photons", experiment)

        from .DEE.dee_qq_m import run

        run(
            mode=mode,
            backend=backend,
            n_photons=n_photons,
        )
        return

    if experiment == "dee-qq-pl":
        q_layers = _require_model_int(model_config, "q_layers", experiment)

        from .DEE.dee_qq_pl import run

        run(
            mode=mode,
            backend=backend,
            model_size=_build_model_size(q_layers),
        )
        return

    if experiment == "taf-cc":
        n_layers = _require_model_int(model_config, "n_layers", experiment)
        n_nodes = _require_model_int(model_config, "n_nodes", experiment)

        from .TAF.taf_cc import run

        run(
            mode=mode,
            backend=backend,
            n_layers=n_layers,
            n_nodes=n_nodes,
        )
        return

    if experiment == "taf-hy-m":
        n_layers = _require_model_int(model_config, "n_layers", experiment)
        n_nodes = _require_model_int(model_config, "n_nodes", experiment)
        n_photons = _require_model_int(model_config, "n_photons", experiment)

        from .TAF.taf_hy_m import run

        run(
            mode=mode,
            backend=backend,
            model_size=_build_model_size(n_nodes, n_layers, n_photons),
        )
        return

    if experiment == "taf-hy-pl":
        n_layers = _require_model_int(model_config, "n_layers", experiment)
        n_nodes = _require_model_int(model_config, "n_nodes", experiment)
        q_layers = _require_model_int(model_config, "q_layers", experiment)

        from .TAF.taf_hy_pl import run

        run(
            mode=mode,
            backend=backend,
            model_size=_build_model_size(n_nodes, n_layers, q_layers),
        )
        return

    if experiment == "taf-qq-m":
        n_photons = _require_model_int(model_config, "n_photons", experiment)

        from .TAF.taf_qq_m import run

        run(
            mode=mode,
            backend=backend,
            n_photons=n_photons,
        )
        return

    if experiment == "taf-qq-pl":
        q_layers = _require_model_int(model_config, "q_layers", experiment)

        from .TAF.taf_qq_pl import run

        run(
            mode=mode,
            backend=backend,
            model_size=_build_model_size(q_layers),
        )
        return

    raise NotImplementedError(
        "Unsupported config-driven experiment: "
        f"{experiment}. Expected one of the DHO, SEE, DEE, or TAF variants."
    )


def train_and_evaluate(cfg: Mapping[str, Any], run_dir: str | Path) -> dict[str, Any]:
    """Run one resolved HQPINN experiment through the shared runtime.

    Parameters
    ----------
    cfg : Mapping[str, Any]
        Resolved experiment configuration supplied by ``implementation.py``.
    run_dir : str | pathlib.Path
        Timestamped raw output directory created by the shared runtime.

    Returns
    -------
    dict[str, Any]
        Minimal run metadata for callers and tests.
    """
    resolved_run_dir = Path(run_dir).resolve()
    resolved_run_dir.mkdir(parents=True, exist_ok=True)

    config = _merge_dicts(_load_json(DEFAULT_CONFIG_PATH), dict(cfg))
    config = _normalize_config(config)
    shared_runner_config = dict(config.get("shared_runner") or {})
    shared_runner_config["run_dir"] = str(resolved_run_dir)
    config["shared_runner"] = shared_runner_config

    run_from_project(config)

    return {
        "status": "completed",
        "run_dir": str(resolved_run_dir),
        "experiment": config.get("experiment"),
        "mode": config.get("mode"),
        "backend": config.get("backend"),
    }
