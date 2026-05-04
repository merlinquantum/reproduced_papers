"""
Simple CLI dispatcher for HQPINN experiments.

Usage:
    python -m HQPINN
    python -m HQPINN --config HQPINN/configs/dho_cc_run.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .runtime import (
    DEFAULT_CONFIG_PATH,
    apply_runtime_config,
    configure_logging,
    log_run_banner,
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


def _load_config(config_path: str) -> dict[str, Any]:
    defaults = _load_json(DEFAULT_CONFIG_PATH)

    path = Path(config_path)
    if not path.is_absolute():
        path = Path.cwd() / path
    config = _load_json(path)
    merged = _merge_dicts(defaults, config)

    experiment = merged.get("experiment")
    if not experiment:
        raise ValueError("Config must define 'experiment'")
    if not isinstance(experiment, str):
        raise ValueError("Config 'experiment' must be a string")
    merged["experiment"] = _canonical_experiment_name(experiment)

    mode = merged.get("mode")
    if mode not in {"train", "run", "remote"}:
        raise ValueError("Config 'mode' must be 'train', 'run', or 'remote'")

    backend = merged.get("backend")
    if not isinstance(backend, str) or not backend:
        raise ValueError("Config must define a non-empty string 'backend'")

    return merged


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
    config = apply_runtime_config(config)
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

        from .lib.DHO.dho_cc import run

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

        from .lib.DHO.dho_hy_pl import run

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

        from .lib.DHO.dho_qq_pl import run

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

        from .lib.DHO.dho_hy_m import run

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

        from .lib.DHO.dho_hy_mp import run

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

        from .lib.DHO.dho_qq_m import run

        run(
            mode=mode,
            backend=backend,
            n_photons=int(n_photons),
            **({"force_retrain": True} if force_retrain else {}),
        )
        return

    if experiment == "dho-qq-mp":
        from .lib.DHO.dho_qq_mp import run

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

        from .lib.SEE.see_cc import run

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

        from .lib.SEE.see_hy_m import run

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

        from .lib.SEE.see_hy_pl import run

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

        from .lib.SEE.see_qq_m import run

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

        from .lib.SEE.see_qq_pl import run

        run(
            mode=mode,
            backend=backend,
            q_layers=int(q_layers),
        )
        return

    if experiment == "dee-cc":
        n_layers = _require_model_int(model_config, "n_layers", experiment)
        n_nodes = _require_model_int(model_config, "n_nodes", experiment)

        from .lib.DEE.dee_cc import run

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

        from .lib.DEE.dee_hy_m import run

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

        from .lib.DEE.dee_hy_pl import run

        run(
            mode=mode,
            backend=backend,
            model_size=_build_model_size(n_nodes, n_layers, q_layers),
        )
        return

    if experiment == "dee-qq-m":
        n_photons = _require_model_int(model_config, "n_photons", experiment)

        from .lib.DEE.dee_qq_m import run

        run(
            mode=mode,
            backend=backend,
            n_photons=n_photons,
        )
        return

    if experiment == "dee-qq-pl":
        q_layers = _require_model_int(model_config, "q_layers", experiment)

        from .lib.DEE.dee_qq_pl import run

        run(
            mode=mode,
            backend=backend,
            model_size=_build_model_size(q_layers),
        )
        return

    if experiment == "taf-cc":
        n_layers = _require_model_int(model_config, "n_layers", experiment)
        n_nodes = _require_model_int(model_config, "n_nodes", experiment)

        from .lib.TAF.taf_cc import run

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

        from .lib.TAF.taf_hy_m import run

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

        from .lib.TAF.taf_hy_pl import run

        run(
            mode=mode,
            backend=backend,
            model_size=_build_model_size(n_nodes, n_layers, q_layers),
        )
        return

    if experiment == "taf-qq-m":
        n_photons = _require_model_int(model_config, "n_photons", experiment)

        from .lib.TAF.taf_qq_m import run

        run(
            mode=mode,
            backend=backend,
            n_photons=n_photons,
        )
        return

    if experiment == "taf-qq-pl":
        q_layers = _require_model_int(model_config, "q_layers", experiment)

        from .lib.TAF.taf_qq_pl import run

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


def _ask_mode() -> str:
    mode = input("Mode? [train/run/remote] ").strip().lower()
    if mode not in {"train", "run", "remote"}:
        raise ValueError("mode must be 'train', 'run', or 'remote'")
    return mode


def _ask_backend(mode: str) -> str:
    if mode == "remote":
        backend = input("Backend? [sim:ascella] ").strip()
        return backend or "sim:ascella"
    return "local"


def _run_interactive() -> None:
    print("Available experiments:")
    print("  dho-cc          -> DHO, Classical-Classical")
    print("  dho-hy-m        -> DHO, Hybrid Merlin")
    print("  dho-hy-pl       -> DHO, Hybrid PennyLane")
    print("  dho-hy-mp       -> DHO, Hybrid Merlin-Perceval")
    print("  dho-qq-m        -> DHO, Quantum-Quantum Merlin")
    print("  dho-qq-mp       -> DHO, Quantum-Quantum Merlin-Perceval")
    print("  dho-qq-pl       -> DHO, Quantum-Quantum PennyLane")
    print("  see-cc          -> SEE, Classical-Classical")
    print("  see-hy-m        -> SEE, Hybrid Merlin")
    print("  see-hy-pl       -> SEE, Hybrid PennyLane")
    print("  see-qq-m        -> SEE, Quantum-Quantum Merlin")
    print("  see-qq-pl       -> SEE, Quantum-Quantum PennyLane")
    print("  dee-cc          -> DEE, Classical-Classical")
    print("  dee-hy-m        -> DEE, Hybrid Merlin")
    print("  dee-hy-pl       -> DEE, Hybrid PennyLane")
    print("  dee-qq-m        -> DEE, Quantum-Quantum Merlin")
    print("  dee-qq-pl       -> DEE, Quantum-Quantum PennyLane")
    print("  taf-cc          -> TAF, Classical-Classical")
    print("  taf-hy-m        -> TAF, Hybrid Merlin")
    print("  taf-hy-pl       -> TAF, Hybrid PennyLane")
    print("  taf-qq-m        -> TAF, Quantum-Quantum Merlin")
    print("  taf-qq-pl       -> TAF, Quantum-Quantum PennyLane")

    print()
    choice = _canonical_experiment_name(input("Which experiment do you want to run? ").strip())

    # DHO experiments
    if choice == "dho-qq-pl":
        from .lib.DHO.dho_qq_pl import run

        mode = _ask_mode()
        backend = _ask_backend(mode)
        run(mode=mode, backend=backend)

    elif choice == "dho-cc":
        from .lib.DHO.dho_cc import run

        mode = _ask_mode()
        backend = _ask_backend(mode)
        run(mode=mode, backend=backend)

    elif choice == "dho-hy-pl":
        from .lib.DHO.dho_hy_pl import run

        mode = _ask_mode()
        backend = _ask_backend(mode)
        run(mode=mode, backend=backend)

    elif choice == "dho-hy-mp":
        from .lib.DHO.dho_hy_mp import run

        mode = _ask_mode()
        backend = _ask_backend(mode)
        run(mode=mode, backend=backend)

    elif choice == "dho-qq-m":
        from .lib.DHO.dho_qq_m import run

        mode = _ask_mode()
        backend = _ask_backend(mode)
        run(mode=mode, backend=backend)

    elif choice == "dho-qq-mp":
        from .lib.DHO.dho_qq_mp import run

        mode = _ask_mode()
        backend = _ask_backend(mode)
        run(mode=mode, backend=backend)

    elif choice == "dho-hy-m":
        from .lib.DHO.dho_hy_m import run

        mode = _ask_mode()
        backend = _ask_backend(mode)
        run(mode=mode, backend=backend)

    # SEE experiments
    elif choice == "see-cc":
        from .lib.SEE.see_cc import run

        mode = _ask_mode()
        backend = _ask_backend(mode)
        if mode == "train":
            run(mode=mode, backend=backend)
        else:
            model_size = input("Model size? [10-4/10-7/20-4] ").strip() or "10-4"
            run(mode=mode, backend=backend, model_size=model_size)

    elif choice == "see-hy-pl":
        from .lib.SEE.see_hy_pl import run

        mode = _ask_mode()
        backend = _ask_backend(mode)
        if mode == "train":
            run(mode=mode, backend=backend)
        else:
            model_size = (
                input("Model size? [10-4-2/10-7-2/20-4-2] ").strip() or "10-4-2"
            )
            run(mode=mode, backend=backend, model_size=model_size)

    elif choice == "see-qq-m":
        from .lib.SEE.see_qq_m import run

        mode = _ask_mode()
        backend = _ask_backend(mode)
        if mode == "train":
            run(mode=mode, backend=backend)
        else:
            n_photons = int(input("Number of photons? [1/../6] ").strip() or "2")
            run(
                mode=mode,
                backend=backend,
                n_photons=n_photons,
            )

    elif choice == "see-qq-pl":
        from .lib.SEE.see_qq_pl import run

        mode = _ask_mode()
        backend = _ask_backend(mode)
        if mode == "train":
            run(mode=mode, backend=backend)
        else:
            model_size = input("Model size? [2/3/4] ").strip() or "2"
            run(mode=mode, backend=backend, model_size=model_size)

    elif choice == "see-hy-m":
        from .lib.SEE.see_hy_m import run

        mode = _ask_mode()
        backend = _ask_backend(mode)
        if mode == "train":
            run(mode=mode, backend=backend)
        else:
            model_size = (
                input("Model size? [10-4-2/10-7-2/20-4-2] ").strip() or "10-4-2"
            )
            run(
                mode=mode,
                backend=backend,
                model_size=model_size,
            )

    # DEE experiments
    elif choice == "dee-cc":
        from .lib.DEE.dee_cc import run

        mode = _ask_mode()
        backend = _ask_backend(mode)
        if mode == "train":
            run(mode=mode, backend=backend)
        else:
            model_size = input("Model size? [10-4/10-7/20-4] ").strip() or "10-4"
            run(mode=mode, backend=backend, model_size=model_size)

    elif choice == "dee-qq-m":
        from .lib.DEE.dee_qq_m import run

        mode = _ask_mode()
        backend = _ask_backend(mode)
        if mode == "train":
            run(mode=mode, backend=backend)
        else:
            n_photons = int(input("Number of photons? [1/../6] ").strip() or "2")
            run(
                mode=mode,
                backend=backend,
                n_photons=n_photons,
            )

    elif choice == "dee-hy-m":
        from .lib.DEE.dee_hy_m import run

        mode = _ask_mode()
        backend = _ask_backend(mode)
        if mode == "train":
            run(mode=mode, backend=backend)
        else:
            model_size = (
                input("Model size? [10-4-1/10-7-1/20-4-1] ").strip() or "10-4-1"
            )
            run(
                mode=mode,
                backend=backend,
                model_size=model_size,
            )

    elif choice == "dee-hy-pl":
        from .lib.DEE.dee_hy_pl import run

        mode = _ask_mode()
        backend = _ask_backend(mode)
        if mode == "train":
            run(mode=mode, backend=backend)
        else:
            model_size = (
                input("Model size? [10-4-2/10-7-2/20-4-2] ").strip() or "10-4-2"
            )
            run(
                mode=mode,
                backend=backend,
                model_size=model_size,
            )

    elif choice == "dee-qq-pl":
        from .lib.DEE.dee_qq_pl import run

        mode = _ask_mode()
        backend = _ask_backend(mode)
        if mode == "train":
            run(mode=mode, backend=backend)
        else:
            model_size = input("Model size? [2/3/4] ").strip() or "2"
            run(mode=mode, backend=backend, model_size=model_size)

    # TAF experiments
    elif choice == "taf-cc":
        from .lib.TAF.taf_cc import run

        mode = _ask_mode()
        backend = _ask_backend(mode)
        if mode == "train":
            run(mode=mode, backend=backend)
        else:
            model_size = input("Model size? [40-4/40-7/80-4] ").strip() or "40-4"
            run(mode=mode, backend=backend, model_size=model_size)

    elif choice == "taf-hy-m":
        from .lib.TAF.taf_hy_m import run

        mode = _ask_mode()
        backend = _ask_backend(mode)
        if mode == "train":
            run(mode=mode, backend=backend)
        else:
            model_size = (
                input("Model size? [40-4-2/40-7-2/80-4-2] ").strip() or "40-4-2"
            )
            run(mode=mode, backend=backend, model_size=model_size)

    elif choice == "taf-hy-pl":
        from .lib.TAF.taf_hy_pl import run

        mode = _ask_mode()
        backend = _ask_backend(mode)
        if mode == "train":
            run(mode=mode, backend=backend)
        else:
            model_size = (
                input("Model size? [40-4-2/40-7-2/80-4-2] ").strip() or "40-4-2"
            )
            run(mode=mode, backend=backend, model_size=model_size)

    elif choice == "taf-qq-m":
        from .lib.TAF.taf_qq_m import run

        mode = _ask_mode()
        backend = _ask_backend(mode)
        if mode == "train":
            run(mode=mode, backend=backend)
        else:
            n_photons = int(input("Number of photons? [1/../6] ").strip() or "2")
            run(mode=mode, backend=backend, n_photons=n_photons)

    elif choice == "taf-qq-pl":
        from .lib.TAF.taf_qq_pl import run

        mode = _ask_mode()
        backend = _ask_backend(mode)
        if mode == "train":
            run(mode=mode, backend=backend)
        else:
            model_size = input("Model size? [2/4/6] ").strip() or "2"
            run(mode=mode, backend=backend, model_size=model_size)

    else:
        print(f"Unknown experiment: {choice}")
        print(
            "Please choose one of: dho-cc, dho-hy-m, dho-hy-pl, dho-hy-mp, "
            "dho-qq-m, dho-qq-mp, dho-qq-pl, see-cc, see-hy-m, see-hy-pl, see-qq-m, see-qq-pl, "
            "dee-cc, dee-hy-m, dee-hy-pl, dee-qq-m, dee-qq-pl, taf-cc, taf-hy-m, taf-hy-pl, taf-qq-m, taf-qq-pl."
        )


def main(argv: list[str] | None = None) -> None:
    """Entry point for the HQPINN command-line interface."""
    configure_logging()
    parser = argparse.ArgumentParser(description="Run HQPINN experiments.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a JSON config file. Currently supported for DHO and SEE variants.",
    )
    args = parser.parse_args(argv)

    if args.config is not None:
        config = _load_config(args.config)
        log_run_banner(config)
        run_from_project(config)
        return

    _run_interactive()
