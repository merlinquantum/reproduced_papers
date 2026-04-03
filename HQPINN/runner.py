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
        )
        return

    if experiment == "dho-cp":
        n_layers = model_config.get("n_layers")
        n_nodes = model_config.get("n_nodes")
        n_qubits = model_config.get("n_qubits")
        if n_layers is None or n_nodes is None or n_qubits is None:
            raise ValueError(
                "dho-cp config requires model.n_layers, model.n_nodes, and model.n_qubits"
            )

        from .DHO.dho_cp import run

        run(
            mode=mode,
            backend=backend,
            n_layers=int(n_layers),
            n_nodes=int(n_nodes),
            n_qubits=int(n_qubits),
        )
        return

    if experiment == "dho-pp":
        n_qubits = model_config.get("n_qubits")
        if n_qubits is None:
            raise ValueError("dho-pp config requires model.n_qubits")

        from .DHO.dho_pp import run

        run(
            mode=mode,
            backend=backend,
            n_qubits=int(n_qubits),
        )
        return

    if experiment == "dho-ci":
        n_layers = model_config.get("n_layers")
        n_nodes = model_config.get("n_nodes")
        n_photons = model_config.get("n_photons")
        if n_layers is None or n_nodes is None or n_photons is None:
            raise ValueError(
                "dho-ci config requires model.n_layers, model.n_nodes, and model.n_photons"
            )

        from .DHO.dho_ci import run

        run(
            mode=mode,
            backend=backend,
            n_layers=int(n_layers),
            n_nodes=int(n_nodes),
            n_photons=int(n_photons),
        )
        return

    if experiment == "dho-cperc":
        n_layers = model_config.get("n_layers")
        n_nodes = model_config.get("n_nodes")
        if n_layers is None or n_nodes is None:
            raise ValueError(
                "dho-cperc config requires model.n_layers and model.n_nodes"
            )

        from .DHO.dho_cperc import run

        run(
            mode=mode,
            backend=backend,
            n_layers=int(n_layers),
            n_nodes=int(n_nodes),
        )
        return

    if experiment == "dho-ii":
        n_photons = model_config.get("n_photons")
        if n_photons is None:
            raise ValueError("dho-ii config requires model.n_photons")

        from .DHO.dho_ii import run

        run(
            mode=mode,
            backend=backend,
            n_photons=int(n_photons),
        )
        return

    if experiment == "dho-percperc":
        from .DHO.dho_percperc import run

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

    if experiment == "see-ci":
        n_layers = model_config.get("n_layers")
        n_nodes = model_config.get("n_nodes")
        n_photons = model_config.get("n_photons")
        if n_layers is None or n_nodes is None or n_photons is None:
            raise ValueError(
                "see-ci config requires model.n_layers, model.n_nodes, and model.n_photons"
            )

        from .SEE.see_ci import run

        run(
            mode=mode,
            backend=backend,
            n_layers=int(n_layers),
            n_nodes=int(n_nodes),
            n_photons=int(n_photons),
        )
        return

    if experiment == "see-cp":
        n_layers = model_config.get("n_layers")
        n_nodes = model_config.get("n_nodes")
        q_layers = model_config.get("q_layers")
        if n_layers is None or n_nodes is None or q_layers is None:
            raise ValueError(
                "see-cp config requires model.n_layers, model.n_nodes, and model.q_layers"
            )

        from .SEE.see_cp import run

        run(
            mode=mode,
            backend=backend,
            n_layers=int(n_layers),
            n_nodes=int(n_nodes),
            q_layers=int(q_layers),
        )
        return

    if experiment == "see-ii":
        n_photons = model_config.get("n_photons")
        if n_photons is None:
            raise ValueError("see-ii config requires model.n_photons")

        from .SEE.see_ii import run

        run(
            mode=mode,
            backend=backend,
            n_photons=int(n_photons),
        )
        return

    if experiment == "see-pp":
        q_layers = model_config.get("q_layers")
        if q_layers is None:
            raise ValueError("see-pp config requires model.q_layers")

        from .SEE.see_pp import run

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

    if experiment == "dee-ci":
        n_layers = _require_model_int(model_config, "n_layers", experiment)
        n_nodes = _require_model_int(model_config, "n_nodes", experiment)
        n_photons = _require_model_int(model_config, "n_photons", experiment)

        from .DEE.dee_ci import run

        run(
            mode=mode,
            backend=backend,
            model_size=_build_model_size(n_nodes, n_layers, n_photons),
        )
        return

    if experiment == "dee-cp":
        n_layers = _require_model_int(model_config, "n_layers", experiment)
        n_nodes = _require_model_int(model_config, "n_nodes", experiment)
        q_layers = _require_model_int(model_config, "q_layers", experiment)

        from .DEE.dee_cp import run

        run(
            mode=mode,
            backend=backend,
            model_size=_build_model_size(n_nodes, n_layers, q_layers),
        )
        return

    if experiment == "dee-ii":
        n_photons = _require_model_int(model_config, "n_photons", experiment)

        from .DEE.dee_ii import run

        run(
            mode=mode,
            backend=backend,
            n_photons=n_photons,
        )
        return

    if experiment == "dee-pp":
        q_layers = _require_model_int(model_config, "q_layers", experiment)

        from .DEE.dee_pp import run

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

    if experiment == "taf-ci":
        n_layers = _require_model_int(model_config, "n_layers", experiment)
        n_nodes = _require_model_int(model_config, "n_nodes", experiment)
        n_photons = _require_model_int(model_config, "n_photons", experiment)

        from .TAF.taf_ci import run

        run(
            mode=mode,
            backend=backend,
            model_size=_build_model_size(n_nodes, n_layers, n_photons),
        )
        return

    if experiment == "taf-cp":
        n_layers = _require_model_int(model_config, "n_layers", experiment)
        n_nodes = _require_model_int(model_config, "n_nodes", experiment)
        q_layers = _require_model_int(model_config, "q_layers", experiment)

        from .TAF.taf_cp import run

        run(
            mode=mode,
            backend=backend,
            model_size=_build_model_size(n_nodes, n_layers, q_layers),
        )
        return

    if experiment == "taf-ii":
        n_photons = _require_model_int(model_config, "n_photons", experiment)

        from .TAF.taf_ii import run

        run(
            mode=mode,
            backend=backend,
            n_photons=n_photons,
        )
        return

    if experiment == "taf-pp":
        q_layers = _require_model_int(model_config, "q_layers", experiment)

        from .TAF.taf_pp import run

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
    print("  dho-cc          -> DHO, Classical–Classical)")
    print("  dho-ci          -> DHO, Classical-Interferometer")
    print("  dho-cp          -> DHO, Classical–PennyLane")
    print("  dho-cperc       -> DHO, Classical–Perceval")
    print("  dho-ii          -> DHO, Interferometer–Interferometer")
    print("  dho-percperc    -> DHO, Perceval–Perceval")
    print("  dho-pp          -> DHO, PennyLane–PennyLane")
    print("  see-cc          -> SEE, Classical–Classical")
    print("  see-ci          -> SEE, Classical–Interferometer")
    print("  see-cp          -> SEE, Classical–PennyLane")
    print("  see-ii          -> SEE, Interferometer–Interferometer")
    print("  see-pp          -> SEE, PennyLane–PennyLane")
    print("  dee-cc          -> DEE, Classical–Classical")
    print("  dee-ci          -> DEE, Classical–Interferometer")
    print("  dee-cp          -> DEE, Classical–PennyLane")
    print("  dee-ii          -> DEE, Interferometer–Interferometer")
    print("  dee-pp          -> DEE, PennyLane–PennyLane")
    print("  taf-cc          -> TAF, Classical–Classical")
    print("  taf-ci          -> TAF, Classical–Interferometer")
    print("  taf-cp          -> TAF, Classical–PennyLane")
    print("  taf-ii          -> TAF, Interferometer–Interferometer")
    print("  taf-pp          -> TAF, PennyLane–PennyLane")

    print()
    choice = input("Which experiment do you want to run? ").strip()

    # DHO experiments
    if choice == "dho-pp":
        from .DHO.dho_pp import run

        mode = _ask_mode()
        backend = _ask_backend(mode)
        run(mode=mode, backend=backend)

    elif choice == "dho-cc":
        from .DHO.dho_cc import run

        mode = _ask_mode()
        backend = _ask_backend(mode)
        run(mode=mode, backend=backend)

    elif choice == "dho-cp":
        from .DHO.dho_cp import run

        mode = _ask_mode()
        backend = _ask_backend(mode)
        run(mode=mode, backend=backend)

    elif choice == "dho-cperc":
        from .DHO.dho_cperc import run

        mode = _ask_mode()
        backend = _ask_backend(mode)
        run(mode=mode, backend=backend)

    elif choice == "dho-ii":
        from .DHO.dho_ii import run

        mode = _ask_mode()
        backend = _ask_backend(mode)
        run(mode=mode, backend=backend)

    elif choice == "dho-percperc":
        from .DHO.dho_percperc import run

        mode = _ask_mode()
        backend = _ask_backend(mode)
        run(mode=mode, backend=backend)

    elif choice == "dho-ci":
        from .DHO.dho_ci import run

        mode = _ask_mode()
        backend = _ask_backend(mode)
        run(mode=mode, backend=backend)

    # SEE experiments
    elif choice == "see-cc":
        from .SEE.see_cc import run

        mode = _ask_mode()
        backend = _ask_backend(mode)
        if mode == "train":
            run(mode=mode, backend=backend)
        else:
            model_size = input("Model size? [10-4/10-7/20-4] ").strip() or "10-4"
            run(mode=mode, backend=backend, model_size=model_size)

    elif choice == "see-cp":
        from .SEE.see_cp import run

        mode = _ask_mode()
        backend = _ask_backend(mode)
        if mode == "train":
            run(mode=mode, backend=backend)
        else:
            model_size = (
                input("Model size? [10-4-2/10-7-2/20-4-2] ").strip() or "10-4-2"
            )
            run(mode=mode, backend=backend, model_size=model_size)

    elif choice == "see-ii":
        from .SEE.see_ii import run

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

    elif choice == "see-pp":
        from .SEE.see_pp import run

        mode = _ask_mode()
        backend = _ask_backend(mode)
        if mode == "train":
            run(mode=mode, backend=backend)
        else:
            model_size = input("Model size? [2/3/4] ").strip() or "2"
            run(mode=mode, backend=backend, model_size=model_size)

    elif choice == "see-ci":
        from .SEE.see_ci import run

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
        from .DEE.dee_cc import run

        mode = _ask_mode()
        backend = _ask_backend(mode)
        if mode == "train":
            run(mode=mode, backend=backend)
        else:
            model_size = input("Model size? [10-4/10-7/20-4] ").strip() or "10-4"
            run(mode=mode, backend=backend, model_size=model_size)

    elif choice == "dee-ii":
        from .DEE.dee_ii import run

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

    elif choice == "dee-ci":
        from .DEE.dee_ci import run

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

    elif choice == "dee-cp":
        from .DEE.dee_cp import run

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

    elif choice == "dee-pp":
        from .DEE.dee_pp import run

        mode = _ask_mode()
        backend = _ask_backend(mode)
        if mode == "train":
            run(mode=mode, backend=backend)
        else:
            model_size = input("Model size? [2/3/4] ").strip() or "2"
            run(mode=mode, backend=backend, model_size=model_size)

    # TAF experiments
    elif choice == "taf-cc":
        from .TAF.taf_cc import run

        mode = _ask_mode()
        backend = _ask_backend(mode)
        if mode == "train":
            run(mode=mode, backend=backend)
        else:
            model_size = input("Model size? [40-4/40-7/80-4] ").strip() or "40-4"
            run(mode=mode, backend=backend, model_size=model_size)

    elif choice == "taf-ci":
        from .TAF.taf_ci import run

        mode = _ask_mode()
        backend = _ask_backend(mode)
        if mode == "train":
            run(mode=mode, backend=backend)
        else:
            model_size = (
                input("Model size? [40-4-2/40-7-2/80-4-2] ").strip() or "40-4-2"
            )
            run(mode=mode, backend=backend, model_size=model_size)

    elif choice == "taf-cp":
        from .TAF.taf_cp import run

        mode = _ask_mode()
        backend = _ask_backend(mode)
        if mode == "train":
            run(mode=mode, backend=backend)
        else:
            model_size = (
                input("Model size? [40-4-2/40-7-2/80-4-2] ").strip() or "40-4-2"
            )
            run(mode=mode, backend=backend, model_size=model_size)

    elif choice == "taf-ii":
        from .TAF.taf_ii import run

        mode = _ask_mode()
        backend = _ask_backend(mode)
        if mode == "train":
            run(mode=mode, backend=backend)
        else:
            n_photons = int(input("Number of photons? [1/../6] ").strip() or "2")
            run(mode=mode, backend=backend, n_photons=n_photons)

    elif choice == "taf-pp":
        from .TAF.taf_pp import run

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
            "Please choose one of: dho-pp, dho-cc, dho-cp, dho-cperc, "
            "dho-ii, dho-percperc, dho-ci, see-cc, see-pp, see-ci, see-ii, see-cp, "
            "dee-cc, dee-ci, dee-cp, dee-ii, dee-pp, taf-cc, taf-ci, taf-cp, taf-ii, taf-pp."
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
