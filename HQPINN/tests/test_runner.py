from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from HQPINN import run_from_project
from HQPINN import runner


CONFIGS_DIR = Path("HQPINN/configs")

EXPERIMENT_MODULES = {
    "dho-cc": "HQPINN.DHO.dho_cc",
    "dho-hy-pl": "HQPINN.DHO.dho_hy_pl",
    "dho-qq-pl": "HQPINN.DHO.dho_qq_pl",
    "dho-hy-m": "HQPINN.DHO.dho_hy_m",
    "dho-hy-mp": "HQPINN.DHO.dho_hy_mp",
    "dho-qq-m": "HQPINN.DHO.dho_qq_m",
    "dho-qq-mp": "HQPINN.DHO.dho_qq_mp",
    "see-cc": "HQPINN.SEE.see_cc",
    "see-hy-m": "HQPINN.SEE.see_hy_m",
    "see-hy-pl": "HQPINN.SEE.see_hy_pl",
    "see-qq-m": "HQPINN.SEE.see_qq_m",
    "see-qq-pl": "HQPINN.SEE.see_qq_pl",
    "dee-cc": "HQPINN.DEE.dee_cc",
    "dee-hy-m": "HQPINN.DEE.dee_hy_m",
    "dee-hy-pl": "HQPINN.DEE.dee_hy_pl",
    "dee-qq-m": "HQPINN.DEE.dee_qq_m",
    "dee-qq-pl": "HQPINN.DEE.dee_qq_pl",
    "taf-cc": "HQPINN.TAF.taf_cc",
    "taf-hy-m": "HQPINN.TAF.taf_hy_m",
    "taf-hy-pl": "HQPINN.TAF.taf_hy_pl",
    "taf-qq-m": "HQPINN.TAF.taf_qq_m",
    "taf-qq-pl": "HQPINN.TAF.taf_qq_pl",
}

LAYER_NODE_EXPERIMENTS = {"dho-cc", "dho-hy-mp", "see-cc", "dee-cc", "taf-cc"}
LAYER_NODE_PHOTON_EXPERIMENTS = {"dho-hy-m", "see-hy-m"}
LAYER_NODE_Q_LAYER_EXPERIMENTS = {"see-hy-pl"}
PHOTON_EXPERIMENTS = {"dho-qq-m", "see-qq-m", "dee-qq-m", "taf-qq-m"}
Q_LAYER_EXPERIMENTS = {"see-qq-pl"}
MODEL_SIZE_LAYER_NODE_PHOTON_EXPERIMENTS = {"dee-hy-m", "taf-hy-m"}
MODEL_SIZE_LAYER_NODE_Q_LAYER_EXPERIMENTS = {"dee-hy-pl", "taf-hy-pl"}
MODEL_SIZE_Q_LAYER_EXPERIMENTS = {"dee-qq-pl", "taf-qq-pl"}


def _copy_model_ints(model: dict, *keys: str) -> dict:
    return {key: int(model[key]) for key in keys}


def _build_model_size(model: dict, *keys: str) -> str:
    return "-".join(str(int(model[key])) for key in keys)


def _expected_kwargs(config: dict) -> dict:
    experiment = config["experiment"]
    model = config.get("model") or {}
    kwargs = {
        "mode": config["mode"],
        "backend": config["backend"],
    }

    if experiment == "dho-hy-pl":
        kwargs |= _copy_model_ints(model, "n_layers", "n_nodes", "n_qubits")
        return kwargs

    if experiment == "dho-qq-pl":
        kwargs |= _copy_model_ints(model, "n_qubits")
        return kwargs

    if experiment in LAYER_NODE_EXPERIMENTS:
        kwargs |= _copy_model_ints(model, "n_layers", "n_nodes")
        return kwargs

    if experiment in LAYER_NODE_PHOTON_EXPERIMENTS:
        kwargs |= _copy_model_ints(model, "n_layers", "n_nodes", "n_photons")
        return kwargs

    if experiment in LAYER_NODE_Q_LAYER_EXPERIMENTS:
        kwargs |= _copy_model_ints(model, "n_layers", "n_nodes", "q_layers")
        return kwargs

    if experiment in PHOTON_EXPERIMENTS:
        kwargs |= _copy_model_ints(model, "n_photons")
        return kwargs

    if experiment in Q_LAYER_EXPERIMENTS:
        kwargs |= _copy_model_ints(model, "q_layers")
        return kwargs

    if experiment in MODEL_SIZE_LAYER_NODE_PHOTON_EXPERIMENTS:
        kwargs["model_size"] = _build_model_size(
            model, "n_nodes", "n_layers", "n_photons"
        )
        return kwargs

    if experiment in MODEL_SIZE_LAYER_NODE_Q_LAYER_EXPERIMENTS:
        kwargs["model_size"] = _build_model_size(
            model, "n_nodes", "n_layers", "q_layers"
        )
        return kwargs

    if experiment in MODEL_SIZE_Q_LAYER_EXPERIMENTS:
        kwargs["model_size"] = _build_model_size(model, "q_layers")
        return kwargs

    return kwargs


class RunnerDispatchTests(unittest.TestCase):
    def test_all_json_configs_dispatch_via_run_from_project(self) -> None:
        config_paths = sorted(
            path for path in CONFIGS_DIR.glob("*.json") if path.name != "defaults.json"
        )
        self.assertTrue(
            config_paths, "Expected HQPINN/configs to contain runnable configs"
        )

        for config_path in config_paths:
            config = runner._load_config(str(config_path))
            experiment = config["experiment"]
            module_name = EXPERIMENT_MODULES[experiment]
            expected = _expected_kwargs(config)
            calls: list[dict] = []
            stub_module = types.ModuleType(module_name)

            def _stub_run(**kwargs):
                calls.append(kwargs)

            stub_module.run = _stub_run

            with self.subTest(config=config_path.name):
                with patch.dict(sys.modules, {module_name: stub_module}):
                    run_from_project(config)
                self.assertEqual(calls, [expected])
