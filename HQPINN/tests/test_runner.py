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
    "dho-cp": "HQPINN.DHO.dho_cp",
    "dho-pp": "HQPINN.DHO.dho_pp",
    "dho-ci": "HQPINN.DHO.dho_ci",
    "dho-cperc": "HQPINN.DHO.dho_cperc",
    "dho-ii": "HQPINN.DHO.dho_ii",
    "dho-percperc": "HQPINN.DHO.dho_percperc",
    "see-cc": "HQPINN.SEE.see_cc",
    "see-ci": "HQPINN.SEE.see_ci",
    "see-cp": "HQPINN.SEE.see_cp",
    "see-ii": "HQPINN.SEE.see_ii",
    "see-pp": "HQPINN.SEE.see_pp",
    "dee-cc": "HQPINN.DEE.dee_cc",
    "dee-ci": "HQPINN.DEE.dee_ci",
    "dee-cp": "HQPINN.DEE.dee_cp",
    "dee-ii": "HQPINN.DEE.dee_ii",
    "dee-pp": "HQPINN.DEE.dee_pp",
    "taf-cc": "HQPINN.TAF.taf_cc",
    "taf-ci": "HQPINN.TAF.taf_ci",
    "taf-cp": "HQPINN.TAF.taf_cp",
    "taf-ii": "HQPINN.TAF.taf_ii",
    "taf-pp": "HQPINN.TAF.taf_pp",
}

LAYER_NODE_EXPERIMENTS = {"dho-cc", "dho-cperc", "see-cc", "dee-cc", "taf-cc"}
LAYER_NODE_PHOTON_EXPERIMENTS = {"dho-ci", "see-ci"}
LAYER_NODE_Q_LAYER_EXPERIMENTS = {"see-cp"}
PHOTON_EXPERIMENTS = {"dho-ii", "see-ii", "dee-ii", "taf-ii"}
Q_LAYER_EXPERIMENTS = {"see-pp"}
MODEL_SIZE_LAYER_NODE_PHOTON_EXPERIMENTS = {"dee-ci", "taf-ci"}
MODEL_SIZE_LAYER_NODE_Q_LAYER_EXPERIMENTS = {"dee-cp", "taf-cp"}
MODEL_SIZE_Q_LAYER_EXPERIMENTS = {"dee-pp", "taf-pp"}


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

    if experiment == "dho-cp":
        kwargs |= _copy_model_ints(model, "n_layers", "n_nodes", "n_qubits")
        return kwargs

    if experiment == "dho-pp":
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
