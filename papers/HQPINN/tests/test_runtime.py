from __future__ import annotations

import unittest

import torch

import HQPINN
import HQPINN.config as project_config
from HQPINN.dtypes import DtypeSpec, coerce_dtype_spec, dtype_label, dtype_torch
from HQPINN.runtime import apply_runtime_config, normalize_dtype_config


class RuntimeDtypeTests(unittest.TestCase):
    def test_package_exports_public_entrypoint(self) -> None:
        self.assertEqual(HQPINN.run_from_project.__name__, "run_from_project")

    def test_dtype_aliases_are_normalized(self) -> None:
        spec = coerce_dtype_spec("double")
        self.assertEqual(spec, DtypeSpec(label="float64", torch_dtype=torch.float64))
        self.assertEqual(dtype_label("float"), "float32")
        self.assertEqual(dtype_torch("cfloat"), torch.complex64)

    def test_normalize_dtype_config_returns_deep_copied_specs(self) -> None:
        raw = {
            "dtype": "float32",
            "model": {
                "weights_dtype": "float64",
            },
        }
        normalized = normalize_dtype_config(raw)

        self.assertEqual(raw["dtype"], "float32")
        self.assertIsInstance(normalized["dtype"], DtypeSpec)
        self.assertEqual(normalized["dtype"].label, "float32")
        self.assertEqual(
            normalized["model"]["weights_dtype"].torch_dtype, torch.float64
        )

    def test_apply_runtime_config_updates_project_dtype(self) -> None:
        initial_dtype = project_config.DTYPE
        try:
            normalized = apply_runtime_config({"dtype": "float32"})
            self.assertEqual(normalized["dtype"].torch_dtype, torch.float32)
            self.assertEqual(project_config.DTYPE, torch.float32)
        finally:
            project_config.set_dtype(initial_dtype)
