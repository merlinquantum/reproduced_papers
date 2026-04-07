from __future__ import annotations

import unittest
from unittest.mock import patch

from HQPINN.lib.DEE import dee_cc
from HQPINN.lib.TAF import taf_cc


class InferenceModelSelectionTests(unittest.TestCase):
    def test_taf_cc_explicit_width_and_depth_override_default_model_size(self) -> None:
        for mode in ("run", "remote"):
            captured: dict[str, object] = {}

            def _capture(**kwargs) -> None:
                captured.update(kwargs)

            with self.subTest(mode=mode):
                with patch(
                    "HQPINN.lib.TAF.taf_cc.run_density_inference_mode",
                    side_effect=_capture,
                ):
                    taf_cc.run(mode=mode, backend="local", n_nodes=40, n_layers=7)

                self.assertEqual(captured["case_prefix"], "taf_cc_40-7")

    def test_dee_cc_explicit_width_and_depth_override_default_model_size(self) -> None:
        for mode in ("run", "remote"):
            captured: dict[str, object] = {}

            def _capture(**kwargs) -> None:
                captured.update(kwargs)

            with self.subTest(mode=mode):
                with patch(
                    "HQPINN.lib.DEE.dee_cc.run_density_inference_mode",
                    side_effect=_capture,
                ):
                    dee_cc.run(mode=mode, backend="local", n_nodes=10, n_layers=7)

                self.assertEqual(captured["case_prefix"], "dee_cc_10-7")
