from __future__ import annotations

import unittest
from pathlib import Path

from HQPINN.paths import (
    RESULTS_ROOT,
    results_case_dir_for_model_dir,
    results_dir_for_model_dir,
)


class PathHelperTests(unittest.TestCase):
    def test_results_dir_for_model_dir_maps_benchmark_root(self) -> None:
        resolved = results_dir_for_model_dir("HQPINN/models/TAF")
        self.assertEqual(Path(resolved), RESULTS_ROOT / "TAF")

    def test_results_case_dir_for_model_dir_appends_case_prefix(self) -> None:
        resolved = results_case_dir_for_model_dir(
            "HQPINN/models/DEE",
            "dee_cc_10-4",
        )
        self.assertEqual(Path(resolved), RESULTS_ROOT / "DEE" / "dee_cc_10-4")
