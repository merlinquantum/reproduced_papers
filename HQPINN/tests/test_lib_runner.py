from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from HQPINN.lib.runner import train_and_evaluate


class SharedRunnerTests(unittest.TestCase):
    def test_train_and_evaluate_merges_defaults_and_creates_run_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "runs" / "dho-cc"

            with (
                patch("HQPINN.lib.runner.run_from_project") as mocked_run,
                patch("HQPINN.lib.runner._sync_selected_results") as mocked_sync,
            ):
                result = train_and_evaluate(
                    {
                        "experiment": "dho-cc",
                        "mode": "train",
                        "backend": "local",
                        "model": {"n_layers": 2, "n_nodes": 16},
                    },
                    run_dir,
                )

            self.assertTrue(run_dir.is_dir())
            mocked_run.assert_called_once()
            mocked_sync.assert_called_once()

            resolved_config = mocked_run.call_args.args[0]
            self.assertEqual(resolved_config["dtype"], "float64")
            self.assertEqual(resolved_config["experiment"], "dho-cc")
            self.assertEqual(resolved_config["model"]["n_layers"], 2)
            self.assertEqual(resolved_config["model"]["n_nodes"], 16)
            self.assertEqual(
                resolved_config["shared_runner"]["run_dir"],
                str(run_dir.resolve()),
            )

            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["run_dir"], str(run_dir.resolve()))
