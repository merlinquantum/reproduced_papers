from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from lib.utils import (
    finalize_training_session,
    get_resume_checkpoint_path,
    prepare_training_session,
    save_training_checkpoint,
)


class TrainingCheckpointTests(unittest.TestCase):
    def test_prepare_and_finalize_training_session_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_dir = Path(tmp_dir)
            case_prefix = "demo_case"
            resume_path = get_resume_checkpoint_path(str(ckpt_dir), case_prefix)

            model = nn.Linear(2, 1)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            with torch.no_grad():
                model.weight.fill_(1.25)
                model.bias.fill_(-0.75)

            save_training_checkpoint(
                resume_path,
                model=model,
                optimizer=optimizer,
                run_id="20260410-101010",
                epoch=7,
                elapsed_s=12.5,
                rows=[[0, "0.10", "1.0", "0.1", "0.2", "0.3"]],
                extra_state={"phase": "adam"},
            )

            resumed_model = nn.Linear(2, 1)
            resumed_optimizer = torch.optim.Adam(resumed_model.parameters(), lr=1e-3)
            run_id, state, returned_resume_path = prepare_training_session(
                model=resumed_model,
                optimizer=resumed_optimizer,
                ckpt_dir=str(ckpt_dir),
                case_prefix=case_prefix,
                default_run_id="fresh-run",
            )

            self.assertEqual(run_id, "20260410-101010")
            self.assertEqual(returned_resume_path, resume_path)
            self.assertIsNotNone(state)
            assert state is not None
            self.assertEqual(state["epoch"], 7)
            self.assertEqual(state["elapsed_s"], 12.5)
            self.assertEqual(state["extra_state"]["phase"], "adam")
            self.assertTrue(
                torch.allclose(resumed_model.weight, model.weight),
                "Expected resumed model weights to match the saved checkpoint",
            )
            self.assertTrue(
                torch.allclose(resumed_model.bias, model.bias),
                "Expected resumed model bias to match the saved checkpoint",
            )

            final_ckpt_path = finalize_training_session(
                model=resumed_model,
                ckpt_dir=str(ckpt_dir),
                case_prefix=case_prefix,
                run_id=run_id,
                resume_checkpoint_path=resume_path,
            )

            self.assertTrue(Path(final_ckpt_path).is_file())
            self.assertFalse(Path(resume_path).exists())

            final_state = torch.load(final_ckpt_path, map_location="cpu")
            self.assertIn("weight", final_state)
            self.assertIn("bias", final_state)

