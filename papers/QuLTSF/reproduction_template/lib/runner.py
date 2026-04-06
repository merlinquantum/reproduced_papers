from __future__ import annotations

import logging
from pathlib import Path


def train_and_evaluate(cfg, run_dir: Path) -> None:
    logger = logging.getLogger(__name__)
    artifact = run_dir / "done.txt"
    artifact.write_text(
        "Training placeholder complete. Replace with actual implementation.\n",
        encoding="utf-8",
    )
    logger.info("Wrote placeholder artifact: %s", artifact)
