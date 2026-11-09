"""Shared-runtime entry point for the barren-plateau experiments."""

from pathlib import Path

from .experiments import run_experiment


def train_and_evaluate(cfg: dict, run_dir: Path) -> None:
    """Run the configured gradient-variance experiment.

    Parameters
    ----------
    cfg : dict
        Resolved experiment configuration.
    run_dir : pathlib.Path
        Timestamped output directory created by the shared runtime.

    Returns
    -------
    None
        Artifacts are written below ``run_dir``.
    """
    run_experiment(cfg, run_dir)
