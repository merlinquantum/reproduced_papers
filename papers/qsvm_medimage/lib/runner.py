"""Minimal catalogue runner for the MerLin photonic adaptation."""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def train_and_evaluate(cfg: dict[str, Any], run_dir: Path) -> None:
    """Run the existing MerLin CPU entry point through the shared runtime.

    Parameters
    ----------
    cfg : dict[str, Any]
        Resolved catalogue configuration.
    run_dir : pathlib.Path
        Timestamped output directory created by the shared runtime.
    """
    if cfg.get("scope") != "merlin_photonic_adaptation":
        raise ValueError("scope must be 'merlin_photonic_adaptation'")
    if str(cfg.get("device")) != "cpu":
        raise ValueError("The qsvm_medimage catalogue runner supports CPU only")

    dtype_value = cfg.get("dtype")
    dtype_label = getattr(dtype_value, "label", dtype_value)
    if dtype_label != "float32":
        raise ValueError("The MerLin adaptation currently supports dtype=float32 only")

    dataset_cfg = cfg.get("dataset")
    experiment_cfg = cfg.get("experiment")
    if not isinstance(dataset_cfg, dict) or "filename" not in dataset_cfg:
        raise ValueError("dataset.filename is required")
    if not isinstance(experiment_cfg, dict):
        raise ValueError("experiment configuration is required")

    required_experiment_keys = {"pca_dim", "circuit_seed", "max_samples"}
    missing_keys = required_experiment_keys - experiment_cfg.keys()
    if missing_keys:
        raise ValueError(
            "Missing experiment configuration: " + ", ".join(sorted(missing_keys))
        )

    pca_dim = int(experiment_cfg["pca_dim"])
    max_samples = int(experiment_cfg["max_samples"])
    fix_leakage = experiment_cfg.get("fix_leakage", False)
    if not isinstance(fix_leakage, bool):
        raise ValueError("experiment.fix_leakage must be a boolean")
    if not 1 <= pca_dim <= 19:
        raise ValueError("experiment.pca_dim must be between 1 and 19")
    if max_samples <= 0:
        raise ValueError("experiment.max_samples must be positive")

    data_path = (
        Path(cfg["data_root"]) / "qsvm_medimage" / str(dataset_cfg["filename"])
    ).resolve()
    if not data_path.is_file() and dataset_cfg.get("download_if_missing") is True:
        from scripts.prepare_pneumoniamnist import main as prepare_data

        prepare_data(
            [
                "--download_path",
                str(data_path.with_suffix(".npz")),
                "--output_path",
                str(data_path),
            ]
        )
    if not data_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}. Prepare it with "
            "scripts/prepare_pneumoniamnist.py as documented in README.md."
        )

    cache_root = Path(tempfile.gettempdir())
    os.environ.setdefault("XDG_DATA_HOME", str(cache_root / "qsvm-merlin-data"))
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "qsvm-merlin-mpl"))

    from scripts.merlin_fidelity_kernel import main as merlin_main

    data_argument = os.path.relpath(data_path, PROJECT_ROOT)
    logging.getLogger(__name__).info(
        "Running MerLin adaptation with pca_dim=%d, max_samples=%d",
        pca_dim,
        max_samples,
    )
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)
    arguments = [
        "--data_path",
        data_argument,
        "--output_dir",
        str(run_dir.resolve()),
        "--pca_dim",
        str(pca_dim),
        "--seed",
        str(int(cfg["seed"])),
        "--circuit_seed",
        str(int(experiment_cfg["circuit_seed"])),
        "--max_samples",
        str(max_samples),
    ]
    if fix_leakage:
        arguments.append("--fix_leakage")
    merlin_main(arguments)
