"""Runtime entry point for the power_of_QNN reproduction (Abbas et al., arXiv:2011.00027).

``experiment.kind`` selects one experiment:

- ``reference_check``: V-00, comparison with the authors' cached arrays.
- ``fisher_effective_dimension``: Fig. 2 / 3a / 8 / 9 style Fisher spectra and
  effective dimension for a list of models (E-01, E-02, E-04, E-05, E-08, A-01, A-03).
- ``sensitivity``: Fig. 8 effective dimension vs Monte Carlo sample count (E-05).
- ``topology_search``: fair classical baseline search B-1 / A-02.
- ``training``: Iris training loss and Fisher-Rao norm (E-03, E-03b, A-01, A-04, B-2).
- ``generalisation``: Fig. 6 effective dimension vs label randomization (E-06).
- ``fair_comparison``: Phase 05 lr-tuned training comparison (B-2 and extensions).

Artifacts in ``run_dir`` (public contract of this reproduction): ``run.log`` and
``config_snapshot.json`` (shared runtime), ``run_status.json``, ``metrics.json``,
``fisher_<model>.npz`` / ``sensitivity_<model>.npz`` / ``training_<model>.npz`` /
``comparison_<model>.npz`` / ``generalisation.npz``, and
``fig_*.png``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch

from runtime_lib.dtypes import dtype_torch

from . import plotting
from .data import dataset_record
from .experiment_logging import git_state, sha256, utc_now, write_json
from .fair_comparison import run_fair_comparison
from .fisher_experiments import (
    run_fisher_effective_dimension,
    run_sensitivity,
    run_topology_search,
)
from .reference import run_reference_check
from .training_experiments import run_generalisation, run_training

logger = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENTS = (
    "reference_check",
    "fisher_effective_dimension",
    "sensitivity",
    "topology_search",
    "training",
    "generalisation",
    "fair_comparison",
)
SYNTHETIC_DATASET = {
    "name": "synthetic: theta ~ U[theta_low, theta_high]^d, x ~ N(0, I) (RandomState per draw seed)",
    "splits": ["fisher_draws"],
    "preprocessing": "none",
    "subset": None,
}


def _device(cfg: dict) -> torch.device:
    """Return the configured device; the Fisher numerics require float64."""
    if dtype_torch(cfg["dtype"]) != torch.float64:
        raise ValueError(
            f"power_of_QNN requires dtype float64 for the Fisher numerics; got {cfg['dtype']}"
        )
    device = torch.device(cfg["device"])
    if device.type == "mps":
        raise ValueError("MPS has no float64 support; use cpu or cuda")
    return device


def _selection(kind: str, cfg: dict) -> dict:
    if kind == "fair_comparison":
        return {
            "name": "per-model lr with the lowest mean final cross-entropy over the selection seeds (predeclared, Phase 05)",
            "direction": "minimize",
            "split": cfg["comparison"]["selection_split"],
            "checkpoint": "final",
        }
    if kind == "training":
        return {
            "name": "final training cross-entropy (Table 1, no model selection)",
            "direction": "minimize",
            "split": "train",
            "checkpoint": "final",
        }
    if kind == "topology_search":
        return {
            "name": "mean F_hat rank, then normalized effective dimension (predeclared)",
            "direction": "maximize",
            "split": "fisher_draws",
            "checkpoint": "n/a",
        }
    if kind == "reference_check":
        return {
            "name": "max abs difference to the authors' arrays (reported, no selection)",
            "direction": "minimize",
            "split": "fisher_draws",
            "checkpoint": "n/a",
        }
    if kind == "generalisation":
        return {
            "name": "normalized effective dimension at the trained theta (reported, no selection)",
            "direction": "maximize",
            "split": "train",
            "checkpoint": "final",
        }
    # Fisher and sensitivity runs report the effective dimension, which the paper
    # reads as "higher is more capacity"; nothing is selected from it.
    return {
        "name": "normalized effective dimension (reported, no selection)",
        "direction": "maximize",
        "split": "fisher_draws",
        "checkpoint": "n/a",
    }


def _figures(kind: str, cfg: dict, metrics: dict, run_dir: Path) -> list[str]:
    written = []
    if kind == "fisher_effective_dimension":
        plotting.plot_spectra(
            run_dir,
            metrics,
            cfg["plots"]["histogram_bins"],
            run_dir / "fig_spectra.png",
        )
        plotting.plot_effective_dimension(
            metrics, run_dir / "fig_effective_dimension.png"
        )
        written = ["fig_spectra.png", "fig_effective_dimension.png"]
    elif kind == "sensitivity":
        plotting.plot_sensitivity(
            metrics,
            cfg["plots"]["sensitivity_n_index"],
            run_dir / "fig_sensitivity.png",
        )
        written = ["fig_sensitivity.png"]
    elif kind in ("training", "fair_comparison"):
        plotting.plot_training_losses(metrics, run_dir / "fig_training_loss.png")
        written = ["fig_training_loss.png"]
    elif kind == "generalisation":
        plotting.plot_generalisation(metrics, run_dir / "fig_generalisation.png")
        written = ["fig_generalisation.png"]
    for name in written:
        logger.info("ARTIFACT_WRITTEN | path=%s", name)
    return written


def train_and_evaluate(cfg: dict, run_dir: Path) -> None:
    """Run the configured experiment and write its artifacts into ``run_dir``.

    Parameters
    ----------
    cfg : dict
        Resolved configuration from the shared runtime. Required keys: ``experiment``
        (``name``, ``kind``), ``seed``, ``dtype``, ``device``, ``data_root``, and the
        sections of the selected kind.
    run_dir : pathlib.Path
        Timestamped output directory created by the shared runtime.

    Raises
    ------
    KeyError
        If a required config key is missing.
    ValueError
        For an unknown experiment kind or an unsupported dtype or device.
    """
    run_dir = Path(run_dir)
    kind = cfg["experiment"]["kind"]
    if kind not in EXPERIMENTS:
        raise ValueError(
            f"Unknown experiment kind {kind!r}; expected one of {EXPERIMENTS}"
        )
    uses_real_data = kind in ("training", "generalisation", "fair_comparison")
    config_path = run_dir / "config_snapshot.json"
    status = {
        "schema_version": 1,
        "run_id": run_dir.name,
        "sweep_id": cfg.get("sweep_id"),
        "candidate": cfg.get("candidate", {}),
        "seed": int(cfg["seed"]),
        "repetition": cfg.get("repetition", 0),
        "status": "RUNNING",
        "started_at": utc_now(),
        "completed_at": None,
        "code": git_state(REPO_ROOT),
        "dataset": dataset_record(cfg) if uses_real_data else SYNTHETIC_DATASET,
        "selection": _selection(kind, cfg),
        "log_path": "run.log",
        "config_path": "config_snapshot.json",
        "config_sha256": sha256(config_path) if config_path.exists() else None,
        "metrics_path": "metrics.json",
        "metrics_sha256": None,
        "error": None,
    }
    write_json(run_dir / "run_status.json", status)
    logger.info(
        "RUN_STARTED | run_id=%s | experiment=%s | kind=%s | device=%s",
        run_dir.name,
        cfg["experiment"]["name"],
        kind,
        cfg["device"],
    )
    try:
        device = _device(cfg)
        if kind == "reference_check":
            metrics = run_reference_check(cfg, run_dir, device)
        elif kind in ("fisher_effective_dimension", "sensitivity", "topology_search"):
            logger.info(
                "DATASET_READY | name=synthetic fisher draws | draw_seeds=%s",
                cfg["fisher"]["draw_seeds"],
            )
            runners = {
                "fisher_effective_dimension": run_fisher_effective_dimension,
                "sensitivity": run_sensitivity,
                "topology_search": run_topology_search,
            }
            metrics = runners[kind](cfg, run_dir, device)
        elif kind == "training":
            metrics = run_training(cfg, run_dir, device)
        elif kind == "fair_comparison":
            metrics = run_fair_comparison(cfg, run_dir, device)
        else:
            metrics = run_generalisation(cfg, run_dir, device)
        metrics = {
            "experiment": cfg["experiment"],
            **metrics,
            "figures": _figures(kind, cfg, metrics, run_dir),
        }
        metrics_path = run_dir / "metrics.json"
        metrics_path.write_text(
            json.dumps(metrics, indent=2, default=float), encoding="utf-8"
        )
        status["metrics_sha256"] = sha256(metrics_path)
        logger.info("METRICS_WRITTEN | path=metrics.json")
        status.update(status="COMPLETED", completed_at=utc_now())
        write_json(run_dir / "run_status.json", status)
        logger.info("RUN_COMPLETED | run_id=%s", run_dir.name)
    except Exception as exc:
        status.update(
            status="FAILED",
            completed_at=utc_now(),
            error={"type": type(exc).__name__, "message": str(exc)},
        )
        write_json(run_dir / "run_status.json", status)
        logger.exception("RUN_FAILED | run_id=%s", run_dir.name)
        raise
