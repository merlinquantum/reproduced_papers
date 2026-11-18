"""Runtime entry point: ``train_and_evaluate(cfg, run_dir)``.

One run trains every agent listed in ``agents.seeds`` (the paper's "10
agents" per configuration) in a single process, evaluates the final greedy
policies, and writes the evidence files below. Paths in ``run_status.json``
are relative to ``run_dir`` so the folder can be copied back from a remote
machine.

Artifacts (public reproduction contract):

* ``run.log``, ``run_status.json``, ``config_snapshot.json`` - logging contract.
* ``metrics.json`` - per-agent and aggregate results (solve episode index,
  scores summary, greedy evaluation, Q* error for Frozen Lake).
* ``scores.npz`` - raw per-agent episode scores (and Q* MAE per time step).
* ``score_curve.png`` (and ``q_star_mae.png``) - paper-style figures.
* ``final_parameters.pt`` - online-network ``state_dict`` of all agents.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from runtime_lib.dtypes import dtype_torch

from .config_access import require
from .dqn import DQNTrainer
from .experiment_logging import (
    configure_logging,
    git_state,
    sha256,
    utc_now,
    write_json,
)
from .metrics import normalised_area_under_curve, solve_summary
from .plotting import plot_q_star_mae, plot_score_curve

logger = logging.getLogger(__name__)

_PAPER_DIR = Path(__file__).resolve().parents[1]
_LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
}
_MAX_SCORE = {"CartPole-v0": 200.0, "FrozenLake-v1": 1.0}


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    label = getattr(value, "label", None)
    return label if label is not None else str(value)


def _resolve_real_dtype(cfg: dict[str, Any]) -> torch.dtype:
    if "dtype" not in cfg or cfg["dtype"] is None:
        raise KeyError("Missing required config key 'dtype' (float32 or float64)")
    real_dtype = dtype_torch(cfg["dtype"])
    if real_dtype not in (torch.float32, torch.float64):
        raise ValueError(f"Unsupported dtype {cfg['dtype']!r}: use float32 or float64")
    return real_dtype


def _write_scores(
    path: Path, scores: list[list[float]], q_star_mae: dict | None
) -> None:
    arrays: dict[str, np.ndarray] = {}
    lengths = np.array([len(agent_scores) for agent_scores in scores])
    padded = np.full((len(scores), max(lengths.max(initial=0), 1)), np.nan)
    for row, agent_scores in enumerate(scores):
        padded[row, : len(agent_scores)] = agent_scores
    arrays["scores"] = padded
    arrays["episodes_run"] = lengths
    if q_star_mae is not None:
        for name, series in q_star_mae.items():
            steps = max(len(values) for values in series)
            matrix = np.full((len(series), steps), np.nan)
            for row, values in enumerate(series):
                matrix[row, : len(values)] = values
            arrays[f"q_star_mae_{name}"] = matrix
    np.savez_compressed(path, **arrays)


def train_and_evaluate(cfg: dict[str, Any], run_dir: Path) -> None:
    """Train and evaluate all agents of one configuration.

    Parameters
    ----------
    cfg : dict
        Resolved configuration (defaults, overlay config and CLI flags).
    run_dir : pathlib.Path
        Timestamped run directory created by the shared runtime.

    Raises
    ------
    KeyError
        If a required config key is missing.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    level_name = str(require(cfg, "logging.level")).lower()
    configure_logging(run_dir / "run.log", level=_LOG_LEVELS[level_name])

    config_path = run_dir / "config_snapshot.json"
    if not config_path.exists():
        config_path.write_text(
            json.dumps(cfg, indent=2, default=_json_default), encoding="utf-8"
        )

    seeds = [int(seed) for seed in require(cfg, "agents.seeds")]
    experiment = require(cfg, "experiment")
    environment_name = require(cfg, "environment.name")
    status: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_dir.name,
        # Set by utils/run_batch.py when the run belongs to a batch; absent for
        # standalone runs.
        "sweep_id": os.environ.get("ALPS_SWEEP_ID"),
        "candidate": {
            "experiment": experiment,
            "model_kind": require(cfg, "model.kind"),
        },
        "seed": seeds,
        "repetition": None,
        "status": "RUNNING",
        "started_at": utc_now(),
        "completed_at": None,
        "code": git_state(_PAPER_DIR),
        "dataset": {
            "name": environment_name,
            "splits": ["online interaction (train)", "greedy evaluation episodes"],
            "preprocessing": "Cart Pole: raw observations; Frozen Lake: state index -> basis state / 4 bits",
            "subset": "none (environment simulator, Gymnasium 1.4.0)",
        },
        "selection": {
            "name": "solved_at_episode_index",
            "direction": "minimize",
            "split": "online interaction (train)",
            "checkpoint": "final",
        },
        "log_path": "run.log",
        "config_path": "config_snapshot.json",
        "config_sha256": sha256(config_path),
        "metrics_path": None,
        "metrics_sha256": None,
        "error": None,
    }
    write_json(run_dir / "run_status.json", status)
    logger.info(
        "RUN_STARTED | run_id=%s | experiment=%s | arm=%s | label=%s | seeds=%s",
        run_dir.name,
        experiment.get("id"),
        experiment.get("arm"),
        experiment.get("label"),
        seeds,
    )

    try:
        device = torch.device(require(cfg, "device"))
        real_dtype = _resolve_real_dtype(cfg)
        greedy_episodes = int(require(cfg, "evaluation.greedy_episodes"))
        if greedy_episodes < 1:
            raise ValueError("evaluation.greedy_episodes must be >= 1")

        trainer = DQNTrainer(cfg, device, real_dtype)
        logger.info(
            "DATASET_READY | environment=%s | max_episode_steps=%s | is_slippery=%s | agents=%d",
            environment_name,
            require(cfg, "environment.max_episode_steps"),
            require(cfg, "environment.is_slippery"),
            len(seeds),
        )
        result = trainer.train()
        max_episodes = int(require(cfg, "training.max_episodes"))
        summary = solve_summary(result.solved_at_episode_index, max_episodes)
        logger.info(
            "TRAINING_COMPLETED | solved=%d/%d | median_solved_episode_index=%s | wall_clock_s=%.1f | lockstep_iterations=%d",
            summary["n_solved"],
            summary["n_agents"],
            summary["median_solved_episode_index"],
            result.wall_clock_seconds,
            result.lockstep_iterations,
        )

        logger.info(
            "EVALUATION_STARTED | split=greedy_policy | episodes_per_agent=%d",
            greedy_episodes,
        )
        greedy_scores = trainer.evaluate_greedy(greedy_episodes)
        greedy_means = [float(np.mean(values)) for values in greedy_scores]
        logger.info(
            "EVALUATION_COMPLETED | split=greedy_policy | mean_score_over_agents=%.3f | per_agent=%s",
            float(np.mean(greedy_means)),
            [round(value, 3) for value in greedy_means],
        )

        max_score = _MAX_SCORE[environment_name]
        _write_scores(run_dir / "scores.npz", result.scores, result.q_star_mae)
        plot_score_curve(
            result.scores,
            max_score,
            f"{experiment.get('id')} {experiment.get('arm')} ({len(seeds)} agents, {experiment.get('label')})",
            run_dir / "score_curve.png",
        )
        artifacts = {
            "scores": "scores.npz",
            "score_curve": "score_curve.png",
            "parameters": "final_parameters.pt",
        }
        if result.q_star_mae is not None:
            # Fig. 3b uses the 44 non-terminal pairs: with U[-0.05, 0.05] init this gives the
            # paper's starting MAE of 0.54 (64 pairs would give 0.47). See LOG.md.
            plot_q_star_mae(
                result.q_star_mae["nonterminal_44"],
                f"{experiment.get('arm')}: MAE to Q* over 44 non-terminal (s, a) pairs",
                run_dir / "q_star_mae.png",
            )
            artifacts["q_star_mae"] = "q_star_mae.png"
        torch.save(trainer.online.state_dict(), run_dir / "final_parameters.pt")
        logger.info("ARTIFACTS_WRITTEN | %s", artifacts)

        agents = []
        for index, seed in enumerate(result.seeds):
            agent_record = {
                "seed": seed,
                "solved": result.solved_at_episode_index[index] is not None,
                "solved_at_episode_index": result.solved_at_episode_index[index],
                "episodes_run": len(result.scores[index]),
                "total_env_steps": result.total_env_steps[index],
                "final_epsilon": result.final_epsilon[index],
                "mean_last_100_scores": float(np.mean(result.scores[index][-100:]))
                if result.scores[index]
                else None,
                "greedy_eval_scores": greedy_scores[index],
                "greedy_eval_mean": greedy_means[index],
            }
            if result.q_star_mae is not None:
                agent_record["final_q_star_mae_all_64"] = result.q_star_mae["all_64"][
                    index
                ][-1]
                agent_record["final_q_star_mae_nonterminal_44"] = result.q_star_mae[
                    "nonterminal_44"
                ][index][-1]
            agents.append(agent_record)

        metrics = {
            "experiment": experiment,
            "environment": environment_name,
            "model_kind": require(cfg, "model.kind"),
            "trainable_parameters_per_agent": result.trainable_parameters,
            "max_episodes": max_episodes,
            "solve_rule": require(cfg, "environment.solve"),
            "solve_summary": summary,
            "normalised_auc_first_1000_episodes": normalised_area_under_curve(
                result.scores, 1000, max_score
            ),
            "greedy_eval_mean_over_agents": float(np.mean(greedy_means)),
            "agents": agents,
            "wall_clock_seconds": result.wall_clock_seconds,
            "lockstep_iterations": result.lockstep_iterations,
            "seconds_per_lockstep_iteration": result.wall_clock_seconds
            / max(result.lockstep_iterations, 1),
            "device": str(device),
            "dtype": str(real_dtype),
            "artifacts": artifacts,
        }
        metrics_path = run_dir / "metrics.json"
        write_json(metrics_path, metrics)
        logger.info("METRICS_WRITTEN | path=metrics.json")

        status.update(
            status="COMPLETED",
            completed_at=utc_now(),
            metrics_path="metrics.json",
            metrics_sha256=sha256(metrics_path),
        )
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
