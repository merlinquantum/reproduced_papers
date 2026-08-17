from __future__ import annotations

# Matplotlib's backend must be selected before importing pyplot.
# ruff: noqa: I001

import json
import logging
from pathlib import Path

import gymnasium as gym
import matplotlib
import numpy as np

from lib.util import (
    MinigridImageOnlyWrapper,
    create_hybrid_model,
    set_global_seed,
    train_environment,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _resolve_environment(env_name: str) -> str:
    """Resolve a Figure 1 environment alias to its Gymnasium ID.

    Parameters
    ----------
    env_name : str
        Environment alias or Gymnasium environment ID.

    Returns
    -------
    str
        Gymnasium environment ID.

    Raises
    ------
    ValueError
        If the environment is not supported by the Figure 1 experiment.
    """
    if env_name == "CartPole":
        return "CartPole-v1"
    if env_name in {"MiniGrid", "MiniGrid-Empty-5x5-v0"}:
        import minigrid  # noqa: F401  # Registers MiniGrid environments.

        return "MiniGrid-Empty-5x5-v0"
    raise ValueError(f"Unknown environment: {env_name}")


def _get_environment_dimensions(gym_env_id: str) -> tuple[int, int]:
    """Return the observation and action dimensions for an environment.

    Parameters
    ----------
    gym_env_id : str
        Gymnasium environment ID.

    Returns
    -------
    tuple[int, int]
        Observation dimension followed by action dimension.
    """
    if gym_env_id == "MiniGrid-Empty-5x5-v0":
        environment = MinigridImageOnlyWrapper(gym.make(gym_env_id))
    else:
        environment = gym.make(gym_env_id)

    observation_dim = environment.observation_space.shape[0]
    action_dim = environment.action_space.n
    environment.close()
    return observation_dim, action_dim


def _plot_results(
    experiment_results: list[dict[str, object]], output_path: Path
) -> None:
    """Save the Figure 1-style reward comparison.

    Parameters
    ----------
    experiment_results : list[dict[str, object]]
        Results containing environment, label, and reward histories.
    output_path : pathlib.Path
        Destination path for the PNG plot.
    """
    figure, axes = plt.subplots(1, 2, figsize=(14, 5))

    for axis, environment_name in zip(axes, ["CartPole", "MiniGrid"]):
        environment_results = [
            result
            for result in experiment_results
            if result["environment"] == environment_name
        ]
        for result in environment_results:
            reward_histories = np.asarray(result["reward_histories"], dtype=float)
            episode_numbers = np.arange(1, reward_histories.shape[1] + 1)
            mean_rewards = reward_histories.mean(axis=0)
            standard_deviations = reward_histories.std(axis=0)
            line = axis.plot(episode_numbers, mean_rewards, label=result["label"])[0]
            axis.fill_between(
                episode_numbers,
                mean_rewards - standard_deviations,
                mean_rewards + standard_deviations,
                color=line.get_color(),
                alpha=0.15,
            )

        axis.set_title(
            "CartPole-v1" if environment_name == "CartPole" else "MiniGrid-Empty-5x5-v0"
        )
        axis.set_xlabel("Episode")
        axis.set_ylabel("Total Reward")
        axis.grid(True, linestyle="--", alpha=0.35)
        axis.legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def run_figure_1(cfg: dict, run_dir: Path, logger: logging.Logger) -> None:
    """Run a Figure 1-style backend and hyperparameter comparison.

    Parameters
    ----------
    cfg : dict
        Resolved Figure 1 experiment configuration.
    run_dir : pathlib.Path
        Directory where plots and summaries are saved.
    logger : logging.Logger
        Logger used for experiment progress.
    """
    figure_config = cfg["figure_1"]
    backend = figure_config.get("backend", "torchquantum")
    default_output_stem = (
        "figure_1-tq" if backend == "torchquantum" else f"figure_1-{backend}"
    )
    output_stem = figure_config.get("output_stem", default_output_stem)
    repeats = int(figure_config.get("repeats", 3))
    base_seed = int(cfg.get("seed", 42))
    resume_run_dir_value = figure_config.get(
        "resume_run_dir", cfg.get("resume_run_dir")
    )
    resume_run_dir = (
        Path(resume_run_dir_value) if resume_run_dir_value is not None else None
    )
    checkpoint_root = (
        resume_run_dir / "checkpoints" if resume_run_dir else run_dir / "checkpoints"
    )
    progress_path = (
        resume_run_dir / f"{output_stem}_progress.json"
        if resume_run_dir
        else run_dir / f"{output_stem}_progress.json"
    )
    common_config = dict(cfg)
    common_config.update(figure_config.get("model", {}))
    if progress_path.exists():
        experiment_results = json.loads(progress_path.read_text(encoding="utf-8"))
    else:
        experiment_results = []

    def save_progress() -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        progress_path.write_text(json.dumps(experiment_results), encoding="utf-8")
        _plot_results(experiment_results, run_dir / f"{output_stem}.png")

    for environment_config in figure_config["environments"]:
        environment_name = environment_config["env_name"]
        gym_env_id = _resolve_environment(environment_name)
        observation_dim, action_dim = _get_environment_dimensions(gym_env_id)
        total_weights_needed = observation_dim * action_dim
        num_episodes = int(environment_config["num_episodes"])

        model_specs = [("Classical", "classic", {"variant_id": "baseline"})]
        if backend == "torchquantum":
            model_specs.extend(
                (
                    f"QTRL, L={depth}",
                    backend,
                    {"q_depth": depth, "variant_id": f"L{depth}"},
                )
                for depth in environment_config["depths"]
            )
        else:
            model_specs.extend(
                (variant["label"], backend, variant)
                for variant in environment_config["variants"]
            )

        for label, model_backend, variant in model_specs:
            result = next(
                (
                    item
                    for item in experiment_results
                    if item["environment"] == environment_name
                    and item["label"] == label
                ),
                None,
            )
            if result is None:
                result = {
                    "environment": environment_name,
                    "label": label,
                    "reward_histories": [],
                }
                experiment_results.append(result)

            for repeat in range(repeats):
                if len(result["reward_histories"]) > repeat:
                    continue

                experiment_seed = base_seed + repeat
                experiment_config = dict(common_config)
                experiment_config.update(environment_config)
                experiment_config.update(variant)
                experiment_config["backend"] = model_backend
                experiment_config["seed"] = experiment_seed
                set_global_seed(experiment_seed)
                model = create_hybrid_model(experiment_config, total_weights_needed)
                checkpoint_path = checkpoint_root / (
                    f"{environment_name}_{model_backend}_"
                    f"{variant['variant_id']}_repeat{repeat}.pt"
                )
                reward_history = train_environment(
                    model,
                    num_episode=num_episodes,
                    learning_rate=float(experiment_config["lr"]),
                    seed=experiment_seed,
                    logit_clip=float(experiment_config.get("logit_clip", 20.0)),
                    checkpoint_path=checkpoint_path,
                    resume_from=checkpoint_path if checkpoint_path.exists() else None,
                    env_name=gym_env_id,
                )
                result["reward_histories"].append(reward_history)
                save_progress()

    run_dir.mkdir(parents=True, exist_ok=True)
    figure_path = run_dir / f"{output_stem}.png"
    _plot_results(experiment_results, figure_path)
    (run_dir / f"{output_stem}_results.json").write_text(
        json.dumps(experiment_results), encoding="utf-8"
    )
    logger.info("Saved Figure 1 plot to %s", figure_path)


__all__ = ["run_figure_1"]
