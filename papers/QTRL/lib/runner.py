from __future__ import annotations

# Matplotlib's backend must be selected before importing pyplot.
# ruff: noqa: I001

import logging
from pathlib import Path

import gymnasium as gym
import matplotlib
import torch

from lib.util import (
    MinigridImageOnlyWrapper,
    create_hybrid_model,
    set_global_seed,
    train_environment,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def save_reward_plot(episode_rewards: list[float], output_path: Path) -> None:
    """Save total episode reward against the episode number.

    Parameters
    ----------
    episode_rewards : list[float]
        Total reward collected during each training episode.
    output_path : pathlib.Path
        Destination path for the PNG plot.
    """
    episode_numbers = range(1, len(episode_rewards) + 1)
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(episode_numbers, episode_rewards, linewidth=1.2)
    axis.set_xlabel("Episode")
    axis.set_ylabel("Total reward")
    axis.set_title("QTRL total reward by episode")
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def _resolve_environment(env_name: str) -> str:
    """Resolve a configured environment alias to its Gymnasium ID.

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
        If the environment is not supported by QTRL.
    """
    if env_name == "CartPole":
        return "CartPole-v1"
    if env_name in {"MiniGrid", "MiniGrid-Empty-5x5-v0"}:
        import minigrid  # noqa: F401  # Registers MiniGrid environments.

        return "MiniGrid-Empty-5x5-v0"
    raise ValueError(f"Unknown environment: {env_name}")


def _get_environment_dimensions(gym_env_id: str) -> tuple[int, int]:
    """Return the flattened observation and action dimensions for an environment.

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
        base_env = gym.make(gym_env_id)
        environment = MinigridImageOnlyWrapper(base_env)
    else:
        environment = gym.make(gym_env_id)

    observation_dim = environment.observation_space.shape[0]
    action_dim = environment.action_space.n
    environment.close()
    return observation_dim, action_dim


def train_and_evaluate(cfg: dict, run_dir: Path) -> None:
    """
    Main function called to train the hybrid model.

    Args:
        cfg (dict): Configuration dictionary passed by the launcher.
        run_dir (Path): Path object specifying the directory where results and model
                        will be saved.
    """
    logger = logging.getLogger(__name__)

    if cfg.get("experiment") == "figure_1":
        from lib.figure_1 import run_figure_1

        run_figure_1(cfg, run_dir, logger)
        return

    # Extract parameters from configuration
    env_name = cfg.get("env_name", "CartPole")
    backend = cfg.get("backend", "merlin_mlp")
    num_episodes = int(cfg.get("num_episodes", 1000))
    lr = float(cfg.get("lr", 0.001))
    seed = int(cfg.get("seed", 42))

    logger.info("==================================================")
    logger.info("Initializing experiment via Launcher on environment: %s", env_name)
    logger.info("Backend used: %s", backend)
    logger.info("Saving directory: %s", run_dir)
    logger.info("==================================================")

    # Fix the random seed for reproducibility
    set_global_seed(seed=seed)

    # Determine the correct Gym environment ID
    gym_env_id = _resolve_environment(env_name)

    # Instantiate a temporary environment to compute dimensions
    state_dim, action_dim = _get_environment_dimensions(gym_env_id)
    total_weights_needed = state_dim * action_dim

    logger.info(
        f"Detected dimensions for {env_name} : State = {state_dim}, Actions = {action_dim}"
    )
    logger.info(f"Total weights required: {total_weights_needed}")

    # Create the hybrid model
    model = create_hybrid_model(cfg, total_weights_needed)
    device = torch.device(cfg.get("device", "cpu"))
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Number of trainable parameters: %d", total_params)

    # Train the model
    episode_rewards = train_environment(
        model,
        num_episode=num_episodes,
        learning_rate=lr,
        seed=seed,
        env_name=gym_env_id,
    )

    # Save the results
    run_dir.mkdir(parents=True, exist_ok=True)
    reward_plot_path = run_dir / "total_reward_vs_episode.png"
    save_reward_plot(episode_rewards, reward_plot_path)
    logger.info("Saved reward plot to %s", reward_plot_path)

    model_path = run_dir / f"{backend}_model.pt"
    torch.save(model.state_dict(), model_path)
    logger.info("Saved model checkpoint to %s", model_path)

    done_marker = run_dir / "done.txt"
    done_marker.write_text("ok", encoding="utf-8")
    logger.info("Saved completion marker to %s", done_marker)


__all__ = ["train_and_evaluate"]
