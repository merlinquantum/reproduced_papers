from __future__ import annotations

import logging
from pathlib import Path

import gymnasium as gym
import torch
from lib.util import (
    MinigridImageOnlyWrapper,
    create_hybrid_model,
    set_global_seed,
    train_environment,
)


def train_and_evaluate(cfg: dict, run_dir: Path) -> None:
    """
    Main function called to train the hybrid model.

    Args:
        cfg (dict): Configuration dictionary passed by the launcher.
        run_dir (Path): Path object specifying the directory where results and model
                        will be saved.
    """
    logger = logging.getLogger(__name__)

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
    if env_name == "CartPole":
        gym_env_id = "CartPole-v1"
    elif env_name == "MiniGrid":
        gym_env_id = "MiniGrid-Empty-5x5-v0"
    else:
        raise ValueError(f"Unknown environment: {env_name}")

    # Instantiate a temporary environment to compute dimensions
    if gym_env_id == "MiniGrid-Empty-5x5-v0":
        base_env = gym.make(gym_env_id)
        temp_env = MinigridImageOnlyWrapper(base_env)
    else:
        temp_env = gym.make(gym_env_id)

    # Get state and action dimensions
    state_dim = temp_env.observation_space.shape[0]
    action_dim = temp_env.action_space.n
    total_weights_needed = state_dim * action_dim

    # Close the temporary environment
    temp_env.close()

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
    train_environment(
        model,
        num_episode=num_episodes,
        learning_rate=lr,
        seed=seed,
        env_name=gym_env_id,
    )

    # Save the results
    run_dir.mkdir(parents=True, exist_ok=True)

    model_path = run_dir / f"{backend}_model.pt"
    torch.save(model.state_dict(), model_path)
    logger.info("Saved model checkpoint to %s", model_path)

    done_marker = run_dir / "done.txt"
    done_marker.write_text("ok", encoding="utf-8")
    logger.info("Saved completion marker to %s", done_marker)


__all__ = ["train_and_evaluate"]
