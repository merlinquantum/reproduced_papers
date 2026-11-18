"""DQN training (paper Sec. 2.2, 3) for several independent agents at once.

All agents of one configuration (one per seed) step their own environments
in lockstep and share batched model evaluations; every agent keeps its own
environment, replay memory, epsilon, counters, parameters, target
parameters and optimiser state. Per agent, the control flow reproduces the
reference code (``QLearningCartpole.perform_episodes``):

.. code-block:: text

    for episode in range(max_episodes):            # stops once solved
        for step in range(max_steps):              # step = 0, 1, ...
            a = epsilon-greedy(Q(s))
            s', r, done = env.step(a)
            replay.add(s, a, r, s', done)
            if done: record score; break           # no update on the last step
            if len(replay) >= batch and step % update_model_every == 0: update
            if step % update_target_every == 0: target <- online
        if solved(scores): stop
        epsilon <- max(epsilon_min, epsilon * decay)   # once per episode

Update (eqs. 8, 9): ``y = r + gamma * max_a' Qhat(s', a') * (1 - done)``,
mean squared error on the taken action (``loss_form="action_only"``,
reference Cart Pole code) or on the full Q-vector whose other entries are
copied from the prediction (``"vector_form"``, eq. 9 and the reference Frozen
Lake code; equals the action-only loss divided by the number of actions).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from .config_access import require, require_choice
from .data import build_replay_memory, encode_observation, make_environment, n_actions
from .models import build_q_function
from .optim import MaskedAdam
from .reference import optimal_q_values, terminal_states

logger = logging.getLogger(__name__)


@dataclass
class AgentState:
    """Mutable bookkeeping of one agent."""

    index: int
    seed: int
    rng: np.random.Generator
    env: Any
    memory: Any
    epsilon: float
    observation: np.ndarray
    episode: int = 0
    step_in_episode: int = 0
    episode_return: float = 0.0
    total_env_steps: int = 0
    scores: list[float] = field(default_factory=list)
    solved_at_episode_index: int | None = None
    active: bool = True


@dataclass
class TrainingResult:
    """Everything a run reports for its agents."""

    seeds: list[int]
    scores: list[list[float]]
    solved_at_episode_index: list[int | None]
    total_env_steps: list[int]
    final_epsilon: list[float]
    q_star_mae: dict[str, list[list[float]]] | None
    wall_clock_seconds: float
    lockstep_iterations: int
    trainable_parameters: int


class DQNTrainer:
    """Vectorised DQN over the agents listed in ``cfg["agents"]["seeds"]``.

    Parameters
    ----------
    cfg : dict
        Resolved configuration.
    device : torch.device
        Device holding the models.
    real_dtype : torch.dtype
        Real parameter precision.
    """

    def __init__(
        self, cfg: dict[str, Any], device: torch.device, real_dtype: torch.dtype
    ):
        self.cfg = cfg
        self.device = device
        self.real_dtype = real_dtype
        self.seeds = [int(seed) for seed in require(cfg, "agents.seeds")]
        if not self.seeds:
            raise ValueError("agents.seeds must list at least one seed")
        self.n_agents = len(self.seeds)
        self.n_actions = n_actions(cfg)
        self.environment_name = require(cfg, "environment.name")

        self.max_episodes = int(require(cfg, "training.max_episodes"))
        self.gamma = float(require(cfg, "training.gamma"))
        self.batch_size = int(require(cfg, "training.batch_size"))
        self.update_model_every = int(require(cfg, "training.update_model_every"))
        self.update_target_every = int(require(cfg, "training.update_target_every"))
        self.loss_form = require_choice(
            cfg, "training.loss_form", ("action_only", "vector_form")
        )
        self.bootstrap_network = require_choice(
            cfg, "training.bootstrap_network", ("target", "online")
        )
        self.terminal_mask = require_choice(
            cfg,
            "training.terminal_mask",
            ("terminated_or_truncated", "terminated_only"),
        )
        self.epsilon_start = float(require(cfg, "training.epsilon_greedy.start"))
        self.epsilon_decay = float(require(cfg, "training.epsilon_greedy.decay"))
        self.epsilon_min = float(require(cfg, "training.epsilon_greedy.min"))
        self.solve_window = int(require(cfg, "environment.solve.window"))
        self.solve_threshold = float(require(cfg, "environment.solve.mean_threshold"))
        self.solve_min_episodes = int(require(cfg, "environment.solve.min_episodes"))
        self.track_q_star_mae = bool(require(cfg, "evaluation.track_q_star_mae"))
        if self.track_q_star_mae and self.environment_name != "FrozenLake-v1":
            raise ValueError(
                "evaluation.track_q_star_mae is only defined for Frozen Lake"
            )

        self.online = build_q_function(cfg, self.seeds, self.n_actions, real_dtype).to(
            device
        )
        self.target = build_q_function(cfg, self.seeds, self.n_actions, real_dtype).to(
            device
        )
        self.target.load_state_dict(self.online.state_dict())
        self.target.requires_grad_(False)

        learning_rates = require(cfg, "training.learning_rates")
        amsgrad_main = bool(require(cfg, "training.amsgrad_main"))
        groups = []
        for group in self.online.parameter_groups():
            if group.learning_rate_key not in learning_rates:
                raise KeyError(
                    f"Missing training.learning_rates.{group.learning_rate_key}"
                )
            amsgrad = (
                group.amsgrad and amsgrad_main
                if group.learning_rate_key == "main"
                else group.amsgrad
            )
            groups.append(
                (group.tensors, learning_rates[group.learning_rate_key], amsgrad)
            )
        self.optimizer = MaskedAdam(
            groups,
            beta1=float(require(cfg, "training.adam.beta1")),
            beta2=float(require(cfg, "training.adam.beta2")),
            epsilon=float(require(cfg, "training.adam.epsilon")),
        )
        self.trainable_parameters = self.online.count_trainable_parameters()

        if self.track_q_star_mae:
            self.q_star = optimal_q_values(self.gamma)
            self.nonterminal_states = [
                state for state in range(16) if state not in terminal_states()
            ]

    # ------------------------------------------------------------------ helpers
    def _to_tensor(self, array: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(array, dtype=self.real_dtype, device=self.device)

    def _new_agents(self) -> list[AgentState]:
        agents = []
        for index, seed in enumerate(self.seeds):
            rng = np.random.default_rng(seed)
            env = make_environment(self.cfg)
            observation, _ = env.reset(seed=seed)
            agents.append(
                AgentState(
                    index=index,
                    seed=seed,
                    rng=rng,
                    env=env,
                    memory=build_replay_memory(self.cfg, rng),
                    epsilon=self.epsilon_start,
                    observation=encode_observation(self.cfg, observation),
                )
            )
        return agents

    def _greedy_actions(
        self, model: torch.nn.Module, observations: np.ndarray
    ) -> np.ndarray:
        """Arg-max actions ``(A,)`` for one observation per agent."""
        with torch.no_grad():
            q_values = model(self._to_tensor(observations[:, None, :]))[:, 0, :]
        return torch.argmax(q_values, dim=-1).cpu().numpy()

    def _is_solved(self, scores: list[float]) -> bool:
        if len(scores) < self.solve_min_episodes:
            return False
        return float(np.mean(scores[-self.solve_window :])) >= self.solve_threshold

    def _q_star_mae(self) -> tuple[np.ndarray, np.ndarray]:
        """Per-agent MAE to Q* over all 64 and over the 44 non-terminal (s, a) pairs."""
        states = np.tile(
            np.arange(16, dtype=np.float64)[None, :, None], (self.n_agents, 1, 1)
        )
        with torch.no_grad():
            q_values = self.online(self._to_tensor(states)).cpu().numpy()
        errors = np.abs(q_values - self.q_star[None])
        return errors.mean(axis=(1, 2)), errors[:, self.nonterminal_states].mean(
            axis=(1, 2)
        )

    def _update(self, agents: list[AgentState], update_mask: np.ndarray) -> float:
        """One masked gradient step for the agents in ``update_mask``; returns mean loss."""
        observation_size = agents[0].observation.shape[0]
        shape = (self.n_agents, self.batch_size)
        states = np.zeros(shape + (observation_size,))
        next_states = np.zeros(shape + (observation_size,))
        actions = np.zeros(shape, dtype=np.int64)
        rewards = np.zeros(shape)
        dones = np.ones(shape)
        for agent in agents:
            if update_mask[agent.index]:
                batch = agent.memory.sample(self.batch_size)
                states[agent.index] = batch.states
                next_states[agent.index] = batch.next_states
                actions[agent.index] = batch.actions
                rewards[agent.index] = batch.rewards
                dones[agent.index] = batch.dones

        bootstrap_model = (
            self.target if self.bootstrap_network == "target" else self.online
        )
        with torch.no_grad():
            next_q = bootstrap_model(self._to_tensor(next_states)).max(dim=-1).values
            targets = self._to_tensor(rewards) + self.gamma * next_q * (
                1.0 - self._to_tensor(dones)
            )

        action_index = torch.as_tensor(actions, device=self.device)
        q_values = self.online(self._to_tensor(states))
        chosen = q_values.gather(-1, action_index[..., None])[..., 0]
        per_agent_loss = ((chosen - targets) ** 2).mean(dim=1)
        if self.loss_form == "vector_form":
            per_agent_loss = per_agent_loss / self.n_actions
        mask = torch.as_tensor(update_mask, device=self.device)
        loss = (per_agent_loss * mask.to(per_agent_loss.dtype)).sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step(mask)
        return float(per_agent_loss[mask].mean().detach().cpu())

    @torch.no_grad()
    def _sync_target(self, sync_mask: np.ndarray) -> None:
        mask = torch.as_tensor(sync_mask, device=self.device)
        for target_tensor, online_tensor in zip(
            self.target.parameters(), self.online.parameters()
        ):
            shape = (-1,) + (1,) * (online_tensor.dim() - 1)
            target_tensor.copy_(
                torch.where(mask.view(shape), online_tensor, target_tensor)
            )

    # -------------------------------------------------------------------- train
    def train(self) -> TrainingResult:
        """Train all agents until solved or ``max_episodes``.

        Returns
        -------
        TrainingResult
            Scores, solve episodes and (Frozen Lake) Q* errors per agent.
        """
        agents = self._new_agents()
        mae_all: list[list[float]] = [[] for _ in agents]
        mae_nonterminal: list[list[float]] = [[] for _ in agents]
        start = time.perf_counter()
        iterations = 0
        logger.info(
            "TRAINING_STARTED | agents=%d | seeds=%s | max_episodes=%d | trainable_parameters=%d | device=%s",
            self.n_agents,
            self.seeds,
            self.max_episodes,
            self.trainable_parameters,
            self.device,
        )

        while any(agent.active for agent in agents):
            iterations += 1
            observations = np.stack([agent.observation for agent in agents])
            explore = np.array([agent.rng.random() < agent.epsilon for agent in agents])
            greedy = (
                self._greedy_actions(self.online, observations)
                if not explore.all()
                else None
            )

            update_mask = np.zeros(self.n_agents, dtype=bool)
            sync_mask = np.zeros(self.n_agents, dtype=bool)
            for agent in agents:
                if not agent.active:
                    continue
                if explore[agent.index]:
                    action = int(agent.rng.integers(self.n_actions))
                else:
                    action = int(greedy[agent.index])
                raw_next, reward, terminated, truncated, _ = agent.env.step(action)
                next_observation = encode_observation(self.cfg, raw_next)
                no_bootstrap = (
                    terminated or truncated
                    if self.terminal_mask == "terminated_or_truncated"
                    else terminated
                )
                agent.memory.add(
                    agent.observation, action, reward, next_observation, no_bootstrap
                )
                agent.episode_return += float(reward)
                agent.total_env_steps += 1

                if terminated or truncated:
                    self._finish_episode(agent)
                    continue
                agent.observation = next_observation
                step = agent.step_in_episode
                update_mask[agent.index] = (
                    len(agent.memory) >= self.batch_size
                    and step % self.update_model_every == 0
                )
                sync_mask[agent.index] = step % self.update_target_every == 0
                agent.step_in_episode += 1

            if update_mask.any():
                loss = self._update(agents, update_mask)
                logger.debug(
                    "update | agents=%d | loss=%.6f", int(update_mask.sum()), loss
                )
            if sync_mask.any():
                self._sync_target(sync_mask)
            if self.track_q_star_mae:
                all_pairs, nonterminal_pairs = self._q_star_mae()
                for agent in agents:
                    if (
                        agent.active
                        or len(mae_all[agent.index]) < agent.total_env_steps
                    ):
                        mae_all[agent.index].append(float(all_pairs[agent.index]))
                        mae_nonterminal[agent.index].append(
                            float(nonterminal_pairs[agent.index])
                        )

        for agent in agents:
            agent.env.close()
        elapsed = time.perf_counter() - start
        return TrainingResult(
            seeds=self.seeds,
            scores=[agent.scores for agent in agents],
            solved_at_episode_index=[agent.solved_at_episode_index for agent in agents],
            total_env_steps=[agent.total_env_steps for agent in agents],
            final_epsilon=[agent.epsilon for agent in agents],
            q_star_mae={"all_64": mae_all, "nonterminal_44": mae_nonterminal}
            if self.track_q_star_mae
            else None,
            wall_clock_seconds=elapsed,
            lockstep_iterations=iterations,
            trainable_parameters=self.trainable_parameters,
        )

    def _finish_episode(self, agent: AgentState) -> None:
        """Record the score, check the solve rule, decay epsilon, reset the environment."""
        agent.scores.append(agent.episode_return)
        recent = float(np.mean(agent.scores[-self.solve_window :]))
        logger.info(
            "TRAIN_EPISODE_COMPLETED | agent_seed=%d | episode=%d/%d | score=%.1f | mean_last_%d=%.3f | epsilon=%.4f | env_steps=%d",
            agent.seed,
            agent.episode + 1,
            self.max_episodes,
            agent.episode_return,
            self.solve_window,
            recent,
            agent.epsilon,
            agent.total_env_steps,
        )
        if self._is_solved(agent.scores):
            agent.solved_at_episode_index = agent.episode
            agent.active = False
            logger.info(
                "AGENT_SOLVED | agent_seed=%d | episode_index=%d | env_steps=%d",
                agent.seed,
                agent.episode,
                agent.total_env_steps,
            )
        agent.episode += 1
        agent.epsilon = max(self.epsilon_min, agent.epsilon * self.epsilon_decay)
        if agent.episode >= self.max_episodes and agent.active:
            agent.active = False
            logger.info(
                "AGENT_STOPPED_UNSOLVED | agent_seed=%d | episodes=%d | env_steps=%d",
                agent.seed,
                agent.episode,
                agent.total_env_steps,
            )
        agent.episode_return = 0.0
        agent.step_in_episode = 0
        if agent.active:
            observation, _ = agent.env.reset()
            agent.observation = encode_observation(self.cfg, observation)

    # ----------------------------------------------------------------- evaluate
    def evaluate_greedy(self, episodes: int) -> list[list[float]]:
        """Greedy (epsilon = 0) episodes with the final online parameters.

        Not a paper metric: an extra check of the learned policy. Each agent
        uses fresh environments seeded with ``seed + 1_000_000 + episode``.

        Parameters
        ----------
        episodes : int
            Episodes per agent.

        Returns
        -------
        list of list of float
            Scores per agent.
        """
        returns: list[list[float]] = [[] for _ in self.seeds]
        for episode in range(episodes):
            envs = [make_environment(self.cfg) for _ in self.seeds]
            observations = []
            for env, seed in zip(envs, self.seeds):
                observation, _ = env.reset(seed=seed + 1_000_000 + episode)
                observations.append(encode_observation(self.cfg, observation))
            observations = np.stack(observations)
            running = np.ones(self.n_agents, dtype=bool)
            totals = np.zeros(self.n_agents)
            while running.any():
                actions = self._greedy_actions(self.online, observations)
                for index, env in enumerate(envs):
                    if not running[index]:
                        continue
                    raw, reward, terminated, truncated, _ = env.step(
                        int(actions[index])
                    )
                    totals[index] += float(reward)
                    observations[index] = encode_observation(self.cfg, raw)
                    if terminated or truncated:
                        running[index] = False
            for index, env in enumerate(envs):
                env.close()
                returns[index].append(float(totals[index]))
        return returns
