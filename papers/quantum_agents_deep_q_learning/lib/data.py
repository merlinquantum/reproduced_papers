"""Environments, replay memory and replay-batch loading.

Reinforcement learning has no static dataset: training data are transitions
generated online by the Gymnasium environments. This module owns everything
that produces or stores those transitions:

* ``make_environment``: Gymnasium environment matching the paper's setting.
* ``ReplayMemory``: one agent's experience-replay buffer.
* ``build_dataloader``: fills a replay memory with uniformly random actions
  and yields replay batches; used to check the data pipeline (Phase 2) and in
  tests. Training itself fills the memory with epsilon-greedy actions.

Environment facts (verified against gym 0.17.2 source, see LOG.md):

* CartPole-v0: 200-step limit, solved at mean score >= 195 over 100 episodes.
* FrozenLake: the paper's FrozenLake-v0 is FrozenLake-v1 in Gymnasium; the
  authors used ``is_slippery=False`` and gym's registered 100-step limit.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from .config_access import require, require_choice

logger = logging.getLogger(__name__)

SUPPORTED_ENVIRONMENTS = ("CartPole-v0", "FrozenLake-v1")
FROZENLAKE_GRID_SIDE = 4
FROZENLAKE_N_STATES = FROZENLAKE_GRID_SIDE**2


def resolve_data_dir(cfg: dict[str, Any]) -> Path:
    """Return this paper's directory under the shared data root.

    Precedence follows ``AGENTS.md``: ``--data-root`` / ``cfg["data_root"]``,
    then the ``DATA_DIR`` environment variable. Environments need no files;
    the directory only holds cached sources (paper PDF, environment sources).

    Parameters
    ----------
    cfg : dict
        Resolved configuration.

    Returns
    -------
    pathlib.Path
        ``<data_root>/quantum_agents_deep_q_learning``.

    Raises
    ------
    KeyError
        If neither ``data_root`` nor ``DATA_DIR`` is set.
    """
    data_root = cfg.get("data_root") or os.environ.get("DATA_DIR")
    if not data_root:
        raise KeyError(
            "Set --data-root (cfg['data_root']) or the DATA_DIR environment variable"
        )
    return Path(data_root) / "quantum_agents_deep_q_learning"


def make_environment(cfg: dict[str, Any]) -> gym.Env:
    """Create the Gymnasium environment described by ``cfg["environment"]``.

    Parameters
    ----------
    cfg : dict
        Resolved configuration with an ``environment`` section.

    Returns
    -------
    gymnasium.Env
        Environment wrapped in its ``TimeLimit``.

    Raises
    ------
    ValueError
        If the environment name is unsupported.
    """
    name = require_choice(cfg, "environment.name", SUPPORTED_ENVIRONMENTS)
    max_episode_steps = require(cfg, "environment.max_episode_steps")
    kwargs: dict[str, Any] = {}
    if max_episode_steps is not None:
        kwargs["max_episode_steps"] = int(max_episode_steps)
    if name == "FrozenLake-v1":
        kwargs["is_slippery"] = bool(require(cfg, "environment.is_slippery"))
        kwargs["map_name"] = "4x4"
    return gym.make(name, **kwargs)


def n_actions(cfg: dict[str, Any]) -> int:
    """Number of discrete actions: 2 for Cart Pole, 4 for Frozen Lake."""
    name = require_choice(cfg, "environment.name", SUPPORTED_ENVIRONMENTS)
    return 2 if name == "CartPole-v0" else 4


def encode_observation(cfg: dict[str, Any], observation: Any) -> np.ndarray:
    """Convert a raw Gymnasium observation into the 4-value model input.

    Cart Pole observations are used as is (float32 in Gymnasium, float64 in
    gym 0.17.2; the model casts them to its dtype). A Frozen Lake state index
    becomes the integer index itself, stored as a float: the PQC turns it into
    a computational basis state and the MLP into 4 bits (``state_bits``).

    Parameters
    ----------
    cfg : dict
        Resolved configuration.
    observation : Any
        Raw observation from ``env.step`` or ``env.reset``.

    Returns
    -------
    numpy.ndarray
        Shape ``(4,)`` for Cart Pole, ``(1,)`` holding the state index for
        Frozen Lake.
    """
    name = require_choice(cfg, "environment.name", SUPPORTED_ENVIRONMENTS)
    if name == "CartPole-v0":
        return np.asarray(observation, dtype=np.float64).reshape(4)
    return np.asarray([int(observation)], dtype=np.float64)


def state_bits(state_indices: np.ndarray) -> np.ndarray:
    """Frozen Lake state index -> 4 bits, most significant bit first.

    Qubit / input ``i`` carries bit ``i`` with ``i = 0`` the most significant
    bit, matching the paper's basis-state encoding (state 1 -> ``0001``).

    Parameters
    ----------
    state_indices : numpy.ndarray
        Integer-valued array of any shape.

    Returns
    -------
    numpy.ndarray
        Array of shape ``state_indices.shape + (4,)`` with 0/1 entries.
    """
    indices = np.asarray(state_indices).astype(np.int64)
    shifts = np.arange(FROZENLAKE_GRID_SIDE - 1, -1, -1)
    return ((indices[..., None] >> shifts) & 1).astype(np.float64)


@dataclass
class ReplayBatch:
    """A batch of transitions ``(s, a, r, s', done)``.

    ``done`` is the bootstrap mask (1.0 means no bootstrap from ``s'``).
    """

    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    dones: np.ndarray


class ReplayMemory:
    """Experience replay for one agent.

    Parameters
    ----------
    capacity : int | None
        Maximum stored transitions. ``None`` means unbounded (reference-code
        Cart Pole behaviour, where ``memory_length`` was never passed).
    observation_size : int
        Length of an encoded observation.
    eviction : str
        ``"fifo"`` (``collections.deque`` in the reference Cart Pole code) or
        ``"random"`` (the reference Frozen Lake code overwrites a random slot
        once full).
    sampling : str
        ``"with_replacement"`` (reference Cart Pole: ``random.choices``) or
        ``"without_replacement"`` (reference Frozen Lake: ``random.sample``).
    rng : numpy.random.Generator
        The owning agent's generator (eviction slots and batch indices).
    """

    _GROWTH_CHUNK = 4096

    def __init__(
        self,
        capacity: int | None,
        observation_size: int,
        eviction: str,
        sampling: str,
        rng: np.random.Generator,
    ) -> None:
        if eviction not in ("fifo", "random"):
            raise ValueError(f"Unknown replay eviction {eviction!r}")
        if sampling not in ("with_replacement", "without_replacement"):
            raise ValueError(f"Unknown replay sampling {sampling!r}")
        self.capacity = capacity
        self.eviction = eviction
        self.sampling = sampling
        self.rng = rng
        initial = capacity if capacity is not None else self._GROWTH_CHUNK
        self._states = np.zeros((initial, observation_size))
        self._next_states = np.zeros((initial, observation_size))
        self._actions = np.zeros(initial, dtype=np.int64)
        self._rewards = np.zeros(initial)
        self._dones = np.zeros(initial)
        self._size = 0
        self._fifo_next = 0

    def __len__(self) -> int:
        return self._size

    def _grow(self) -> None:
        extra = self._GROWTH_CHUNK
        self._states = np.concatenate(
            [self._states, np.zeros((extra, self._states.shape[1]))]
        )
        self._next_states = np.concatenate(
            [self._next_states, np.zeros((extra, self._next_states.shape[1]))]
        )
        self._actions = np.concatenate([self._actions, np.zeros(extra, dtype=np.int64)])
        self._rewards = np.concatenate([self._rewards, np.zeros(extra)])
        self._dones = np.concatenate([self._dones, np.zeros(extra)])

    def add(self, state, action, reward, next_state, done) -> None:
        """Store one transition, evicting according to the configured rule."""
        if self.capacity is None:
            if self._size == len(self._actions):
                self._grow()
            slot = self._size
            self._size += 1
        elif self._size < self.capacity:
            slot = self._size
            self._size += 1
        elif self.eviction == "fifo":
            slot = self._fifo_next
            self._fifo_next = (self._fifo_next + 1) % self.capacity
        else:
            slot = int(self.rng.integers(self.capacity))
        self._states[slot] = state
        self._actions[slot] = action
        self._rewards[slot] = reward
        self._next_states[slot] = next_state
        self._dones[slot] = float(done)

    def sample(self, batch_size: int) -> ReplayBatch:
        """Draw a batch of transitions.

        Raises
        ------
        ValueError
            If fewer than ``batch_size`` transitions are stored.
        """
        if self._size < batch_size:
            raise ValueError(
                f"Replay holds {self._size} transitions, batch needs {batch_size}"
            )
        if self.sampling == "with_replacement":
            indices = self.rng.integers(self._size, size=batch_size)
        else:
            indices = self.rng.choice(self._size, size=batch_size, replace=False)
        return ReplayBatch(
            states=self._states[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            next_states=self._next_states[indices],
            dones=self._dones[indices],
        )


def build_replay_memory(cfg: dict[str, Any], rng: np.random.Generator) -> ReplayMemory:
    """Create one agent's replay memory from ``cfg["training"]``."""
    name = require(cfg, "environment.name")
    observation_size = 4 if name == "CartPole-v0" else 1
    return ReplayMemory(
        capacity=require(cfg, "training.replay_capacity"),
        observation_size=observation_size,
        eviction=require(cfg, "training.replay_eviction"),
        sampling=require(cfg, "training.replay_sampling"),
        rng=rng,
    )


def build_dataloader(
    cfg: dict[str, Any],
    split: str,
    *,
    n_transitions: int | None = None,
    seed: int | None = None,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Yield replay batches collected with a uniformly random policy.

    Used to check the environment / replay pipeline without a model. During
    training the replay memory is filled by the epsilon-greedy agent instead
    (``lib.dqn``). Only ``split="train"`` exists: RL has no held-out split.

    Parameters
    ----------
    cfg : dict
        Resolved configuration.
    split : str
        Must be ``"train"``.
    n_transitions : int | None
        Transitions to collect before batching. Default value is None, which
        means ``10 * training.batch_size``.
    seed : int | None
        Seed for the environment and the sampler. Default value is None,
        which means the first agent seed in ``agents.seeds``.

    Yields
    ------
    tuple of numpy.ndarray
        ``(states, actions, rewards, next_states, dones)`` with leading
        dimension ``training.batch_size``.

    Raises
    ------
    ValueError
        If ``split`` is not ``"train"``.
    """
    if split != "train":
        raise ValueError(
            f"Reinforcement learning has only a 'train' split, got {split!r}"
        )
    batch_size = int(require(cfg, "training.batch_size"))
    if seed is None:
        seed = int(require(cfg, "agents.seeds")[0])
    if n_transitions is None:
        n_transitions = 10 * batch_size
    rng = np.random.default_rng(seed)
    memory = build_replay_memory(cfg, rng)
    env = make_environment(cfg)
    observation, _ = env.reset(seed=seed)
    env.action_space.seed(seed)
    for _ in range(n_transitions):
        action = int(env.action_space.sample())
        next_observation, reward, terminated, truncated, _ = env.step(action)
        memory.add(
            encode_observation(cfg, observation),
            action,
            reward,
            encode_observation(cfg, next_observation),
            terminated or truncated,
        )
        observation = next_observation
        if terminated or truncated:
            observation, _ = env.reset()
    env.close()
    while True:
        batch = memory.sample(batch_size)
        yield batch.states, batch.actions, batch.rewards, batch.next_states, batch.dones
