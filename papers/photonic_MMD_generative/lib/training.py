# Training semantics follow merlinquantum/generative_classical_training
# (src/models/training.py::Trainer, itself adapted from XanaduAI/iqpopt,
# Apache License 2.0), commit 17a12c4.
"""Adam training of an MMD loss with a fresh random key per step.

Matches the original Trainer (jaxopt.OptaxSolver(optax.adam), non-turbo
path) without the deprecated jaxopt dependency:
- step t uses key_t from repeated splits of the root key, so K and Z are
  resampled every step;
- the recorded training loss is the estimate at the parameters *before*
  the update (jaxopt's ``state.value``);
- convergence check (when ``convergence_interval`` is set): after more than
  2 * interval steps, stop if |mean(last window) - mean(previous window)|
  <= std(last window) / sqrt(interval) / 2, or if the last window's mean is
  larger than the previous one.
"""

from __future__ import annotations

import logging
import time

import jax
import numpy as np
import optax

jax.config.update("jax_enable_x64", True)

logger = logging.getLogger(__name__)


def converged(losses: list[float], interval: int) -> bool:
    """Return True when the original code's window-based stopping rule fires."""
    if len(losses) <= 2 * interval + 1:
        return False
    last = np.asarray(losses[-interval:])
    previous = np.asarray(losses[-2 * interval : -interval])
    small_change = (
        abs(previous.mean() - last.mean()) <= last.std() / np.sqrt(interval) / 2
    )
    return bool(small_change or last.mean() > previous.mean())


def train_adam(
    loss_fn,
    params,
    X_train,
    steps: int,
    learning_rate: float,
    key,
    convergence_interval: int | None,
    log_every: int = 1,
) -> dict:
    """Minimize ``loss_fn(params, key, X_train)`` with Adam.

    Parameters
    ----------
    loss_fn : Callable
        Jitted stochastic loss of (params, key, X).
    params : jax array
        Initial parameters.
    X_train : jax array
        Target training samples.
    steps : int
        Maximum number of optimizer steps.
    learning_rate : float
        Adam step size.
    key : jax PRNG key
        Root of the per-step key chain (one split per step).
    convergence_interval : int | None
        Window of the stopping rule; None disables it.
    log_every : int
        Log one TRAIN_STEP_COMPLETED line every this many steps (INFO).
        Default value is 1.

    Returns
    -------
    dict
        ``params`` (final), ``losses`` (list of floats), ``steps_run``,
        ``converged`` (bool), ``seconds``.
    """
    optimizer = optax.adam(learning_rate)
    state = optimizer.init(params)
    value_and_grad = jax.jit(jax.value_and_grad(loss_fn))

    @jax.jit
    def update(params, state, key):
        value, grad = value_and_grad(params, key, X_train)
        updates, state = optimizer.update(grad, state, params)
        return optax.apply_updates(params, updates), state, value

    losses: list[float] = []
    stopped_early = False
    start = time.perf_counter()
    logger.info(
        "TRAINING_STARTED | steps=%d | optimizer=adam | lr=%g | convergence_interval=%s",
        steps,
        learning_rate,
        convergence_interval,
    )
    for step in range(1, steps + 1):
        key, subkey = jax.random.split(key)
        step_start = time.perf_counter()
        params, state, value = update(params, state, subkey)
        loss_value = float(value)
        losses.append(loss_value)
        if step % log_every == 0 or step == 1:
            logger.info(
                "TRAIN_STEP_COMPLETED | step=%d/%d | loss=%.8g | seconds=%.3f",
                step,
                steps,
                loss_value,
                time.perf_counter() - step_start,
            )
        if not np.isfinite(loss_value):
            raise FloatingPointError(f"Non-finite loss {loss_value} at step {step}.")
        if convergence_interval is not None and converged(losses, convergence_interval):
            stopped_early = True
            logger.info(
                "TRAINING_CONVERGED | step=%d | rule=window_%d",
                step,
                convergence_interval,
            )
            break
    seconds = time.perf_counter() - start
    logger.info(
        "TRAINING_COMPLETED | steps_run=%d | converged=%s | seconds=%.2f",
        len(losses),
        stopped_early,
        seconds,
    )
    return {
        "params": params,
        "losses": losses,
        "steps_run": len(losses),
        "converged": stopped_early,
        "seconds": seconds,
    }


__all__ = ["converged", "train_adam"]
