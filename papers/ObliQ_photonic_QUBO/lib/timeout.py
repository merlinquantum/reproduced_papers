"""Out-of-process timeout enforcement.

Most solvers here have no native time limit: a variational loop will happily run
past the Q-score's 60 s per-instance budget. Killing the *process* is the only
reliable way to stop one mid-optimization, so the call is relocated to a spawned
child that can be terminated.
"""

from __future__ import annotations

import multiprocessing as mp
from collections.abc import Callable

import numpy as np


def _process_wrapper(
    queue: mp.Queue, func: Callable, args: tuple, kwargs: dict
) -> None:
    """Execute ``func`` in the child and push ``(success, payload)`` to a queue."""
    try:
        result = func(*args, **kwargs)
        queue.put((True, result))
    except Exception as exc:  # noqa: BLE001 - ferried to the parent and re-raised
        queue.put((False, exc))


def run_with_timeout(func: Callable, timeout: int | None, /, *args, **kwargs):
    """Execute ``func``, terminating it if it overruns ``timeout`` seconds.

    ``func`` and ``timeout`` are positional-only so that a ``timeout`` entry in
    ``kwargs`` is forwarded to ``func`` rather than colliding with this wrapper.

    The child is started with the ``spawn`` context, so ``func`` and its arguments
    must be picklable and the child re-imports the module tree. That also means
    the child does *not* inherit the parent's RNG state -- solver seeds have to be
    passed explicitly, which :mod:`models.solver` does.

    Args:
        func: callable to run.
        timeout: budget in seconds; ``None`` or non-positive runs ``func`` inline.
        *args: positional arguments for ``func``.
        **kwargs: keyword arguments for ``func``.

    Returns:
        Whatever ``func`` returned, or ``nan`` if it was killed.

    Raises:
        Exception: whatever ``func`` raised, re-raised in the parent.
    """
    if timeout is None or timeout <= 0:
        return func(*args, **kwargs)

    ctx = mp.get_context("spawn")
    queue: mp.Queue = ctx.Queue()
    process = ctx.Process(
        target=_process_wrapper, args=(queue, func, args, kwargs), daemon=True
    )
    process.start()
    process.join(timeout)

    if process.is_alive():
        process.terminate()
        process.join()
        queue.close()
        queue.join_thread()
        return np.nan

    if queue.empty():
        queue.close()
        queue.join_thread()
        return np.nan

    success, payload = queue.get()
    queue.close()
    queue.join_thread()
    if success:
        return payload
    raise payload
