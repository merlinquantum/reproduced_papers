"""Deterministic seeding for the benchmark.

Every source of randomness in a run traces back to one number: ``sweep.seed`` in
the config. The sweep walks that base seed forward one step per instance
(``seed + idx``); that instance seed selects the graph *verbatim* (see
:func:`utils.graphs.sample_instance_graph`), and the solver's own randomness runs
off a sub-seed derived from it by :func:`derive_seed`.

Why derive rather than reuse: the instance seed already selects the graph. Using
it verbatim for the solver's parameter initialization too would couple the two, so
a solver's starting point would be a deterministic function of its problem
instance. Hashing with a domain label decorrelates them while keeping everything
reproducible from the single base seed.

Why hashlib and not ``hash()``: sweeps run under
``ProcessPoolExecutor(mp_context="spawn")``, and Python salts ``hash()`` per
process (``PYTHONHASHSEED``). A blake2b digest is stable across processes,
platforms and interpreter versions, so a worker derives the same sub-seed as the
parent would have.

Nothing here reads the config, so adding derived seeds does not change any
config's content hash -- existing ``results/<hash>/`` directories stay valid.
"""

from __future__ import annotations

import hashlib
import random
from typing import Any

import numpy as np

#: Upper bound for derived seeds. Below 2**31 so the value is safe for NumPy's
#: legacy ``seed()``, torch, and every sampler that wants a C ``int``.
SEED_MODULUS = 2**31 - 1


def derive_seed(base_seed: int | None, *parts: Any) -> int | None:
    """Derive a stable sub-seed from ``base_seed`` and one or more labels.

    Returns ``None`` when ``base_seed`` is ``None`` (i.e. the caller explicitly
    asked for an unseeded run); otherwise the result is a deterministic function
    of the inputs, identical in any process on any platform.

    Call sites label their domain, so the sub-seeds stay independent:
    ``derive_seed(seed, "solver", name)`` for a solver's own randomness and
    ``derive_seed(seed, "beta")`` for the sampled random baseline.

    >>> derive_seed(101200, "solver", "obliq-hybrid") == derive_seed(101200, "solver", "obliq-hybrid")
    True
    """
    if base_seed is None:
        return None
    payload = "|".join([str(base_seed), *(str(part) for part in parts)])
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % SEED_MODULUS


def set_global_seed(seed: int | None) -> None:
    """Seed the ``random``, NumPy and torch global generators.

    Solvers that expose an explicit seed argument are seeded through it (see
    :mod:`models.solver`); this covers the rest -- library internals that reach
    for a global generator without asking, notably:

    * ``merlin.QuantumLayer`` trainable-parameter initialization,
    * ``torch.multinomial`` sampling in the CVaR-VQE decoder,
    * ``qiskit_algorithms``' random initial point for ``SamplingVQE``,
    * ``networkx``/NumPy draws inside the random-baseline estimators.

    torch is imported lazily so the classical-only paths (D-Wave, QAOA) do not
    pay for it.
    """
    if seed is None:
        return

    seed = int(seed) % SEED_MODULUS
    random.seed(seed)
    np.random.seed(seed % 2**32)

    try:
        import torch
    except ImportError:  # pragma: no cover - torch is optional for classical solvers
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - CPU in CI
        torch.cuda.manual_seed_all(seed)
