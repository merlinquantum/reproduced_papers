"""D-Wave Ocean solvers: QPU, Simulated Annealing, Tabu, and Leap hybrid.

**Reproducibility.** The two local samplers (Simulated Annealing and Tabu) accept
an explicit ``seed`` and are seeded from the harness. The QPU and Leap hybrid
solvers run on remote hardware/services, so their results cannot be made
deterministic -- and neither can ``minorminer``'s embedding search, which is
seeded here only where the API allows it.
"""

from __future__ import annotations

import time
from functools import partial

import dimod
from dwave.embedding.chain_strength import uniform_torque_compensation
from dwave.samplers import SimulatedAnnealingSampler, TabuSampler
from dwave.system import LeapHybridSampler
from dwave.system.composites import FixedEmbeddingComposite
from dwave.system.samplers import DWaveSampler
from minorminer import find_embedding

#: Default minorminer embedding-search budget when no timeout is given.
DEFAULT_EMBEDDING_TIMEOUT = 1000


def _sample_to_bitstring(sample, size: int) -> list[int]:
    """Convert a dimod sample mapping into a dense 0/1 list."""
    bitstring = [0] * size
    for key, value in sample.items():
        bitstring[int(key)] = int(value)
    return bitstring


def run_dwave_qpu(
    Q: dict,
    size: int,
    solver: str,
    num_reads: int = 1000,
    timeout: int | None = None,
    seed: int | None = None,
) -> list[int] | None:
    """Solve a Q-score instance on a D-Wave QPU.

    Args:
        Q: QUBO as a ``{(i, j): coeff}`` dict, the form the samplers take.
        size: problem size.
        solver: QPU solver name.
        num_reads: number of states to read from the sampler.
        timeout: budget for the ``minorminer`` embedding search.
        seed: seeds the embedding search (the anneal itself is hardware).

    Returns:
        Bitstring of the best sample; ``None`` when no embedding could be found.
    """
    chain_strength = partial(uniform_torque_compensation, prefactor=2)
    sampler = DWaveSampler(solver=solver)
    try:
        bqm = dimod.BQM.from_qubo(Q)
        source_edgelist = list(bqm.quadratic) + [(v, v) for v in bqm.linear]
        target_graph = sampler.to_networkx_graph()
        embedding_kwargs = {
            "timeout": DEFAULT_EMBEDDING_TIMEOUT if timeout is None else timeout
        }
        if seed is not None:
            embedding_kwargs["random_seed"] = int(seed)
        embedding = find_embedding(source_edgelist, target_graph, **embedding_kwargs)
    except Exception:  # noqa: BLE001 - any failure here means no embedding exists
        print("Failed to find embedding.")
        return None

    sampler_embedded = FixedEmbeddingComposite(DWaveSampler(solver=solver), embedding)
    sampleset = sampler_embedded.sample_qubo(
        Q,
        chain_strength=chain_strength,
        num_reads=num_reads,
        label=f"Problem-{size:2d}",
    )
    return _sample_to_bitstring(sampleset.first.sample, size)


def run_SA(
    Q: dict,
    size: int,
    num_reads: int,
    timeout: int | None = None,
    seed: int | None = None,
) -> list[int] | None:
    """Solve a Q-score instance with D-Wave's Simulated Annealing sampler.

    Args:
        Q: QUBO as a ``{(i, j): coeff}`` dict, the form the samplers take.
        size: problem size.
        num_reads: number of anneal restarts.
        timeout: soft budget; ``None`` is returned when it is exceeded.
        seed: makes the anneal reproducible.

    Returns:
        Bitstring of the best sample, or ``None`` if the timeout was exceeded.
    """
    start = time.time()

    # No `chain_strength` here: that is an embedding parameter for a QPU, and a
    # software annealer has no chains to break.
    sample_kwargs = {"num_reads": num_reads, "label": f"Problem-{size:2d}"}
    if seed is not None:
        sample_kwargs["seed"] = int(seed)
    sampleset = SimulatedAnnealingSampler().sample_qubo(Q, **sample_kwargs)

    if timeout is not None and time.time() - start > timeout:
        print("Failed to find a solution within timeout limit.")
        return None

    return _sample_to_bitstring(sampleset.first.sample, size)


def run_tabu(
    Q: dict,
    size: int,
    num_reads: int | None = None,
    timeout: int | None = None,
    seed: int | None = None,
) -> list[int] | None:
    """Solve a Q-score instance with D-Wave's Tabu sampler.

    Args:
        Q: QUBO as a ``{(i, j): coeff}`` dict, the form the samplers take.
        size: problem size.
        num_reads: number of tabu restarts. Left unset by the shipped config: a
            tabu "read" runs a complete local search to convergence, not a single
            shot or anneal, so matching another solver's shot count multiplies the
            runtime (~100x at 5000 reads) without changing the answer at these
            sizes.
        timeout: soft budget; ``None`` is returned when it is exceeded.
        seed: makes the search reproducible.

    Returns:
        Bitstring of the best sample, or ``None`` if the timeout was exceeded.
    """
    start = time.time()

    sample_kwargs = {"label": f"Problem-{size:2d}"}
    if num_reads is not None:
        sample_kwargs["num_reads"] = int(num_reads)
    if seed is not None:
        sample_kwargs["seed"] = int(seed)
    sampleset = TabuSampler().sample_qubo(Q, **sample_kwargs)

    if timeout is not None and time.time() - start > timeout:
        print("Failed to find a solution within timeout limit.")
        return None

    return _sample_to_bitstring(sampleset.first.sample, size)


def run_hybrid(Q: dict, size: int, timeout: int | None = None) -> list[int]:
    """Solve a Q-score instance with the D-Wave Leap hybrid solver.

    Args:
        Q: QUBO as a ``{(i, j): coeff}`` dict, the form the samplers take.
        size: problem size.
        timeout: native ``time_limit`` for the hybrid service.

    Returns:
        Bitstring of the best sample.
    """
    bqm = dimod.BQM.from_qubo(Q)
    sampler = LeapHybridSampler(solver={"category": "hybrid"})
    sampleset = sampler.sample(bqm, label=f"Problem-{size:2d}", time_limit=timeout)
    return _sample_to_bitstring(sampleset.first.sample, size)
