"""Solver dispatch: name in, bitstring out.

:func:`run_solver` is the reusable caller, decoupled from the benchmark. Hand it a
QUBO matrix (and, for the graph-native solvers, the source graph) plus solver
options and it returns a bitstring. It does not sample graphs, time itself, or enforce
timeouts -- that is :mod:`benchmark`'s job.

Alongside the dispatch are the capability sets the harness needs to treat solvers
differently: which ones have no native time limit, which need the graph rather
than the QUBO, which require an explicit read budget.

**Seeding.** Callers pass the instance's derived ``seed`` (see :mod:`lib.seeding`).
It is applied two ways: to the global generators, and as an explicit argument to
the solvers that accept one. A ``seed`` already present in ``solver_options``
always wins, so a config can pin a solver's seed by hand.
"""

from __future__ import annotations

from lib.seeding import set_global_seed
from networkx import Graph
from utils.qubo import qubo_matrix_to_dict, to_quadratic_program

#: Every solver name accepted by :func:`run_solver`.
SOLVERS = (
    "obliq-static",
    "obliq-vqc",
    "obliq-hybrid",
    "Photonic_CVARVQE",
    "QAOA",
    "Simulated_Annealing",
    "tabu",
    "hybrid",
    "Advantage_system4.1",
)

#: ObliQ variants (all served by :mod:`models.obliq`).
OBLIQ_SOLVERS = {"obliq-static", "obliq-vqc", "obliq-hybrid"}

#: Solvers with no native time limit; :mod:`benchmark` enforces theirs by
#: killing an out-of-process run. The D-Wave family instead takes ``timeout`` as a
#: native sampler argument, so it is never process-wrapped.
PROCESS_TIMEOUT_SOLVERS = {
    "QAOA",
    "Photonic_CVARVQE",
    "obliq-vqc",
    "obliq-static",
    "obliq-hybrid",
}

#: Solvers that need the source graph rather than the QUBO dict.
GRAPH_NATIVE_SOLVERS = {"QAOA", "Photonic_CVARVQE"}

#: Solvers that require an explicit read/sample budget.
NUM_READS_SOLVERS = {"Advantage_system4.1", "Simulated_Annealing"}


def run_solver(
    solver: str,
    Q,
    *,
    graph: Graph | None = None,
    problem_type: str | None = None,
    solver_options: dict | None = None,
    timeout: int | None = None,
    num_reads: int | None = None,
    provider: str | None = None,
    backend: str | None = None,
    seed: int | None = None,
):
    """Run one solver on a prepared instance and return its bitstring.

    Args:
        solver: solver name (see :data:`SOLVERS`).
        Q: QUBO as a symmetric matrix. Consumed by ObliQ, and by the D-Wave family
            after conversion to a dict; QAOA and CVaR-VQE build their instance
            from ``graph`` instead.
        graph: source graph -- required by the solvers in
            :data:`GRAPH_NATIVE_SOLVERS`.
        problem_type: ``"max-cut"`` / ``"max-clique"`` -- required by the
            graph-native solvers.
        solver_options: solver-specific keyword arguments.
        timeout: forwarded to the D-Wave samplers as a native time budget;
            ignored by the others (their timeout is enforced by the caller).
        num_reads: reads/samples for the D-Wave QPU / Simulated Annealing.
        provider: QAOA hardware provider.
        backend: QAOA provider backend, or a Quandela platform for the photonic
            solvers.
        seed: derived per-instance seed for the solver's own randomness.

    Returns:
        A bitstring (sequence of 0/1), or ``None``/``nan`` when the solver
        produced no usable result.

    Raises:
        NotImplementedError: for an unimplemented solver.
        ValueError: for missing or invalid solver arguments.
    """
    if num_reads is None and solver in NUM_READS_SOLVERS:
        raise ValueError("num_reads has not been submitted while required by solver.")

    if graph is None and solver in GRAPH_NATIVE_SOLVERS:
        raise ValueError(
            f"{', '.join(sorted(GRAPH_NATIVE_SOLVERS))} solvers require graph."
        )

    options = dict(solver_options or {})

    # Covers library internals that reach for a global generator without asking.
    set_global_seed(seed)

    # Solver backends are imported lazily: a photonic run should not import
    # qiskit, and a QAOA run should not import perceval/merlin.
    if solver == "QAOA":
        from models.qaoa import run_QAOA

        options.setdefault("seed", seed)
        return run_QAOA(
            to_quadratic_program(problem_type, graph), provider, backend, **options
        )

    if solver == "Photonic_CVARVQE":
        from models.cvar_vqe import run_photonic_cvarvqe

        options.setdefault("seed", seed)
        return run_photonic_cvarvqe(graph, problem_type, **options)

    if solver in OBLIQ_SOLVERS:
        from models.obliq import run_obliq_solver

        options.setdefault("seed", seed)
        return run_obliq_solver(Q, variant=solver, backend=backend, **options).bitstring

    # D-Wave family: their samplers take a QUBO dict, so the matrix is converted
    # here -- the one place that form is still needed. `timeout` reaches the Leap
    # hybrid solver as its native `time_limit` and the QPU path as the embedding
    # search budget; for the two local samplers it is a post-hoc check.
    size = Q.shape[0]
    Q_dict = qubo_matrix_to_dict(Q)
    if solver == "Advantage_system4.1":
        from models.dwave import run_dwave_qpu

        return run_dwave_qpu(Q_dict, size, solver, num_reads, timeout, seed=seed)
    if solver == "hybrid":
        from models.dwave import run_hybrid

        return run_hybrid(Q_dict, size, timeout)
    if solver == "Simulated_Annealing":
        from models.dwave import run_SA

        return run_SA(Q_dict, size, num_reads, timeout, seed=seed)
    if solver == "tabu":
        from models.dwave import run_tabu

        return run_tabu(Q_dict, size, num_reads, timeout, seed=seed)

    raise NotImplementedError(f"Provided Solver {solver} is not implemented")
