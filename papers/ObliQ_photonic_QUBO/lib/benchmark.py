"""Run the Q-score benchmark: one instance, a sweep, or from the command line.

:func:`run_instance` evaluates a single instance -- sample a graph, run a solver,
score the bitstring, compute beta. :func:`run_sweep` repeats that across problem
sizes and stores the results. They share one contract, the seed sequence:

    size s uses seeds [seed, seed + 1, ..., seed + instances - 1]
    the next size continues where the previous one stopped

That sequence is the reproducibility backbone of the harness. An instance seed
selects its graph verbatim (:mod:`utils.graphs`) and the solver's own randomness
runs off a sub-seed derived from it (:mod:`lib.seeding`), so re-running a sweep
reproduces both the instances and the solver trajectories. :mod:`utils.plotter`'s
exact-beta path regenerates the instances by replaying the same sequence.

Two subcommands, both described by ``benchmark_cli.json`` so adding a flag needs
no change here. Run as a module from the paper directory (this file lives inside
the ``lib`` package, so ``python lib/benchmark.py`` would break its own
``lib.config``/``lib.seeding`` imports -- ``-m`` puts the paper directory, not
``lib/``, on ``sys.path``)::

    # a full sweep from a config (writes results/<hash>/)
    python -m lib.benchmark sweep --config configs/obliq_maxclique.json

    # override any declared config path
    python -m lib.benchmark sweep --config configs/obliq_maxclique.json --sizes 2,3,4 --instances 20

    # one instance, no config file
    python -m lib.benchmark run --problem max-clique --size 8 --solver obliq-hybrid --seed 101200

Also reachable without a sweep-specific config file, through the repository's
shared runner (see :mod:`lib.runner`)::

    python implementation.py --paper ObliQ_photonic_QUBO --config configs/obliq_maxclique.json

Each sweep is content-addressed by a hash of its config: the results
(``results.json``) and a copy of the resolved config (``config.json``) are written
to ``<output.dir>/<hash>/``, with ``output.dir`` defaulting to ``results``.

What lives elsewhere: solver dispatch in :mod:`models.solver`, timeout enforcement
in :mod:`lib.timeout`, instance sampling in :mod:`utils.graphs`.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from lib.config import (
    CONFIG_FILE,
    RESULTS_FILE,
    add_arguments,
    apply_overrides,
    collect_kwargs,
    load_cli_spec,
    load_config,
    run_dir,
)
from lib.seeding import derive_seed
from lib.timeout import run_with_timeout
from models.solver import PROCESS_TIMEOUT_SOLVERS, run_solver
from networkx import Graph
from utils.graphs import sample_instance_graph
from utils.qubo import build_qubo, calculate_beta, exact_optimum, qubo_objective

_HERE = Path(__file__).resolve().parent
_CLI_SPEC = _HERE / "benchmark_cli.json"


# ---------------------------------------------------------------------------
# One instance
# ---------------------------------------------------------------------------
def _objective_or_nan(bitstring, elapsed, timeout, enforce_timeout, Q) -> float:
    """Score a solver bitstring against ``Q``, or return ``nan`` on failure.

    The score is the negated QUBO energy, i.e. the cut or clique size. ``nan`` is
    returned when a non-enforced run overran ``timeout`` (the soft check) or when
    the solver produced no usable bitstring (``None``/``nan``).
    """
    if (
        not enforce_timeout
        and timeout is not None
        and timeout > 0
        and elapsed > timeout
    ):
        return float("nan")
    if bitstring is None or (isinstance(bitstring, float) and np.isnan(bitstring)):
        return float("nan")
    return -qubo_objective(Q, bitstring)


def run_instance(
    problem_type: str,
    size: int,
    solver: str,
    timeout: int | None = None,
    seed: int | None = None,
    num_reads: int | None = None,
    provider: str | None = None,
    backend: str | None = None,
    min_timeout_size: int | None = None,
    solver_options: dict | None = None,
) -> tuple[float, float, Graph]:
    """Evaluate one Q-score instance.

    Args:
        problem_type: ``"max-cut"`` or ``"max-clique"``.
        size: size of the problem instance.
        solver: solver name (see :data:`models.solver.SOLVERS`).
        timeout: maximum time the solver may use, in seconds.
        seed: instance seed; ``None`` draws one at random.
        num_reads: reads/samples for the D-Wave QPU / Simulated Annealing.
        provider: hardware provider (QAOA).
        backend: backend name (QAOA) or Quandela platform (photonic).
        min_timeout_size: smallest size that enforces process-based timeouts.
        solver_options: extra keyword arguments forwarded to the solver.

    Returns:
        ``(objective, elapsed, graph)``. ``objective`` is the cut/clique size
        found, or ``nan`` when the run failed or timed out.

        Beta is deliberately not computed here. The sweep discards it and
        :mod:`utils.plotter` derives it from the stored objectives, so scoring the exact
        baselines on every instance -- 1000 random-search passes per Max-Clique
        instance -- would be pure waste. The graph is returned instead, which is all
        :func:`utils.qubo.calculate_beta` needs.

    Raises:
        NotImplementedError: for an unimplemented problem type or solver.
        ValueError: for missing or invalid solver arguments.
    """
    if seed is None:
        seed = int(np.random.randint(100000))
    G = sample_instance_graph(size, seed)
    Q = build_qubo(problem_type, G)

    enforce_timeout = (
        timeout is not None
        and timeout > 0
        and (min_timeout_size is None or size >= min_timeout_size)
    )

    solver_kwargs = {
        "graph": G,
        "problem_type": problem_type,
        "solver_options": solver_options,
        "timeout": timeout,
        "num_reads": num_reads,
        "provider": provider,
        "backend": backend,
        "seed": derive_seed(seed, "solver", solver),
    }

    # Solvers without a native time limit are killed out-of-process when the
    # timeout is enforced; the D-Wave family gets ``timeout`` as a sampler
    # argument and is called directly.
    process_timeout = enforce_timeout and solver in PROCESS_TIMEOUT_SOLVERS
    start_time = time.time()
    if process_timeout:
        bitstring = run_with_timeout(run_solver, timeout, solver, Q, **solver_kwargs)
    else:
        bitstring = run_solver(solver, Q, **solver_kwargs)
    elapsed = time.time() - start_time

    # The soft-timeout check only applies to the non-native solvers; for the
    # D-Wave family (or an enforced run) it is skipped.
    score_enforce = solver not in PROCESS_TIMEOUT_SOLVERS or enforce_timeout
    objective_result = _objective_or_nan(bitstring, elapsed, timeout, score_enforce, Q)

    return objective_result, elapsed, G


# ---------------------------------------------------------------------------
# Many instances
# ---------------------------------------------------------------------------
def seeds_for_size(seed: int | None, nb_instances: int) -> list:
    """Instance seeds for one problem size."""
    if seed is None:
        return [None] * nb_instances
    return [seed + idx for idx in range(nb_instances)]


def _run_single_instance(kwargs: dict) -> tuple[float, float]:
    """Run one instance; picklable entry point for the process pool.

    Only the objective and the elapsed time cross the process boundary. The graph
    does not need to: it is a pure function of ``(size, seed)``, so anything that
    wants it back regenerates it with :func:`utils.graphs.sample_instance_graph`.
    """
    objective_result, elapsed_time, _graph = run_instance(**kwargs)
    return objective_result, elapsed_time


def run_sweep(
    nb_instances_per_size: int,
    size_range: list,
    file_name: str,
    include_exact_results: bool,
    problem_type: str,
    timeout: int | None,
    solver: str,
    seed: int | None,
    num_reads: int | None,
    provider: str | None,
    backend: str | None,
    parallel_workers: int = 1,
    min_timeout_size: int | None = None,
    solver_options: dict | None = None,
    output_dir: str = "results",
) -> dict:
    """Run the sweep, writing ``<output_dir>/<file_name>`` as it goes.

    Results are flushed after every size, so an interrupted sweep still leaves
    the completed sizes on disk. The sweep stops early if more than half the
    instances at a size fail or time out -- past that point the mean beta is not
    meaningful.

    Args:
        nb_instances_per_size: instances per graph size.
        size_range: graph sizes to sweep.
        file_name: results filename inside ``output_dir``.
        include_exact_results: also store each instance's exact optimum, by
            regenerating the instances from their seeds. Only advisable at small
            sizes, where an exact solve is cheap. Requires ``seed``.
        problem_type: ``"max-cut"`` or ``"max-clique"``.
        timeout: per-instance solver timeout, in seconds.
        solver: solver name.
        seed: base seed; instance seeds walk forward from here.
        num_reads: reads/samples for the D-Wave QPU / Simulated Annealing.
        provider: hardware provider (QAOA).
        backend: backend name (QAOA) or Quandela platform (photonic).
        parallel_workers: worker processes; 1 runs inline.
        min_timeout_size: below this size, timeouts are only checked post-hoc.
        solver_options: extra keyword arguments forwarded to the solver.
        output_dir: directory for the results file (created if missing).

    Returns:
        The results mapping, keyed by problem size.

    Raises:
        ValueError: if ``include_exact_results`` is set without a ``seed``.
    """
    if include_exact_results and seed is None:
        raise ValueError(
            "include_exact_results needs a seed: the exact optima are computed by "
            "regenerating each instance from its seed, which an unseeded sweep "
            "cannot reproduce."
        )

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, file_name)

    # One entry per problem size. Run inputs live alongside in config.json, so
    # they are not duplicated into the results file.
    all_data: dict = {str(size): None for size in size_range}
    executor: ProcessPoolExecutor | None = None
    if parallel_workers and parallel_workers > 1:
        import multiprocessing as mp

        executor = ProcessPoolExecutor(
            max_workers=parallel_workers, mp_context=mp.get_context("spawn")
        )

    try:
        for size in size_range:
            instance_seeds = seeds_for_size(seed, nb_instances_per_size)
            task_args = [
                {
                    "problem_type": problem_type,
                    "size": size,
                    "solver": solver,
                    "timeout": timeout,
                    "seed": instance_seed,
                    "num_reads": num_reads,
                    "provider": provider,
                    "backend": backend,
                    "min_timeout_size": min_timeout_size,
                    "solver_options": solver_options,
                }
                for instance_seed in instance_seeds
            ]

            if executor:
                instance_results = list(executor.map(_run_single_instance, task_args))
            else:
                instance_results = [
                    _run_single_instance(task_arg) for task_arg in task_args
                ]

            result = [objective for objective, _ in instance_results]
            times = [elapsed for _, elapsed in instance_results]
            all_data[str(size)] = {"result": result, "times": times}

            if include_exact_results:
                # Regenerated from the seeds, and scored by the same
                # ``exact_optimum`` the plotter uses, so the two agree.
                all_data[str(size)]["exact-result"] = [
                    exact_optimum(problem_type, sample_instance_graph(size, s))
                    for s in instance_seeds
                ]

            with open(output_path, "w", encoding="utf-8") as handle:
                json.dump(all_data, handle)

            if seed is not None:
                seed = instance_seeds[-1] + 1

            completed = [
                value for value in result if value is not None and not math.isnan(value)
            ]
            avg_objective = float(np.mean(completed)) if completed else float("nan")
            avg_time = float(np.mean(times)) if times else float("nan")

            print(
                f"{solver} - {problem_type} - problem size: {size}, "
                f"completed {len(completed)}/{len(result)}, "
                f"average objective: {avg_objective}, "
                f"average resolution time: {avg_time:.2f}."
            )

            if len(completed) < len(result) / 2:
                print(
                    f"At least half of the instances timed out for size {size}; "
                    "skipping remaining sizes."
                )
                break
    finally:
        if executor:
            executor.shutdown(wait=True)

    return all_data


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------
def build_parser(cli_spec: dict) -> argparse.ArgumentParser:
    """Build the two-subcommand parser from a ``benchmark_cli.json`` spec."""
    parser = argparse.ArgumentParser(
        prog="python -m lib.benchmark", description=cli_spec.get("description", "")
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    commands = cli_spec.get("commands", {})

    sweep_spec = commands.get("sweep", {})
    sweep = subparsers.add_parser("sweep", help=sweep_spec.get("description"))
    sweep.add_argument(
        "--config",
        required=True,
        help="Run config JSON: a path, or a bare filename inside configs/.",
    )
    add_arguments(sweep, sweep_spec.get("arguments", []))

    run_spec = commands.get("run", {})
    single = subparsers.add_parser("run", help=run_spec.get("description"))
    add_arguments(single, run_spec.get("arguments", []))

    return parser


def run_sweep_from_config(config: dict, output_dir: str | None = None) -> str:
    """Run a sweep from a resolved config dict; returns the directory results land in.

    Content-addresses the run by a hash of its experiment identity: results
    (``results.json``) and a copy of the resolved config (``config.json``) are
    written to ``<output_dir or output.dir>/<hash>/``. Shared by
    :func:`command_sweep` (the ``benchmark_cli.json``-driven CLI) and
    :func:`lib.runner.train_and_evaluate` (the repository's shared runner), so
    both entrypoints produce the exact same results for the exact same config.
    """
    sweep = config.get("sweep", {})

    hash_dir = output_dir or run_dir(config)
    os.makedirs(hash_dir, exist_ok=True)
    with open(os.path.join(hash_dir, CONFIG_FILE), "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    print(
        f"[solver] solver={config['solver']} problem={config['problem_type']} "
        f"sizes={sweep.get('size_range')} instances={sweep.get('nb_instances_per_size')} "
        f"seed={sweep.get('seed')} hash={os.path.basename(hash_dir)} "
        f"-> {os.path.join(hash_dir, RESULTS_FILE)}"
    )

    run_sweep(
        nb_instances_per_size=sweep["nb_instances_per_size"],
        size_range=sweep["size_range"],
        file_name=RESULTS_FILE,
        include_exact_results=sweep.get("include_exact_results", False),
        problem_type=config["problem_type"],
        timeout=sweep.get("timeout"),
        solver=config["solver"],
        seed=sweep.get("seed"),
        num_reads=sweep.get("num_reads"),
        provider=config.get("provider"),
        backend=config.get("backend"),
        parallel_workers=sweep.get("parallel_workers", 1),
        min_timeout_size=sweep.get("min_timeout_size"),
        solver_options=config.get("solver_options"),
        output_dir=hash_dir,
    )
    return hash_dir


def command_sweep(cli_spec: dict, args: argparse.Namespace) -> None:
    """Run a config-driven sweep and store its results."""
    arguments = cli_spec.get("commands", {}).get("sweep", {}).get("arguments", [])
    config = load_config(args.config)
    apply_overrides(config, arguments, args)
    run_sweep_from_config(config)


def command_run(cli_spec: dict, args: argparse.Namespace) -> None:
    """Run a single instance and print its objective, beta and runtime."""
    arguments = cli_spec.get("commands", {}).get("run", {}).get("arguments", [])
    kwargs = collect_kwargs(arguments, args)

    objective, elapsed, graph = run_instance(**kwargs)

    # --exact scores against this instance's own optimum; otherwise the Q-score
    # standard's asymptotic constants are used, as in `plotter.py` with and
    # without -e.
    exact = bool(getattr(args, "exact", False))
    beta = calculate_beta(
        kwargs["problem_type"],
        graph if exact else kwargs["size"],
        objective,
        seed=derive_seed(kwargs.get("seed"), "beta"),
    )
    print(
        f"Finished problem size: {kwargs['size']}, "
        f"objective: {objective}, "
        f"{'exact ' if exact else ''}beta: {beta:.2f}, "
        f"problem time: {elapsed:.2f}."
    )


def main() -> None:
    """Parse arguments and dispatch to the selected subcommand."""
    cli_spec = load_cli_spec(_CLI_SPEC)
    args = build_parser(cli_spec).parse_args()

    if args.command == "sweep":
        command_sweep(cli_spec, args)
    else:
        command_run(cli_spec, args)


if __name__ == "__main__":
    main()
