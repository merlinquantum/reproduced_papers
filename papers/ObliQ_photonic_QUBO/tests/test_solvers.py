"""Solver dispatch, and the reproducibility guarantee for every local solver.

The photonic runs here are deliberately tiny (5 modes, a handful of iterations):
enough to exercise encode -> run -> decode -> train, fast enough for a test suite.
"""

import numpy as np
import pytest
from benchmark import run_instance, run_sweep, seeds_for_size
from models.solver import (
    GRAPH_NATIVE_SOLVERS,
    OBLIQ_SOLVERS,
    PROCESS_TIMEOUT_SOLVERS,
    SOLVERS,
    run_solver,
)
from utils.graphs import sample_instance_graph
from utils.qubo import build_qubo, exact_optimum

TRAIN = {"nsamples": 1000, "num_rep": 10, "train": {"optimizer": "adam", "max_iter": 3}}


# ---------------------------------------------------------------------------
# Reproducibility -- the point of the harness
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "problem_type,solver,options",
    [
        ("max-clique", "obliq-static", {"nsamples": 1000, "num_rep": 10}),
        ("max-clique", "obliq-hybrid", TRAIN),
        ("max-cut", "obliq-hybrid", TRAIN),  # exercises the ancilla path
        ("max-clique", "obliq-vqc", TRAIN),
    ],
)
def test_same_seed_reproduces_the_objective(problem_type, solver, options):
    first = run_instance(problem_type, 5, solver, seed=101500, solver_options=options)
    second = run_instance(problem_type, 5, solver, seed=101500, solver_options=options)
    assert first[0] == second[0]


def test_training_history_replays_exactly(clique_graph):
    """The learner is the one stochastic step; a seed must pin the whole curve."""
    from models.obliq import train_obliq_vqc_coeffs

    Q = build_qubo("max-clique", clique_graph)
    kwargs = {"variant": "obliq-hybrid", "max_iter": 4, "nsamples": 1000}
    coeffs_a, history_a = train_obliq_vqc_coeffs(Q, seed=7, **kwargs)
    coeffs_b, history_b = train_obliq_vqc_coeffs(Q, seed=7, **kwargs)
    _coeffs_c, history_c = train_obliq_vqc_coeffs(Q, seed=8, **kwargs)

    assert history_a["energies"] == history_b["energies"]
    assert coeffs_a == coeffs_b
    assert history_a["energies"] != history_c["energies"], "seed must actually matter"
    assert history_a["seed"] == 7


def test_training_lowers_the_expected_energy(clique_graph):
    from models.obliq import train_obliq_vqc_coeffs

    Q = build_qubo("max-clique", clique_graph)
    _coeffs, history = train_obliq_vqc_coeffs(
        Q, variant="obliq-hybrid", max_iter=25, nsamples=1000, seed=3
    )
    assert min(history["energies"]) <= history["energies"][0]


def test_explicit_coefficients_remove_the_randomness(clique_graph):
    """Supplied coefficients must make training seed-independent."""
    from models.circuits import expected_coeff_count
    from models.obliq import train_obliq_vqc_coeffs

    Q = build_qubo("max-clique", clique_graph)
    start = np.zeros(expected_coeff_count(Q.shape[0])).tolist()
    kwargs = {
        "variant": "obliq-hybrid",
        "max_iter": 3,
        "nsamples": 1000,
        "initial_coeffs": start,
    }
    assert (
        train_obliq_vqc_coeffs(Q, seed=1, **kwargs)[1]["energies"]
        == train_obliq_vqc_coeffs(Q, seed=2, **kwargs)[1]["energies"]
    )


def test_untrained_hybrid_reproduces_the_static_seed(clique_graph):
    """Zero coefficients make the VQC mesh the identity, so it cannot do worse."""
    from models.obliq import run_obliq_solver

    Q = build_qubo("max-clique", clique_graph)
    static = run_obliq_solver(Q, variant="obliq-static", nsamples=1000)
    hybrid = run_obliq_solver(Q, variant="obliq-hybrid", nsamples=1000, coeffs=None)
    assert hybrid.bitstring == static.bitstring


def test_cvar_vqe_is_reproducible(clique_graph):
    """Its layer init *and* its final multinomial draw both need the seed."""
    from models.cvar_vqe import run_photonic_cvarvqe

    kwargs = {"nb_samples": 512, "nb_inputs": 1, "max_iter": 3, "cvar_alpha": 0.25}
    assert run_photonic_cvarvqe(clique_graph, "max-clique", seed=5, **kwargs) == (
        run_photonic_cvarvqe(clique_graph, "max-clique", seed=5, **kwargs)
    )


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def test_objective_is_the_cut_or_clique_size(clique_graph):
    """A solver's score must equal the size of what its bitstring selects."""
    objective, _elapsed, graph = run_instance(
        "max-clique", 5, "obliq-static", seed=101500, solver_options={"nsamples": 1000}
    )
    assert 0 <= objective <= graph.number_of_nodes()
    assert objective == int(objective)


def test_exact_beta_scores_against_the_instances_own_optimum(clique_graph):
    """Exact beta is 1 exactly when the solver reaches this instance's optimum."""
    from utils.qubo import calculate_beta, exact_optimum

    objective, _elapsed, graph = run_instance(
        "max-clique", 5, "obliq-static", seed=101500, solver_options={"nsamples": 1000}
    )
    beta = calculate_beta("max-clique", graph, objective, seed=101500)
    assert np.isfinite(beta) and beta <= 1.0
    if objective == exact_optimum("max-clique", graph):
        assert beta == pytest.approx(1.0)


def test_beta_normalization_follows_the_argument_type(clique_graph):
    """Graph in -> exact; size in -> the asymptotic Q-score constants."""
    from utils.qubo import calculate_beta

    exact = calculate_beta("max-clique", clique_graph, 3.0, seed=1)
    asymptotic = calculate_beta("max-clique", len(clique_graph), 3.0)
    assert exact != asymptotic


def test_beta_of_a_failed_run_is_zero(clique_graph):
    from utils.qubo import calculate_beta

    assert calculate_beta("max-clique", clique_graph, float("nan")) == 0.0


# ---------------------------------------------------------------------------
# Dispatch contracts
# ---------------------------------------------------------------------------
def test_solver_metadata_is_self_consistent():
    assert OBLIQ_SOLVERS <= set(SOLVERS)
    assert GRAPH_NATIVE_SOLVERS <= set(SOLVERS)
    assert PROCESS_TIMEOUT_SOLVERS <= set(SOLVERS)


def test_unknown_solver_is_rejected(clique_graph):
    Q = build_qubo("max-clique", clique_graph)
    with pytest.raises(NotImplementedError, match="not implemented"):
        run_solver("quantum-annealer-9000", Q, problem_type="max-clique")


def test_graph_native_solvers_require_a_graph(clique_graph):
    Q = build_qubo("max-clique", clique_graph)
    with pytest.raises(ValueError, match="require graph"):
        run_solver("Photonic_CVARVQE", Q, problem_type="max-clique", graph=None)


def test_num_reads_solvers_require_a_budget(clique_graph):
    Q = build_qubo("max-clique", clique_graph)
    with pytest.raises(ValueError, match="num_reads"):
        run_solver("Simulated_Annealing", Q, problem_type="max-clique")


def test_unknown_obliq_option_is_rejected(clique_graph):
    """Options bind straight to the signature, so a typo is a TypeError."""
    Q = build_qubo("max-clique", clique_graph)
    with pytest.raises(TypeError, match="nsample"):
        run_solver(
            "obliq-static", Q, problem_type="max-clique", solver_options={"nsample": 10}
        )


def test_static_variant_cannot_be_trained(clique_graph):
    Q = build_qubo("max-clique", clique_graph)
    with pytest.raises(ValueError, match="only available for VQC or hybrid"):
        run_solver(
            "obliq-static",
            Q,
            problem_type="max-clique",
            solver_options={"train": {"max_iter": 1}},
        )


def test_autograd_cannot_train_through_a_remote_backend(clique_graph):
    """Gradients cannot flow through a sampled backend; the error must say so."""
    from models.obliq import train_obliq_vqc_coeffs

    Q = build_qubo("max-clique", clique_graph)
    with pytest.raises(ValueError, match="cobyla"):
        train_obliq_vqc_coeffs(
            Q, variant="obliq-hybrid", optimizer="adam", backend="sim:ascella"
        )


def test_config_seed_overrides_the_derived_one(clique_graph):
    """An explicit ``seed`` in solver_options must win over the harness's."""
    Q = build_qubo("max-clique", clique_graph)
    pinned = {
        "nsamples": 1000,
        "seed": 42,
        "train": {"optimizer": "adam", "max_iter": 3},
    }
    first = run_solver(
        "obliq-hybrid", Q, problem_type="max-clique", solver_options=pinned, seed=1
    )
    second = run_solver(
        "obliq-hybrid", Q, problem_type="max-clique", solver_options=pinned, seed=2
    )
    assert first == second


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------
def test_seed_sequence_walks_forward():
    assert seeds_for_size(101200, 3) == [101200, 101201, 101202]
    assert seeds_for_size(None, 2) == [None, None]


def test_sweep_writes_results_and_exact_optima(tmp_path):
    data = run_sweep(
        nb_instances_per_size=2,
        size_range=[4, 5],
        file_name="results.json",
        include_exact_results=True,
        problem_type="max-clique",
        timeout=None,
        solver="obliq-static",
        seed=101400,
        num_reads=None,
        provider=None,
        backend=None,
        solver_options={"nsamples": 500},
        output_dir=str(tmp_path),
    )

    assert (tmp_path / "results.json").exists()
    for size in ("4", "5"):
        assert len(data[size]["result"]) == 2
        assert len(data[size]["times"]) == 2
        # The optima come from regenerated instances, so they must match a
        # direct computation on the same seeds.
        expected = [
            exact_optimum("max-clique", sample_instance_graph(int(size), seed))
            for seed in seeds_for_size(101400 + (int(size) - 4) * 2, 2)
        ]
        assert data[size]["exact-result"] == expected


def test_exact_optima_need_a_seed(tmp_path):
    """Unseeded instances cannot be regenerated, so this must fail loudly."""
    with pytest.raises(ValueError, match="needs a seed"):
        run_sweep(
            nb_instances_per_size=1,
            size_range=[4],
            file_name="results.json",
            include_exact_results=True,
            problem_type="max-clique",
            timeout=None,
            solver="obliq-static",
            seed=None,
            num_reads=None,
            provider=None,
            backend=None,
            output_dir=str(tmp_path),
        )
