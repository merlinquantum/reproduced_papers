"""Run configs: loading, hashing, and the hashes the shipped results depend on."""

import json
import os

import pytest

from lib.config import (
    RESULTS_FILE,
    canonical_config,
    config_hash,
    load_config,
    run_dir,
)

#: The hash each shipped config must keep producing. A config edit that changes
#: one of these silently orphans its ``results/<hash>/`` directory -- the figures
#: and the notebook stop finding their data with only a "no results found" print.
#: Regenerate deliberately if a config genuinely changes.
EXPECTED_HASHES = {
    "cvarvqe_maxclique.json": "2927be755cf7",
    "cvarvqe_maxcut.json": "e5b889499a30",
    "obliq_maxclique.json": "33256fcdf407",
    "obliq_maxcut.json": "ee150155034a",
    "obliqstatic_maxclique.json": "15803899258d",
    "obliqstatic_maxcut.json": "fde29f76db4e",
    "obliqvqc_maxclique.json": "714661f4f072",
    "obliqvqc_maxcut.json": "7696e8899378",
    "qaoa_maxclique.json": "85c3c3034e63",
    "qaoa_maxcut.json": "7582691d649c",
    "sa_maxcut.json": "386479106055",
    "tabu_maxcut.json": "cb1f7390ca7b",
}


@pytest.mark.parametrize("name,expected", sorted(EXPECTED_HASHES.items()))
def test_shipped_config_hashes_are_stable(name, expected):
    assert config_hash(load_config(name)) == expected


def test_every_shipped_config_is_covered(configs_dir):
    """A new config must be added to EXPECTED_HASHES, not silently skipped."""
    on_disk = {path.name for path in configs_dir.glob("*.json")}
    assert on_disk == set(EXPECTED_HASHES)


@pytest.mark.parametrize("name", sorted(EXPECTED_HASHES))
def test_stored_results_are_well_formed(name, configs_dir):
    """Validate any results a config *does* resolve to.

    Skipped when a config has not been swept yet -- a fresh checkout, or a config
    whose sweep settings changed since its last run. What this catches is a results
    file that exists but disagrees with its config: missing sizes, or objective and
    timing lists of different lengths.
    """
    config = load_config(name)
    path = configs_dir.parent / run_dir(config) / RESULTS_FILE
    if not path.exists():
        pytest.skip(f"no sweep results for {name} (run benchmark.py sweep first)")

    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    assert data, f"empty results for {name}"
    assert set(data) == {str(s) for s in config["sweep"]["size_range"]}
    for size, entry in data.items():
        if entry is None:
            continue  # a size the sweep stopped before reaching
        assert len(entry["result"]) == len(entry["times"]), size


def test_execution_only_fields_do_not_change_the_hash():
    """Re-running with more workers or a new plot label must reuse the folder."""
    base = load_config("obliq_maxclique.json")
    variant = json.loads(json.dumps(base))
    variant["name"] = "A Different Label"
    variant["sweep"]["parallel_workers"] = 1
    variant["output"] = {"dir": "somewhere_else"}
    assert config_hash(variant) == config_hash(base)


def test_experiment_fields_do_change_the_hash():
    base = load_config("obliq_maxclique.json")
    for path, value in (
        ("solver", "obliq-static"),
        ("problem_type", "max-cut"),
    ):
        variant = json.loads(json.dumps(base))
        variant[path] = value
        assert config_hash(variant) != config_hash(base), path

    variant = json.loads(json.dumps(base))
    variant["sweep"]["seed"] = 1
    assert config_hash(variant) != config_hash(base)

    variant = json.loads(json.dumps(base))
    variant["solver_options"]["num_rep"] = 3
    assert config_hash(variant) != config_hash(base)


def test_canonical_config_strips_only_the_ignored_fields():
    base = load_config("obliq_maxclique.json")
    canonical = canonical_config(base)
    assert "name" not in canonical
    assert "parallel_workers" not in canonical["sweep"]
    assert canonical["solver"] == base["solver"]
    assert base["name"], "the original must not be mutated"


def test_paths_are_built_from_the_output_dir():
    config = load_config("obliq_maxclique.json")
    assert run_dir(config).replace(os.sep, "/") == f"results/{config_hash(config)}"

    config["output"] = {"dir": "elsewhere"}
    assert run_dir(config).startswith("elsewhere")


def test_load_config_accepts_a_bare_filename_or_a_path():
    assert load_config("obliq_maxclique.json") == load_config(
        "configs/obliq_maxclique.json"
    )


def test_load_config_reports_a_missing_file():
    with pytest.raises(FileNotFoundError, match=r"nope\.json"):
        load_config("nope.json")


#: Knobs that mean the same thing across solvers and must therefore agree.
SHARED_HYPERPARAMETERS = {
    "shots": 5000,
    "iterations": 100,
    "learning_rate": 0.05,
}


@pytest.mark.parametrize("name", sorted(EXPECTED_HASHES))
def test_hyperparameters_are_comparable_across_solvers(name):
    """Every solver gets the same shot budget, iteration count and learning rate.

    Solver-intrinsic knobs (`num_rep`, `cvar_alpha`, `nb_inputs`, `reps`) have no
    counterpart elsewhere and are deliberately not constrained here.
    """
    config = load_config(name)
    opts = config.get("solver_options", {})
    sweep = config["sweep"]
    solver = config["solver"]
    shots = SHARED_HYPERPARAMETERS["shots"]
    iterations = SHARED_HYPERPARAMETERS["iterations"]
    rate = SHARED_HYPERPARAMETERS["learning_rate"]

    assert sweep["size_range"] == [2, 3, 4, 5, 6, 7, 8]
    assert sweep["nb_instances_per_size"] == 100
    assert sweep["seed"] == 101200
    assert sweep["timeout"] == 300

    if solver.startswith("obliq-"):
        assert opts["nsamples"] == shots
        train = opts.get("train")
        if train:  # obliq-static has nothing to train
            assert train["optimizer"] == "adam"
            assert train["max_iter"] == iterations
            assert train["learning_rate"] == rate
    elif solver == "Photonic_CVARVQE":
        assert opts["nb_samples"] == shots
        assert opts["optimizer"] == "adam"
        assert opts["max_iter"] == iterations
        assert opts["learning_rate"] == rate
        assert "seed" not in opts, "a pinned seed defeats per-instance seeding"
    elif solver == "QAOA":
        assert opts["number_of_shots"] == shots
        assert opts["maxiter"] == iterations
    elif solver in ("Simulated_Annealing", "Advantage_system4.1"):
        assert sweep["num_reads"] == shots
    elif solver == "tabu":
        # A tabu "read" is a full local search, not a shot, so it is not on the
        # shared budget -- see models/dwave.py::run_tabu.
        assert "num_reads" not in sweep
