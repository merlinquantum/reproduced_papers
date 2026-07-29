"""The CLI is declared in cli.json; these pin what that declaration buys."""

import json

import pytest
from benchmark import build_parser
from lib.config import config_hash, load_cli_spec, load_config


@pytest.fixture
def cli_spec(configs_dir):
    return load_cli_spec(configs_dir.parent / "cli.json")


@pytest.fixture
def parser(cli_spec):
    return build_parser(cli_spec)


def test_both_subcommands_exist(cli_spec):
    assert set(cli_spec["commands"]) == {"sweep", "run"}


def test_a_subcommand_is_required(parser):
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_sweep_parses_declared_overrides(parser):
    args = parser.parse_args(
        [
            "sweep",
            "--config",
            "configs/obliq_maxclique.json",
            "--sizes",
            "2,3,4",
            "--instances",
            "20",
            "--workers",
            "2",
        ]
    )
    assert args.command == "sweep"
    assert args.sweep__size_range == [2, 3, 4]
    assert args.sweep__nb_instances_per_size == 20


def test_unpassed_flags_stay_none(parser):
    """This is what keeps an unused override out of the config -- and its hash."""
    args = parser.parse_args(["sweep", "--config", "configs/obliq_maxclique.json"])
    assert args.sweep__size_range is None
    assert (
        args.sweep__include_exact_results is None
    ), "store_true must default to None, not False"


def test_overrides_only_touch_what_was_passed(parser, cli_spec):
    from lib.config import apply_overrides

    arguments = cli_spec["commands"]["sweep"]["arguments"]
    config = load_config("obliq_maxclique.json")
    before = json.loads(json.dumps(config))

    args = parser.parse_args(["sweep", "--config", "x.json"])
    apply_overrides(config, arguments, args)
    assert config == before, "nothing passed, nothing changed"
    assert config_hash(config) == config_hash(before)

    args = parser.parse_args(["sweep", "--config", "x.json", "--sizes", "2,3"])
    apply_overrides(config, arguments, args)
    assert config["sweep"]["size_range"] == [2, 3]
    assert config["solver_options"] == before["solver_options"], "unrelated keys intact"


def test_overriding_the_sweep_changes_the_hash(parser, cli_spec):
    """Why the docs tell you to override on both sides, or neither.

    Results live under a hash of the config, so a sweep run with ``--sizes`` and
    then plotted without it would look in a directory that does not exist.
    """
    from lib.config import apply_overrides

    arguments = cli_spec["commands"]["sweep"]["arguments"]
    config = load_config("obliq_maxclique.json")
    original = config_hash(config)

    args = parser.parse_args(["sweep", "--config", "x.json", "--sizes", "2,3"])
    apply_overrides(config, arguments, args)
    assert config_hash(config) != original


def test_run_requires_problem_size_and_solver(parser):
    with pytest.raises(SystemExit):
        parser.parse_args(["run", "--problem", "max-clique"])


def test_run_collects_kwargs_for_the_benchmark(parser, cli_spec):
    from lib.config import collect_kwargs

    args = parser.parse_args(
        [
            "run",
            "--problem",
            "max-clique",
            "--size",
            "6",
            "--solver",
            "obliq-static",
            "--seed",
            "101600",
            "--solver-options",
            '{"nsamples": 500}',
        ]
    )
    kwargs = collect_kwargs(cli_spec["commands"]["run"]["arguments"], args)
    assert kwargs == {
        "problem_type": "max-clique",
        "size": 6,
        "solver": "obliq-static",
        "seed": 101600,
        "solver_options": {"nsamples": 500},
    }

    # Every collected key must be a real run_instance parameter.
    import inspect

    from benchmark import run_instance

    assert set(kwargs) <= set(inspect.signature(run_instance).parameters)


def test_problem_choices_are_enforced(parser):
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "run",
                "--problem",
                "travelling-salesman",
                "--size",
                "4",
                "--solver",
                "tabu",
            ]
        )
