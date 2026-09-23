from __future__ import annotations

import json
import sys

import pytest
from common import PROJECT_DIR, build_project_cli_parser, load_runtime_ready_config

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


def test_cli_help_exits_cleanly():
    parser, _ = build_project_cli_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--help"])
    assert exc.value.code == 0


def test_cli_overrides_parse():
    parser, _ = build_project_cli_parser()
    args = parser.parse_args(
        ["--experiment", "reservoir_instance_sweep", "--n-qubits", "10",
         "--horizons", "1,5", "--n-instances", "7", "--selection-split", "test"]
    )
    assert args.experiment == "reservoir_instance_sweep"
    assert args.n_qubits == 10
    assert args.horizons == [1, 5]
    assert args.n_instances == 7
    assert args.selection_split == "test"


def test_defaults_contain_expected_keys():
    cfg = load_runtime_ready_config()
    for key in (
        "experiment",
        "dataset",
        "evaluation",
        "quantum_reservoir",
        "feature_selection",
        "baselines",
        "controls",
        "instance_sweep",
        "photonic",
    ):
        assert key in cfg, f"Defaults missing top-level key {key!r}"


def test_defaults_match_paper_reservoir_geometry():
    """Paper: 10 qubits, n1 = 7 encoded features, n2 = 3 hidden, memory depth k = 3."""
    cfg = load_runtime_ready_config()
    quantum = cfg["quantum_reservoir"]
    assert quantum["n_qubits_total"] == 10
    assert quantum["evolution_time_tau"] == 1.0
    assert quantum["transverse_field_v"] == 1.0
    assert quantum["ridge_delta"] == 1e-8
    # n1 is set by the feature set and n2 = 10 - n1 (paper Sec. IV.D: 7 and 3).
    for variant in ("qr1", "qr2"):
        assert len(cfg["feature_selection"][f"paper_optimal_{variant}"]) == 7
    assert quantum["virtual_nodes"] == {"QR1": 1, "QR2": 2}
    assert cfg["dataset"]["n_lags"] == 3
    assert cfg["evaluation"]["n_out_of_sample"] == 245


def test_photonic_defaults_use_at_least_two_photons():
    """A single-photon linear-optical circuit is not a photonic reservoir."""
    cfg = load_runtime_ready_config()
    for name, variant in cfg["photonic"]["variants"].items():
        assert variant["n_photons"] >= 2, f"{name} uses fewer than two photons"
        assert variant["n_photons"] <= variant["n_modes"]


def test_every_experiment_choice_is_dispatchable():
    """Each --experiment choice must resolve to a runner implementation."""
    from lib import runner

    parser, _ = build_project_cli_parser()
    choices = next(
        action.choices
        for action in parser._actions  # noqa: SLF001 - argparse exposes no public view
        if action.dest == "experiment"
    )
    known = set(runner.EXPERIMENTS) | {"photonic"}
    assert set(choices) <= known, f"undispatchable choices: {set(choices) - known}"


def test_default_config_declares_a_leakage_free_selection_split():
    """The default protocol must not select models on the out-of-sample window."""
    cfg = load_runtime_ready_config()
    assert cfg["evaluation"]["selection_split"] == "validation"
    assert cfg["evaluation"]["n_validation"] > 0


def test_train_and_evaluate_writes_the_required_run_artifacts(tmp_path):
    """A minimal paper_table2 run must emit the full run-evidence bundle."""
    from lib import runner

    cfg = load_runtime_ready_config()
    cfg["evaluation"] = {
        **cfg["evaluation"], "n_out_of_sample": 12, "n_validation": 12,
        "horizons": [1], "mcs": {"size": 0.05, "reps": 50},
    }
    cfg["quantum_reservoir"] = {**cfg["quantum_reservoir"], "variants": ["QR1"]}
    cfg["baselines"] = {**cfg["baselines"], "enabled": ["har"]}
    cfg["controls"] = {**cfg["controls"], "enabled": False}
    run_dir = tmp_path / "run"

    runner.train_and_evaluate(cfg, run_dir)

    for name in ("run.log", "run_status.json", "config_snapshot.json", "metrics.json",
                 "predictions_S1.csv"):
        assert (run_dir / name).exists(), f"missing run artifact {name}"
    status = json.loads((run_dir / "run_status.json").read_text())
    assert status["status"] == "COMPLETED"
    assert status["metrics_sha256"] and status["config_sha256"]
    log = (run_dir / "run.log").read_text()
    assert log.count("RUN_STARTED") == 1
    assert log.count("RUN_COMPLETED") == 1


def test_unknown_experiment_fails_loudly(tmp_path):
    from lib import runner

    cfg = load_runtime_ready_config()
    cfg["experiment"] = "not_an_experiment"
    with pytest.raises(KeyError):
        runner.train_and_evaluate(cfg, tmp_path / "run")
