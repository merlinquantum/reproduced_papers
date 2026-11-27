from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest
from common import PROJECT_DIR, build_project_cli_parser, load_runtime_ready_config

# Optional: path to the experiment-logging contract validator
# (validate_logging.py). When set, the runner test also validates its run dir.
VALIDATOR = os.environ.get("ALPS_LOGGING_VALIDATOR", "")


def test_cli_help_exits_cleanly():
    parser, _ = build_project_cli_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--help"])
    assert exc.value.code == 0


def test_cli_parses_task_flag():
    parser, _ = build_project_cli_parser()
    args = parser.parse_args(["--task", "theory_checks"])
    assert args.task == "theory_checks"


@pytest.mark.parametrize(
    "path", sorted((PROJECT_DIR / "configs").glob("*.json")), ids=lambda p: p.name
)
def test_configs_are_json_with_description(path):
    cfg = json.loads(path.read_text())
    assert cfg.get("description")
    assert "<<" not in path.read_text()


def test_train_and_evaluate_writes_contract_artifacts(tmp_path):
    from fermi_dirac_neurons.lib import runner

    cfg = load_runtime_ready_config()
    cfg["training"]["epochs"] = 5
    cfg["training"]["eval_every"] = 2
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    runner.train_and_evaluate(cfg, run_dir)

    for name in (
        "run.log",
        "run_status.json",
        "config_snapshot.json",
        "metrics.json",
        "history.csv",
    ):
        assert (run_dir / name).exists(), name
    status = json.loads((run_dir / "run_status.json").read_text())
    assert status["status"] == "COMPLETED"
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert set(metrics["models"]) == {"quantum_tfim", "classical_im"}
    if VALIDATOR and os.path.exists(VALIDATOR):
        result = subprocess.run(
            [
                sys.executable,
                VALIDATOR,
                "--run-dir",
                str(run_dir),
                "--require-training",
                "--require-evaluation",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr


def test_unknown_task_fails(tmp_path):
    from fermi_dirac_neurons.lib import runner

    cfg = load_runtime_ready_config()
    cfg["experiment"]["task"] = "nope"
    with pytest.raises(ValueError, match="experiment.task"):
        runner.train_and_evaluate(cfg, tmp_path)


def test_non_cpu_device_fails(tmp_path):
    from fermi_dirac_neurons.lib import runner

    cfg = load_runtime_ready_config()
    cfg["device"] = "cuda:0"
    with pytest.raises(ValueError, match="device"):
        runner.train_and_evaluate(cfg, tmp_path)


def _run_extension_task(tmp_path, overlay):
    from fermi_dirac_neurons.lib import runner

    from runtime_lib.config import deep_update

    cfg = deep_update(load_runtime_ready_config(), overlay)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    runner.train_and_evaluate(cfg, run_dir)
    status = json.loads((run_dir / "run_status.json").read_text())
    assert status["status"] == "COMPLETED"
    if VALIDATOR and os.path.exists(VALIDATOR):
        result = subprocess.run(
            [
                sys.executable,
                VALIDATOR,
                "--run-dir",
                str(run_dir),
                "--require-evaluation",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr
    return json.loads((run_dir / "metrics.json").read_text())


def test_gradient_variance_task_runs(tmp_path):
    metrics = _run_extension_task(
        tmp_path,
        {
            "experiment": {"task": "gradient_variance", "id": "smoke_e10"},
            "gradient_variance": {
                "term_set": "tfim",
                "include_identity": False,
                "n_qubits": 3,
                "init_range": 0.5,
                "temperatures": {"fixed": [2.0], "sqrt_n_factors": [1.0]},
                "inputs": ["zero", "haar"],
                "parameter_positions": ["first", "last"],
                "draws": 5,
                "target_range": 2.0,
                "data_variance_check": {
                    "temperature": 2.0,
                    "parameter_position": "first",
                    "n_states": 50,
                },
            },
        },
    )
    # 2 temperatures x 2 inputs x 2 parameters x {output, loss}.
    assert len(metrics["records"]) == 16
    assert metrics["haar_data_variance_check"]["n_states"] == 50


def test_contextuality_scan_task_runs(tmp_path):
    metrics = _run_extension_task(
        tmp_path,
        {
            "experiment": {"task": "contextuality_scan", "id": "smoke_e12"},
            "contextuality": {
                "temperatures": {"low": 0.1, "high": 10.0, "count": 3},
                "invariance": {
                    "n_qubits": [3],
                    "sets_per_kind": 1,
                    "sizes": [5],
                    "locality": 2,
                    "activations": ["tanh"],
                    "epsilon": 1e-8,
                    "theta_range": 1.0,
                },
                "graded": {
                    "families": ["tfim"],
                    "n_qubits": [3],
                    "draws": 1,
                    "theta_range": 2.0,
                    "epsilons": [1e-3],
                    "activation": "tanh",
                    "graded_max_support": 300,
                },
            },
        },
    )
    assert not any(record["flag_changed_at_some_T"] for record in metrics["invariance"])
    assert len(metrics["graded"]) == 3


def test_pauli_compressibility_task_runs(tmp_path):
    metrics = _run_extension_task(
        tmp_path,
        {
            "experiment": {"task": "pauli_compressibility", "id": "smoke_e15c"},
            "pauli_compressibility": {
                "term_set": "ising_x0",
                "n_qubits": 3,
                "temperature": 2.0,
                "activation": "tanh",
                "theta_ranges": [0.5],
                "draws": 2,
                "epsilons": [1e-2, 1e-3],
            },
        },
    )
    assert metrics["kirby_love_noncontextual"] and not metrics["commuting"]
    assert len(metrics["records"]) == 2
    needed = metrics["records"][0]["paulis_needed"]
    assert needed["0.01"] <= needed["0.001"] <= metrics["total_strings"]


def test_gradient_variance_task_accepts_explicit_labels(tmp_path):
    metrics = _run_extension_task(
        tmp_path,
        {
            "experiment": {"task": "gradient_variance", "id": "smoke_e15b"},
            "gradient_variance": {
                "term_set": "explicit",
                "term_labels": ["XZI", "IYY", "ZII", "IIX", "XXI"],
                "include_identity": False,
                "n_qubits": 3,
                "init_range": 0.5,
                "temperatures": {"fixed": [2.0], "sqrt_n_factors": []},
                "inputs": ["zero"],
                "parameter_positions": ["first", "last"],
                "draws": 3,
                "target_range": 2.0,
                "data_variance_check": None,
            },
        },
    )
    assert metrics["term_labels"] == ["XZI", "IYY", "ZII", "IIX", "XXI"]
