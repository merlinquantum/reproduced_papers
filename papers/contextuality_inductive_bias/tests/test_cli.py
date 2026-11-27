from __future__ import annotations

import json

import pytest
from common import PROJECT_DIR, build_project_cli_parser, load_runtime_ready_config


def test_cli_help_exits_cleanly():
    parser, _ = build_project_cli_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--help"])
    assert exc.value.code == 0


def test_cli_parses_paper_flags():
    parser, _ = build_project_cli_parser()
    args = parser.parse_args(
        [
            "--model",
            "surrogate",
            "--spectrum",
            "2",
            "--protocol",
            "code",
            "--steps",
            "3",
            "--l2",
            "1e-3",
        ]
    )
    assert args.model == "surrogate" and args.spectrum == 2 and args.protocol == "code"


@pytest.mark.parametrize(
    "path", sorted((PROJECT_DIR / "configs").glob("*.json")), ids=lambda p: p.name
)
def test_configs_are_json_with_description(path):
    cfg = json.loads(path.read_text())
    assert cfg.get("description")


def test_train_and_evaluate_writes_contract_artifacts(tmp_path):
    from contextuality_inductive_bias.lib import runner

    cfg = load_runtime_ready_config()
    cfg["data"]["test"]["n"] = 100
    cfg["model"]["layers"] = 1
    cfg["training"]["steps"] = 3
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    runner.train_and_evaluate(cfg, run_dir)

    status = json.loads((run_dir / "run_status.json").read_text())
    assert status["status"] == "COMPLETED"
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["n_parameters"] == 3 * 9 * 2 + 1
    for name in (
        "run.log",
        "config_snapshot.json",
        "history.csv",
        "model_state.pt",
        "predictions.npz",
    ):
        assert (run_dir / name).exists(), name


def test_theory_task_runs(tmp_path):
    from contextuality_inductive_bias.lib import runner

    cfg = load_runtime_ready_config()
    cfg["experiment"]["task"] = "theory_lp"
    runner.train_and_evaluate(cfg, tmp_path)
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics["lemma_lhs_max"] == pytest.approx(2.5)


def test_unknown_task_fails(tmp_path):
    from contextuality_inductive_bias.lib import runner

    cfg = load_runtime_ready_config()
    cfg["experiment"]["task"] = "nope"
    with pytest.raises(ValueError, match="experiment.task"):
        runner.train_and_evaluate(cfg, tmp_path)
