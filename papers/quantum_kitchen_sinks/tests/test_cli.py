from __future__ import annotations

import pytest
from common import build_project_cli_parser


def test_cli_help_exits_cleanly():
    parser, _ = build_project_cli_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--help"])
    assert exc.value.code == 0


def test_cli_known_flags_parse():
    parser, _ = build_project_cli_parser()
    args = parser.parse_args(
        [
            "--experiment", "picture_frames",
            "--circuit", "cnot2",
            "--n-qubits", "2",
            "--n-episodes", "100",
            "--sigma", "1.0",
            "--shots-per-episode", "1",
            "--n-layers", "1",
            "--classifier-kind", "logistic_regression",
            "--classifier-c", "1.0",
        ]
    )
    assert args.experiment == "picture_frames"
    assert args.circuit == "cnot2"
    assert args.n_qubits == 2
    assert args.n_episodes == 100
