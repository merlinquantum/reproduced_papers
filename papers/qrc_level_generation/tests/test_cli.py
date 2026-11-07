from __future__ import annotations

import pytest
from common import build_project_cli_parser


def test_cli_help_exits_cleanly():
    parser, _ = build_project_cli_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--help"])
    assert exc.value.code == 0


def test_cli_accepts_known_flags():
    parser, _ = build_project_cli_parser()
    ns = parser.parse_args(
        [
            "--n-qubits",
            "5",
            "--epochs",
            "3",
            "--temperatures",
            "1,2",
            "--n-samples",
            "4",
            "--backend",
            "qubit",
        ]
    )
    assert ns.n_qubits == 5
    assert ns.epochs == 3
    assert ns.temperatures == "1,2"
    assert ns.n_samples == 4
    assert ns.backend == "qubit"
