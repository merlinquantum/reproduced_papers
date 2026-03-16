from __future__ import annotations

import pytest
from utils import figure_5, figure_architecture_grid, figure_tau_alpha_grid


@pytest.mark.parametrize(
    "module, argv",
    [
        (
            figure_5,
            ["--config", "configs/train_eval_circles.json", "--previous_run", "run"],
        ),
        (
            figure_architecture_grid,
            [
                "--config",
                "configs/design_benchmark_circles.json",
                "--previous_run",
                "run",
            ],
        ),
        (
            figure_tau_alpha_grid,
            [
                "--config",
                "configs/tau_alpha_benchmark_circles.json",
                "--previous_run",
                "run",
            ],
        ),
    ],
)
def test_figure_cli_rejects_conflicting_args(monkeypatch, module, argv):
    monkeypatch.setattr(module.sys, "argv", ["script.py", *argv])
    with pytest.raises(SystemExit):
        module.main()
