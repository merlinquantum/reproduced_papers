"""``utils/make_results.py`` calls every result script, with the four sweeps."""

from __future__ import annotations

from pathlib import Path

from common import PROJECT_DIR
from QuaRK.utils.make_results import SWEEPS, build_commands

UTILS_DIR = PROJECT_DIR / "utils"


def _commands():
    return build_commands(Path("/runs/run_x"), Path("/runs"), Path("/release/run"))


def test_every_called_script_exists():
    for command in _commands():
        assert Path(command[0]).is_file(), command[0]


def test_every_figure_and_report_script_is_called():
    called = {Path(command[0]).name for command in _commands()}
    plots = {path.name for path in UTILS_DIR.glob("plot_*.py")} - {"plot_style.py"}
    reports = {
        "forecast_noise_floor.py",
        "report_statistics.py",
        "compare_with_release.py",
        "compare_photonic_with_quark.py",
        "compare_fair_baselines.py",
        "build_result_tables.py",
    }
    assert plots | reports <= called


def test_inputs_reach_the_scripts():
    arguments = {argument for command in _commands() for argument in command[1:]}
    assert {f"/runs/sweep_{name}" for name in SWEEPS} <= arguments
    assert {"/runs/run_x", "/release/run"} <= arguments


def test_tables_come_after_the_reports_they_read():
    order = [Path(command[0]).name for command in _commands()]
    assert order.index("forecast_noise_floor.py") < order.index("report_statistics.py")
    assert order.index("compare_with_release.py") < order.index(
        "compare_fair_baselines.py"
    )
    assert order.index("compare_fair_baselines.py") < order.index(
        "build_result_tables.py"
    )
