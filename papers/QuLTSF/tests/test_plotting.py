"""Figures render from a minimal comparison artifact and save to both places."""

from __future__ import annotations

import common  # noqa: F401  (puts the repository root on sys.path)

from papers.QuLTSF.lib.plotting import (
    plot_ablations,
    plot_lookback,
    plot_photonic,
    plot_table1,
    save_figure,
)

MODELS = ("qultsf", "linear", "nlinear", "dlinear")


def _row(mse, paper=None):
    return {
        "mse_mean": mse,
        "mse_std": 0.001,
        "mae_mean": mse + 0.05,
        "mae_std": 0.001,
        "trainable_parameters": 10,
        "paper": paper,
    }


def _comparison():
    horizons = (96, 720)
    table1 = [
        {
            "T": horizon,
            "rows": {
                name: _row(0.1 + 0.01 * i + horizon / 1e4, (0.2, 0.3))
                for i, name in enumerate(MODELS)
            },
            "best_baseline": "linear",
        }
        for horizon in horizons
    ]
    ablations = [
        {"T": horizon, "rows": {"identity_circuit": _row(0.1), "mlp": _row(0.12)}}
        for horizon in horizons
    ]
    lookback = [
        {
            "T": horizon,
            "L": length,
            "rows": {name: _row(0.2 - length / 1e4) for name in MODELS},
        }
        for horizon in horizons
        for length in (48, 336)
    ]
    labels = dict.fromkeys(
        ("table1", "budget", "ablations", "lookback", "merlin"), "test"
    )
    photonic = [
        {
            "T": horizon,
            "best_baseline": "linear",
            "rows": {
                "qultsf_merlin_e20": _row(0.105),
                "identity_interferometer_merlin": _row(0.103),
            },
        }
        for horizon in horizons
    ]
    return {
        "table1": table1,
        "ablations": ablations,
        "lookback": lookback,
        "labels": labels,
        "photonic": photonic,
    }


def test_every_figure_saves_to_run_and_results(tmp_path):
    comparison = _comparison()
    for name, plot in (
        ("table1", plot_table1),
        ("lookback", plot_lookback),
        ("ablations", plot_ablations),
        ("photonic", plot_photonic),
    ):
        written = save_figure(
            plot(comparison), name, tmp_path / "run", tmp_path / "results"
        )
        assert [path.exists() and path.stat().st_size > 0 for path in written] == [
            True,
            True,
        ]


def test_fair_comparison_figure_renders(tmp_path):
    from papers.QuLTSF.lib.plotting import plot_fair_comparison

    def local(name, mse):
        return {
            "system": name,
            "provenance": "local",
            "successful_runs": 3,
            "mse_mean": mse,
            "mse_std": 0.001,
            "trainable_parameters": 1000,
        }

    paper = {
        "system": "qultsf (paper Table 1)",
        "provenance": "paper-reported",
        "mse": 0.15,
    }
    table = {
        "horizons": [
            {"T": 96, "rows": [paper, local("qultsf", 0.15), local("linear", 0.17)]},
            {"T": 720, "rows": [paper, local("qultsf", 0.31), local("mlp", 0.32)]},
        ]
    }
    written = save_figure(
        plot_fair_comparison(table), "fair", tmp_path / "run", tmp_path / "results"
    )
    assert all(path.stat().st_size > 0 for path in written)
