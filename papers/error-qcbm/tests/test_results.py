"""Protect Figure 2 selection, failed/retry accounting and sampled plot sources."""

import csv
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from lib.results import (
    choose_panel_f,
    finite_curves,
    load_result_bundle,
    make_figures,
    select_runs,
    summarize_results,
)

RESULTS = Path(__file__).resolve().parents[1] / "results"


@pytest.fixture(scope="module")
def bundle():
    """Load only committed public numeric results, never experiment archives."""
    return load_result_bundle(RESULTS)


def test_inventory_preserves_three_original_failures_and_separate_retries(bundle):
    assert len(bundle["runs"]) == 263
    failed = [r for r in bundle["runs"] if r["status"] == "failed"]
    assert len(failed) == 3 and {r["seed"] for r in failed} == {8}
    assert {r["last_recorded_step"] for r in failed} == {680}
    assert {r["completed_steps"] for r in failed} == {683}
    for row in failed:
        assert len(bundle["histories"][row["run_id"]]) == 69
    original = select_runs(bundle, "belenos", "belenos_recycling")
    supplemented = select_runs(
        bundle, "belenos", "belenos_recycling", supplemented=True
    )
    assert len(original) == 19 and len(supplemented) == 20
    assert (
        next(r for r in supplemented if r["seed"] == 8)["attempt"] == "sampling_retry"
    )


def test_best5_are_selected_per_condition_by_final_sampled_kl(bundle):
    summary = summarize_results(bundle)
    assert summary["groups"]["gaussian/lossless"][
        "best5_seeds_by_final_finite_loss"
    ] == [16, 18, 2, 7, 1]
    assert summary["groups"]["gaussian/recycling"][
        "best5_seeds_by_final_finite_loss"
    ] == [6, 15, 1, 4, 14]
    assert (
        summary["comparisons"]["gaussian/recycling_vs_postselection"][
            "finite_treatment_wins"
        ]
        == 20
    )
    assert (
        summary["comparisons"]["gaussian/recycling_vs_postselection"][
            "exact_diagnostic_treatment_wins"
        ]
        == 9
    )


def test_panel_f_keeps_original_selected_seed_across_all_controls(bundle):
    assert choose_panel_f(bundle) == 6
    values = summarize_results(bundle)["panel_f"]["final_finite_kl"]
    assert values["belenos_recycling"] == pytest.approx(0.401152, abs=1e-6)
    assert values["lossless_partial"] == pytest.approx(0.103642, abs=1e-6)
    assert values["lossless_ideal"] == pytest.approx(0.222962, abs=1e-6)
    assert values["lossless_overlap084"] == pytest.approx(0.146314, abs=1e-6)


def test_financial_histogram_is_a_subset_of_stored_samples(bundle):
    h = bundle["histogram"]
    assert h["target_count"].sum() == h["displayed_qcbm_count"].sum() == 5123
    assert h["available_qcbm_count"].sum() == 39219
    assert np.all(h["displayed_qcbm_count"] <= h["available_qcbm_count"])
    assert h["target_count"].reshape(-1, 5).sum(axis=1).shape == (99,)


def test_committed_figure_hashes_identify_the_exported_images(bundle):
    for name, hashes in bundle["manifest"]["figure_sha256"].items():
        for suffix, expected in hashes.items():
            assert (
                hashlib.sha256((RESULTS / f"{name}.{suffix}").read_bytes()).hexdigest()
                == expected
            )


def test_plot_artists_retain_individual_curves_and_finite_shot_metric(bundle):
    import matplotlib.pyplot as plt

    figures = make_figures(bundle)
    try:
        gauss = figures["gaussian_fig2ab"]
        assert len(gauss.axes[1].lines) == 10
        a = finite_curves(bundle, select_runs(bundle, "gaussian", "recycling"))
        np.testing.assert_array_equal(
            gauss.axes[0].lines[2].get_ydata(), a.mean(axis=0)
        )
        e, f = figures["belenos_fig2ef_original"].axes
        assert len(e.lines) == len(f.lines) == 6
        assert not e.collections
        assert all(line.get_linestyle() == "-" for line in e.lines)
        assert all("Experiment" not in line.get_label() for line in f.lines)
        assert all("Lossy" != line.get_label() for line in f.lines)
        assert len(figures["financial_fig2cd"].axes[1].patches) == 2
        footer = "\n".join(
            text.get_text() for text in figures["financial_fig2cd"].texts
        )
        assert "Source: European Central Bank (ECB), EXR.D.USD.EUR.SP00.A" in footer
        assert (
            "Log returns and histograms calculated by the reproduction authors"
            in footer
        )
        combined = figures["figure2_combined"]
        assert len(combined.axes) == 6
        pairs = (gauss, figures["financial_fig2cd"], figures["belenos_fig2ef_original"])
        for row, pair in enumerate(pairs):
            for column, original in enumerate(pair.axes):
                merged = combined.axes[2 * row + column]
                assert len(merged.lines) == len(original.lines)
                assert len(merged.collections) == len(original.collections)
                assert len(merged.patches) == len(original.patches)
                for actual, expected in zip(merged.lines, original.lines, strict=True):
                    np.testing.assert_array_equal(
                        actual.get_xdata(), expected.get_xdata()
                    )
                    np.testing.assert_array_equal(
                        actual.get_ydata(), expected.get_ydata()
                    )
                for actual, expected in zip(
                    merged.patches, original.patches, strict=True
                ):
                    np.testing.assert_array_equal(
                        actual.get_path().vertices, expected.get_path().vertices
                    )
        combined_credit = "\n".join(
            text.get_text()
            for subfigure in combined.subfigs
            for text in subfigure.texts
        )
        assert "Source: European Central Bank (ECB)" in combined_credit
        assert "Belenos-characterized MerLin simulation" in combined_credit
    finally:
        for figure in figures.values():
            plt.close(figure)


@pytest.mark.parametrize("change", ["hash", "selection", "drop_failure"])
def test_invalid_result_bundle_fails_clearly(tmp_path, change):
    for name in (
        "figure2_manifest.json",
        "runs.csv",
        "curves.csv",
        "financial_histogram.csv",
    ):
        shutil.copyfile(RESULTS / name, tmp_path / name)
    manifest = json.loads((tmp_path / "figure2_manifest.json").read_text())
    if change == "hash":
        with (tmp_path / "curves.csv").open("a") as stream:
            stream.write("edited")
        message = "SHA256 mismatch"
    elif change == "selection":
        manifest["panels"]["f"]["seed"] = 1
        message = "seed policy"
    else:
        path = tmp_path / "runs.csv"
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        rows = [r for r in rows if r["status"] != "failed"]
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        manifest["input_sha256"]["runs.csv"] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        message = "retaining three original failures"
    (tmp_path / "figure2_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=message):
        load_result_bundle(tmp_path)


def test_shared_runtime_redraw_uses_no_random_sampling(tmp_path, monkeypatch):
    from lib.runner import train_and_evaluate

    def forbidden(*args, **kwargs):
        raise AssertionError("Result rendering must not resample counts")

    monkeypatch.setattr(np.random, "default_rng", forbidden)
    train_and_evaluate(
        {"task": "plot_results", "results": {"directory": str(RESULTS)}}, tmp_path
    )
    assert len(list(tmp_path.glob("*.png"))) == 4
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics["new_photonic_shots"] == metrics["qpu_jobs_submitted"] == 0
