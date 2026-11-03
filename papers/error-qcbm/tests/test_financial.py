"""Check the source direction, return definition and fixed financial bins."""

import hashlib

import numpy as np
import pytest
from lib.data import collision_free_states, financial_target


def financial_fixture(directory):
    source = directory / "rates.csv"
    rows = ["KEY,TIME_PERIOD,OBS_VALUE"]
    for i in range(1, 25):
        rate = np.exp(0.01 * np.sin(i))
        rows.append(f"EXR.D.USD.EUR.SP00.A,2004-01-{i:02d},{rate:.14f}")
    rows.append("EXR.D.USD.EUR.SP00.A,2004-01-25,")
    source.write_text("\n".join(rows), encoding="utf-8")
    settings = {
        "name": "fx",
        "source_file": source.name,
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "start_date": "2004-01-01",
        "end_date": "2004-01-31",
        "binning": "equal_width_observed_range",
        "source_url": "fixture://known-rates",
    }
    return source, settings


def test_financial_log_returns_missing_rows_and_complete_histogram(tmp_path):
    source, settings = financial_fixture(tmp_path)
    states = collision_free_states(6, 2)
    probabilities, arrays, metadata = financial_target(source, settings, states)
    expected = np.diff(0.01 * np.sin(np.arange(1, 25)))
    assert arrays["returns"] == pytest.approx(expected, abs=1e-13)
    assert arrays["histogram_counts"].sum() == len(expected)
    assert len(arrays["bin_edges"]) == len(states) + 1
    assert probabilities.sum() == pytest.approx(1)
    assert metadata["missing_rate_rows_dropped"] == 1
    assert metadata["observed_period"] == ["2004-01-01", "2004-01-24"]
    assert (
        metadata["source_credit"] == "European Central Bank (ECB), EXR.D.USD.EUR.SP00.A"
    )
    assert metadata["source_terms_url"].startswith("https://www.ecb.europa.eu/")
    assert "not ECB-produced derived data" in metadata["transformation_notice"]
    assert np.array_equal(arrays["states"], states)
    settings["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA256"):
        financial_target(source, settings, states)


def test_financial_duplicate_dates_are_not_silently_merged(tmp_path):
    source, settings = financial_fixture(tmp_path)
    with source.open("a", encoding="utf-8") as stream:
        stream.write("\nEXR.D.USD.EUR.SP00.A,2004-01-02,1.5\n")
    settings["sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="unique"):
        financial_target(source, settings, collision_free_states(6, 2))
