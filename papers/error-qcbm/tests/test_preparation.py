"""Exercise portable offline targets and immutable acquisition success/failure."""

import copy
import hashlib
import json

import numpy as np
import pytest
from lib.data import acquire_financial_source, prepare_data
from test_financial import financial_fixture


def gaussian_config(tmp_path):
    """Return a complete toy preparation config with no external inputs."""
    return {
        "data_root": str(tmp_path / "data"),
        "model": {"input_state": [1, 0, 1, 0]},
        "dataset": {"name": "gaussian", "means": [1, 4], "standard_deviations": [1, 1]},
        "data_preparation": {"download": False, "timeout_seconds": 1},
    }


def test_gaussian_preparation_creates_a_known_vector_without_network(
    tmp_path, monkeypatch
):
    def forbidden(*args, **kwargs):
        raise AssertionError("Offline Gaussian preparation requested a download")

    monkeypatch.setitem(prepare_data.__globals__, "_download_bytes", forbidden)
    run = tmp_path / "run"
    run.mkdir()
    prepare_data(gaussian_config(tmp_path), run)
    with np.load(tmp_path / "data/error-qcbm/gaussian/target.npz") as target:
        assert len(target["probabilities"]) == 6
        assert target["probabilities"].sum() == pytest.approx(1)
    report = json.loads((run / "data_preparation.json").read_text())
    assert report["status"] == "complete" and not report["training_started"]
    assert not report["acquisition"]["network_requested"]


def test_financial_preparation_is_offline_and_records_derived_data_credit(tmp_path):
    cfg = gaussian_config(tmp_path)
    directory = tmp_path / "data/error-qcbm/fx"
    directory.mkdir(parents=True)
    _, dataset = financial_fixture(directory)
    cfg["dataset"] = dataset
    run = tmp_path / "run"
    run.mkdir()
    prepare_data(cfg, run)
    metadata = json.loads((directory / "metadata.json").read_text())
    assert metadata["valid_rates"] == 24 and metadata["return_count"] == 23
    assert "European Central Bank" in metadata["source_credit"]
    assert "not ECB-produced" in metadata["transformation_notice"]
    assert metadata["acquisition"]["original_acquired_at_utc"] is None


@pytest.mark.parametrize("matches", [False, True])
def test_download_candidates_are_verified_before_promotion(
    tmp_path, monkeypatch, matches
):
    raw = b"KEY,TIME_PERIOD,OBS_VALUE\nEXR.D.USD.EUR.SP00.A,2004-01-02,1.2\n"
    directory = tmp_path / "fx"
    directory.mkdir()
    settings = {
        "source_file": "rates.csv",
        "source_url": "fixture://mocked-network",
        "sha256": hashlib.sha256(raw).hexdigest() if matches else "0" * 64,
    }
    monkeypatch.setitem(
        acquire_financial_source.__globals__,
        "_download_bytes",
        lambda settings, timeout: (raw, {"http_status": 200}),
    )
    report = {"status": "preparing"}
    options = {"download": True, "timeout_seconds": 1}
    if matches:
        result = acquire_financial_source(
            settings, directory, options, report, tmp_path / "report.json"
        )
        assert result.read_bytes() == raw
    else:
        with pytest.raises(ValueError, match="SHA256"):
            acquire_financial_source(
                settings, directory, options, report, tmp_path / "report.json"
            )
        assert not (directory / "rates.csv").exists()
    assert len(list((directory / "downloads").glob("*.csv"))) == 1
    assert report["acquisition"]["candidate_sha256"] == hashlib.sha256(raw).hexdigest()


def test_failed_preparation_does_not_replace_an_existing_target(tmp_path):
    cfg = gaussian_config(tmp_path)
    directory = tmp_path / "data/error-qcbm/fx"
    directory.mkdir(parents=True)
    source, dataset = financial_fixture(directory)
    cfg["dataset"] = copy.deepcopy(dataset)
    cfg["dataset"]["sha256"] = "0" * 64
    target = directory / "target.npz"
    target.write_bytes(b"preserved-target")
    run = tmp_path / "run"
    run.mkdir()
    with pytest.raises(ValueError, match="SHA256"):
        prepare_data(cfg, run)
    assert target.read_bytes() == b"preserved-target"
    assert source.exists()
    assert json.loads((run / "data_preparation.json").read_text())["status"] == "failed"


def test_download_never_overwrites_a_different_local_source(tmp_path, monkeypatch):
    source = tmp_path / "rates.csv"
    source.write_bytes(b"existing-modified-source")
    raw = b"new-valid-candidate"
    settings = {
        "source_file": "rates.csv",
        "sha256": hashlib.sha256(raw).hexdigest(),
        "source_url": "fixture://mocked-network",
    }
    monkeypatch.setitem(
        acquire_financial_source.__globals__,
        "_download_bytes",
        lambda settings, timeout: (raw, {}),
    )
    with pytest.raises(ValueError, match="refusing to overwrite"):
        acquire_financial_source(
            settings,
            tmp_path,
            {"download": True, "timeout_seconds": 1},
            {},
            tmp_path / "report.json",
        )
    assert source.read_bytes() == b"existing-modified-source"


def test_network_failure_is_recorded_without_synthetic_replacement(
    tmp_path, monkeypatch
):
    cfg = gaussian_config(tmp_path)
    directory = tmp_path / "data/error-qcbm/fx"
    directory.mkdir(parents=True)
    _, dataset = financial_fixture(directory)
    cfg["dataset"] = dataset
    cfg["data_preparation"]["download"] = True

    def unavailable(settings, timeout):
        raise TimeoutError("mocked explicit download timeout")

    monkeypatch.setitem(
        acquire_financial_source.__globals__, "_download_bytes", unavailable
    )
    run = tmp_path / "run"
    run.mkdir()
    with pytest.raises(TimeoutError):
        prepare_data(cfg, run)
    report = json.loads((run / "data_preparation.json").read_text())
    assert report["status"] == "failed" and report["error_type"] == "TimeoutError"
    assert not (directory / "target.npz").exists()
