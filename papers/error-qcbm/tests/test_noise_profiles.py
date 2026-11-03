"""Check explicit public profiles, units, legacy signatures and missing inputs."""

import copy
import hashlib
import json
from pathlib import Path

import pytest
from lib.noise import PUBLIC_PROFILE_SCHEMA, load_configured_profiles, load_profiles

PROJECT = Path(__file__).resolve().parents[1]
PROFILE = PROJECT / "configs/noise/belenos_20261017.json"


def legacy_fixture():
    """Return a synthetic legacy response, never an account-derived Hub reply."""
    return {
        "platform": "qpu:belenos",
        "http_status": 200,
        "queried_at_utc": "2026-01-01T00:00:00+00:00",
        "calibration_timestamp": None,
        "processor": {
            "perfs": {
                "Transmittance (%)": 2.8,
                "HOM (%)": 72.9,
                "g2 (%)": 1.1,
                "Clock (MHz)": 4.94,
            }
        },
    }


def test_bundled_public_profile_has_the_same_physical_mapping_as_legacy(tmp_path):
    path = tmp_path / "synthetic_legacy.json"
    path.write_text(json.dumps(legacy_fixture()))
    old, legacy = load_profiles(path)
    new, provenance = load_profiles(PROFILE)
    assert old == new
    assert new["belenos_partial"]["transmittance"] == pytest.approx(0.028)
    assert new["belenos_partial"]["indistinguishability"] == pytest.approx(0.729)
    assert new["belenos_loss"]["indistinguishability"] == 1
    assert provenance["schema"] == PUBLIC_PROFILE_SCHEMA
    assert (
        provenance["profile_sha256"] == hashlib.sha256(PROFILE.read_bytes()).hexdigest()
    )
    assert legacy["snapshot_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert "schema" not in legacy and "profile_sha256" not in legacy
    assert legacy["measured_g2_percent_not_applied"] == 1.1


def test_public_profile_resolution_requires_no_data_directory(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError(
            "Public profile must not inspect or create the private data root"
        )

    namespace = load_configured_profiles.__globals__
    monkeypatch.setitem(namespace, "paper_data_dir", forbidden)
    cfg = {
        "noise_profile": "configs/noise/belenos_20261017.json",
        "data_root": str(tmp_path / "absent"),
    }
    _, provenance = load_configured_profiles(cfg)
    assert provenance["profile_id"] == "belenos_20261017"
    assert not (tmp_path / "absent").exists()


@pytest.mark.parametrize(
    "change",
    ["schema", "units", "range", "nan", "bool", "brightness", "g2", "source_hash"],
)
def test_invalid_public_profiles_fail_without_defaults(tmp_path, change):
    profile = copy.deepcopy(json.loads(PROFILE.read_text()))
    if change == "schema":
        profile["schema"] = "unsupported"
    elif change == "units":
        profile["parameters"]["hom_fraction"] = (
            profile["parameters"].pop("hom_percent") / 100
        )
    elif change in ("range", "nan", "bool"):
        profile["parameters"]["hom_percent"] = {
            "range": 101,
            "nan": float("nan"),
            "bool": True,
        }[change]
    elif change in ("brightness", "g2"):
        profile["assumptions"][change] = 0.5
    else:
        profile["source"]["snapshot_sha256"] = "not-a-hash"
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(profile))
    with pytest.raises(ValueError):
        load_profiles(path)


def test_missing_or_ambiguous_sources_do_not_fall_back_to_bundled_values(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_configured_profiles({"noise_profile": str(tmp_path / "absent.json")})
    with pytest.raises(ValueError, match="exactly one"):
        load_configured_profiles({})
    with pytest.raises(ValueError, match="exactly one"):
        load_configured_profiles(
            {"noise_profile": str(PROFILE), "noise_snapshot": "private.json"}
        )
    with pytest.raises(FileNotFoundError):
        load_configured_profiles(
            {"noise_snapshot": "absent.json", "data_root": str(tmp_path)}
        )


def test_explicit_legacy_route_keeps_original_provenance_exactly(tmp_path):
    path = tmp_path / "error-qcbm/hub/fixture.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(legacy_fixture()))
    expected = load_profiles(path)
    assert (
        load_configured_profiles(
            {"noise_snapshot": "fixture.json", "data_root": str(tmp_path)}
        )
        == expected
    )
    assert (
        load_configured_profiles({"data_root": str(tmp_path)}, "fixture.json")
        == expected
    )


def test_belenos_training_configs_explicitly_use_the_bundled_profile():
    for name in (
        "belenos",
        "belenos_smoke",
        "belenos_vectorized",
        "belenos_loss_only",
        "belenos_overlap084",
        "lossless_overlap084",
    ):
        cfg = json.loads((PROJECT / f"configs/{name}.json").read_text())
        assert cfg["noise_profile"] == "configs/noise/belenos_20261017.json"
        assert "noise_snapshot" not in cfg
