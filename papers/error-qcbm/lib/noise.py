"""Translate a frozen Belenos characterization into explicit local ablations."""

import hashlib
import json
import math
import re
from numbers import Real
from pathlib import Path

from runtime_lib.data_paths import paper_data_dir

PUBLIC_PROFILE_SCHEMA = "error_qcbm.noise_profile.v1"
OMITTED_NOISE = [
    "multi-photon emission (g2): preserve four-photon loss ablation",
    "dark counts: detection window unknown",
    "mode-dependent transmission: calibration convention unverified",
    "phase errors, drift and correlated noise: unavailable characterization",
]


def _profiles(survival, hom):
    return {
        name: {
            "name": name,
            "transmittance": transmittance,
            "indistinguishability": indistinguishability,
            "g2": 0.0,
            "detector": "threshold",
        }
        for name, transmittance, indistinguishability in [
            ("ideal", 1.0, 1.0),
            ("paper_loss", 0.2, 1.0),
            ("belenos_loss", survival, 1.0),
            ("belenos_partial", survival, hom),
        ]
    }


def _public_profiles(snapshot, raw):
    if snapshot["schema"] != PUBLIC_PROFILE_SCHEMA:
        raise ValueError("Unsupported public noise profile schema")
    if snapshot["platform"] != "qpu:belenos":
        raise ValueError("Expected a Belenos noise profile")
    if not isinstance(snapshot["profile_id"], str) or not snapshot["profile_id"]:
        raise ValueError("Public profile requires a nonempty profile_id")
    parameters = snapshot["parameters"]
    if set(parameters) != {"transmittance_percent", "hom_percent"}:
        raise ValueError("Public noise profile requires percent transmission and HOM")
    for name, value in parameters.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(value)
            or not 0 <= value <= 100
        ):
            raise ValueError(f"Invalid public profile percentage: {name}")
    if parameters["transmittance_percent"] == 0:
        raise ValueError("Public profile transmission must be positive")
    if snapshot["assumptions"] != {
        "brightness": 1.0,
        "g2": 0.0,
        "detector": "threshold",
    }:
        raise ValueError(
            "Public profile supports brightness=1, g2=0 and threshold detection"
        )
    if any(
        isinstance(snapshot["assumptions"][key], bool) for key in ("brightness", "g2")
    ):
        raise ValueError("Public profile assumptions require numeric brightness and g2")
    source = snapshot["source"]
    if source["kind"] != "selected_fields_from_read_only_hub_snapshot":
        raise ValueError("Unsupported public noise profile source kind")
    if not re.fullmatch(r"[0-9a-f]{64}", source["snapshot_sha256"]):
        raise ValueError("Public profile requires the original snapshot SHA256")
    if source["field_mapping"] != {
        "transmittance_percent": "processor.perfs.Transmittance (%)",
        "hom_percent": "processor.perfs.HOM (%)",
    }:
        raise ValueError("Public profile source-field mapping is unsupported")
    survival = parameters["transmittance_percent"] / 100
    hom = parameters["hom_percent"] / 100
    provenance = {
        "schema": PUBLIC_PROFILE_SCHEMA,
        "profile_id": snapshot["profile_id"],
        "profile_sha256": hashlib.sha256(raw).hexdigest(),
        "snapshot_sha256": source["snapshot_sha256"],
        "queried_at_utc": source["queried_at_utc"],
        "calibration_timestamp": source["calibration_timestamp"],
        "platform": snapshot["platform"],
        "source_fields": source["field_mapping"],
        "reported_percentages": parameters,
        "model_values": {"transmittance": survival, "indistinguishability": hom},
        "assumptions": snapshot["assumptions"],
        "omitted": OMITTED_NOISE.copy(),
        "scope": "Frozen selected Belenos characteristics in a local global-loss/overlap model; not a current device calibration or QPU run",
    }
    return _profiles(survival, hom), provenance


def load_profiles(snapshot_path: Path) -> tuple[dict[str, dict], dict]:
    """Load the two requested noise models and paper/ideal controls.

    Parameters
    ----------
    snapshot_path : pathlib.Path
        Explicit minimal public profile or legacy Belenos response wrapper.
        No authentication or network request is performed.

    Returns
    -------
    tuple
        Profile dictionaries and characterization provenance.

    Raises
    ------
    ValueError
        If the snapshot does not identify Belenos or lacks physical values.
    """
    raw = snapshot_path.read_bytes()
    snapshot = json.loads(raw)
    if "schema" in snapshot:
        return _public_profiles(snapshot, raw)
    if snapshot["platform"] != "qpu:belenos" or snapshot["http_status"] != 200:
        raise ValueError("Expected a successful Belenos characterization")
    performance = snapshot["processor"]["perfs"]
    survival = float(performance["Transmittance (%)"]) / 100
    hom = float(performance["HOM (%)"]) / 100
    if not 0 < survival <= 1 or not 0 <= hom <= 1:
        raise ValueError("Invalid transmission or HOM percentage")
    profiles = _profiles(survival, hom)
    provenance = {
        "snapshot_sha256": hashlib.sha256(raw).hexdigest(),
        "queried_at_utc": snapshot["queried_at_utc"],
        "calibration_timestamp": snapshot["calibration_timestamp"],
        "platform": snapshot["platform"],
        "measured_g2_percent_not_applied": performance["g2 (%)"],
        "clock_mhz_not_a_local_runtime_prediction": performance["Clock (MHz)"],
        "omitted": [
            "multi-photon emission (g2): preserve four-photon loss ablation",
            "dark counts: detection window unknown",
            "mode-dependent transmission: calibration convention unverified",
            "phase errors, drift and correlated noise: unavailable characterization",
        ],
        "scope": "Belenos-calibrated global loss and partial-indistinguishability model, not a digital twin",
    }
    return profiles, provenance


def load_configured_profiles(
    cfg: dict, snapshot_file: str | None = None
) -> tuple[dict, dict]:
    """Resolve exactly one public-profile or legacy-snapshot input.

    Parameters
    ----------
    cfg : dict
        Config with ``noise_profile`` relative to the paper directory, or legacy
        ``noise_snapshot`` under the shared data root's ``error-qcbm/hub``.
    snapshot_file : str or None
        Legacy native/phase-one task field. Default value is None.

    Returns
    -------
    tuple
        Physical profiles and source provenance. Legacy provenance remains
        unchanged so existing explicitly configured checkpoints stay compatible.

    Raises
    ------
    ValueError
        If sources are missing, ambiguous or the public schema is absent.
    FileNotFoundError
        If the explicitly chosen input is missing. There is no fallback.
    """
    public = cfg.get("noise_profile")
    legacy = cfg.get("noise_snapshot")
    if sum(value is not None for value in (public, legacy, snapshot_file)) != 1:
        raise ValueError(
            "Specify exactly one noise_profile or legacy noise_snapshot/snapshot_file"
        )
    if public is not None:
        path = Path(public)
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[1] / path
        raw = path.read_bytes()
        payload = json.loads(raw)
        if payload.get("schema") != PUBLIC_PROFILE_SCHEMA:
            raise ValueError("noise_profile requires the public profile schema")
        return _public_profiles(payload, raw)
    else:
        path = (
            paper_data_dir(
                "error-qcbm", data_root=cfg["data_root"], ensure_exists=False
            )
            / "hub"
            / (legacy if legacy is not None else snapshot_file)
        )
    return load_profiles(path)
