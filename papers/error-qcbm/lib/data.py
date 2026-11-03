"""Target distributions, immutable financial sources and explicit data preparation."""

import csv
import hashlib
import json
from datetime import datetime, timezone
from itertools import combinations
from math import comb
from pathlib import Path
from urllib.parse import parse_qs, urlsplit
from urllib.request import Request, urlopen

import numpy as np

from runtime_lib.data_paths import paper_data_dir

from .artifacts import atomic_replace, write_json


def collision_free_states(modes: int, photons: int) -> tuple[tuple[int, ...], ...]:
    """Enumerate the fixed collision-free output basis in author bin order.

    Parameters
    ----------
    modes : int
        Positive number of optical modes.
    photons : int
        Photon number, between zero and ``modes`` inclusive.

    Returns
    -------
    tuple of tuple of int
        Binary occupations in ascending lexicographic order. This is the same
        ordering as the authors' base-``modes + 1`` integer key.

    Raises
    ------
    ValueError
        If the mode or photon counts are invalid.
    """
    if isinstance(modes, bool) or not isinstance(modes, int) or modes < 1:
        raise ValueError("modes must be a positive integer")
    if (
        isinstance(photons, bool)
        or not isinstance(photons, int)
        or not 0 <= photons <= modes
    ):
        raise ValueError("photons must be an integer between zero and modes")
    states = []
    for occupied_modes in combinations(range(modes), photons):
        state = [0] * modes
        for mode in occupied_modes:
            state[mode] = 1
        states.append(tuple(state))
    result = tuple(sorted(states))
    if len(result) != comb(modes, photons):
        raise RuntimeError("Collision-free basis has the wrong cardinality")
    return result


def gaussian_target(
    num_bins: int = 495,
    means: tuple[float, float] | None = None,
    standard_deviations: tuple[float, float] | None = None,
) -> np.ndarray:
    """Evaluate and normalize the authors' two-Gaussian synthetic target.

    Parameters
    ----------
    num_bins : int, optional
        Number of integer bin coordinates starting at zero. Default is 495,
        the collision-free dimension for 12 modes and four photons.
    means : tuple of float or None, optional
        Means of the two Gaussian densities. Default is None, selecting the
        authors' ``(2 * num_bins // 7, 5 * num_bins // 7)``; these are 141 and
        353 for the paper's 495-bin experiment.
    standard_deviations : tuple of float or None, optional
        Positive widths of the two densities. Default is None, selecting
        ``num_bins // 8`` for each; this is 61 for the paper's experiment.

    Returns
    -------
    numpy.ndarray
        Float64 probabilities, in the fixed output-bin order. The continuous
        densities are evaluated at integer coordinates and then normalized;
        this is not an integral over finite-width histogram bins.

    Raises
    ------
    ValueError
        If the bin count, means, or widths are invalid, or all densities
        underflow to zero at the requested coordinates.
    """
    if isinstance(num_bins, bool) or not isinstance(num_bins, int) or num_bins < 1:
        raise ValueError("num_bins must be a positive integer")
    mean_values = np.asarray(
        (2 * num_bins // 7, 5 * num_bins // 7) if means is None else means,
        dtype=np.float64,
    )
    widths = np.asarray(
        (num_bins // 8, num_bins // 8)
        if standard_deviations is None
        else standard_deviations,
        dtype=np.float64,
    )
    if mean_values.shape != (2,) or not np.isfinite(mean_values).all():
        raise ValueError("means must contain exactly two finite values")
    if widths.shape != (2,) or not np.isfinite(widths).all() or (widths <= 0).any():
        raise ValueError("standard_deviations must contain two positive finite values")
    coordinates = np.arange(num_bins, dtype=np.float64)[:, None]
    densities = np.exp(-0.5 * ((coordinates - mean_values) / widths) ** 2) / (
        widths * np.sqrt(2.0 * np.pi)
    )
    target = densities.sum(axis=1)
    mass = float(target.sum())
    if mass <= 0 or not np.isfinite(mass):
        raise ValueError("Gaussian target has no finite positive probability mass")
    return target / mass


ECB_COPYRIGHT_URL = (
    "https://www.ecb.europa.eu/services/using-our-site/disclaimer/html/index.en.html"
)


def financial_target(source: Path, settings: dict, states: tuple) -> tuple:
    """Build an empirical target without interpolation or synthetic observations.

    Parameters
    ----------
    source : pathlib.Path
        Frozen ECB SDMX CSV containing USD per EUR reference rates.
    settings : dict
        Required SHA256, start_date, end_date, source_url and binning policy.
    states : tuple
        Collision-free states in the fixed author bin order.

    Returns
    -------
    tuple
        Probability vector, arrays for a target artifact, and provenance metadata.
    """
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != settings["sha256"]:
        raise ValueError(
            "Financial source SHA256 differs from the frozen configuration"
        )
    if settings["binning"] != "equal_width_observed_range":
        raise ValueError("Unsupported financial binning policy")
    observations = {}
    missing = 0
    with source.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["KEY"] != "EXR.D.USD.EUR.SP00.A":
                raise ValueError("Expected daily ECB USD-per-EUR observations")
            day = row["TIME_PERIOD"]
            if not settings["start_date"] <= day <= settings["end_date"]:
                raise ValueError(
                    "Source contains an observation outside the frozen period"
                )
            if not row["OBS_VALUE"]:
                missing += 1
                continue
            rate = float(row["OBS_VALUE"])
            if not np.isfinite(rate) or rate <= 0 or day in observations:
                raise ValueError(
                    "Financial rates must be positive, finite and unique by date"
                )
            observations[day] = rate
    dates = sorted(observations)
    if len(dates) < 3:
        raise ValueError("Financial data needs at least three valid daily observations")
    rates = np.array([observations[day] for day in dates])
    returns = np.diff(np.log(rates))
    if np.ptp(returns) <= 0:
        raise ValueError("Financial returns have no nonzero range")
    counts, edges = np.histogram(returns, bins=len(states))
    probabilities = counts / counts.sum()
    arrays = {
        "probabilities": probabilities,
        "states": np.asarray(states),
        "bin_edges": edges,
        "coordinates": (edges[:-1] + edges[1:]) / 2,
        "histogram_counts": counts,
        "returns": returns,
        "return_dates": np.asarray(dates[1:]),
        "rates": rates,
        "rate_dates": np.asarray(dates),
    }
    metadata = {
        "dataset": "ECB daily EUR/USD log returns",
        "source_credit": "European Central Bank (ECB), EXR.D.USD.EUR.SP00.A",
        "source_terms_url": ECB_COPYRIGHT_URL,
        "transformation_notice": "Reproduction authors calculated natural-log returns and an equal-width empirical histogram from ECB rates; not ECB-produced derived data",
        "source_url": settings["source_url"],
        "source_sha256": digest,
        "requested_period": [settings["start_date"], settings["end_date"]],
        "observed_period": [dates[0], dates[-1]],
        "valid_rates": len(rates),
        "missing_rate_rows_dropped": missing,
        "return_count": len(returns),
        "return_definition": "ln(USD_per_EUR[t]) - ln(USD_per_EUR[t-1_valid_observation])",
        "calendar_policy": "successive available observations; no interpolation or forward fill",
        "bins": len(states),
        "binning": settings["binning"],
        "return_range": [float(returns.min()), float(returns.max())],
        "occupied_bins": int(np.count_nonzero(counts)),
        "paper_deviations": [
            "Original provider, exact dates and bin boundaries are unpublished.",
            "ECB 2004-2023 and equal-width observed-range bins are reproduction choices.",
            "All returns define the empirical target; no held-out generalization claim.",
        ],
    }
    return probabilities, arrays, metadata


def save_financial_target(directory: Path, arrays: dict, metadata: dict) -> None:
    """Save a frozen processed target and its provenance in the shared data root.

    Parameters
    ----------
    directory : pathlib.Path
        Dataset destination.
    arrays : dict
        Arrays returned by financial_target.
    metadata : dict
        JSON-serializable source and preprocessing specification.
    """
    directory.mkdir(parents=True, exist_ok=True)
    np.savez(directory / "target.npz", **arrays)
    (directory / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )


AUTHOR_COMMIT = "d593d209128e403644b9aee595c474ce551391ae"


def _download_bytes(settings, timeout):
    url = settings["source_url"]
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.netloc != "data-api.ecb.europa.eu"
        or parsed.path != "/service/data/EXR/D.USD.EUR.SP00.A"
        or parsed.fragment
        or parse_qs(parsed.query)
        != {
            "startPeriod": [settings["start_date"]],
            "endPeriod": [settings["end_date"]],
            "format": ["csvdata"],
        }
    ):
        raise ValueError(
            "Download requires the configured official ECB daily USD-per-EUR CSV query"
        )
    request = Request(
        url, headers={"Accept": "text/csv", "User-Agent": "error-qcbm-reproduction"}
    )
    with urlopen(request, timeout=timeout) as response:
        if response.status != 200:
            raise ValueError(f"ECB download returned HTTP {response.status}")
        limit = 20 * 1024 * 1024
        raw = response.read(limit + 1)
        if len(raw) > limit:
            raise ValueError("ECB response exceeds the 20 MiB acquisition limit")
        return raw, {
            "http_status": response.status,
            "content_type": response.headers.get("Content-Type"),
            "last_modified": response.headers.get("Last-Modified"),
            "etag": response.headers.get("ETag"),
        }


def acquire_financial_source(
    settings: dict, directory: Path, options: dict, report: dict, report_path: Path
) -> Path:
    """Use an explicit local CSV or download and verify an immutable candidate.

    Parameters
    ----------
    settings : dict
        Financial source filename, SHA256, official URL and date interval.
    directory : pathlib.Path
        Shared ``error-qcbm/fx`` directory.
    options : dict
        Explicit boolean ``download`` and positive ``timeout_seconds``.
    report : dict
        Mutable run-level report, including acquisition errors and candidate hash.
    report_path : pathlib.Path
        Where to persist candidate provenance before hash validation can fail.

    Returns
    -------
    pathlib.Path
        Source verified against the frozen configuration. Existing different
        files are never overwritten; rejected downloads remain hash-named files.

    Raises
    ------
    ValueError
        If settings, download origin or a source hash disagree.
    FileNotFoundError
        If offline mode has no source. Network errors propagate explicitly.
    """
    name = settings["source_file"]
    if (
        not isinstance(name, str)
        or Path(name).name != name
        or "/" in name
        or "\\" in name
    ):
        raise ValueError("Financial source_file must be a plain filename")
    if not isinstance(options["download"], bool) or options["timeout_seconds"] <= 0:
        raise ValueError(
            "Data preparation needs boolean download and positive timeout_seconds"
        )
    source = directory / name
    if not options["download"]:
        report["acquisition"] = {
            "mode": "existing_local_csv",
            "original_acquired_at_utc": None,
            "network_requested": False,
        }
        raw = source.read_bytes()
    else:
        report["acquisition"] = {
            "mode": "explicit_official_download",
            "network_requested": True,
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "url": settings["source_url"],
        }
        write_json(report_path, report)
        raw, headers = _download_bytes(settings, options["timeout_seconds"])
        digest = hashlib.sha256(raw).hexdigest()
        candidate = directory / "downloads" / f"{digest}.csv"
        candidate.parent.mkdir(parents=True, exist_ok=True)
        if candidate.exists():
            if candidate.read_bytes() != raw:
                raise ValueError(
                    "Existing download candidate is inconsistent with its hash"
                )
        else:
            candidate.write_bytes(raw)
        report["acquisition"] = {
            "mode": "explicit_official_download",
            "network_requested": True,
            "acquired_at_utc": datetime.now(timezone.utc).isoformat(),
            "url": settings["source_url"],
            "candidate_sha256": digest,
            "candidate_relative_to_fx": str(candidate.relative_to(directory)),
            "response": headers,
        }
        write_json(report_path, report)
    digest = hashlib.sha256(raw).hexdigest()
    report["source_sha256"] = digest
    if digest != settings["sha256"]:
        raise ValueError(
            "Financial source SHA256 differs from the frozen configuration; candidate retained, no replacement or training"
        )
    if options["download"]:
        if source.exists():
            if source.read_bytes() != raw:
                raise ValueError(
                    "Existing financial source differs; refusing to overwrite it"
                )
        else:
            temporary = source.with_suffix(source.suffix + ".tmp")
            temporary.write_bytes(raw)
            atomic_replace(temporary, source)
    return source


def prepare_data(cfg: dict, run_dir: Path) -> None:
    """Prepare known targets through the shared runtime without training.

    Parameters
    ----------
    cfg : dict
        Dataset, model, data_root and explicit data_preparation options. Gaussian
        targets need no network; FX requires a local frozen CSV or opt-in download.
    run_dir : pathlib.Path
        Timestamped shared-runtime directory for the preparation report.

    Raises
    ------
    ValueError
        If the dataset, acquisition policy or data contents are invalid.
    FileNotFoundError
        If a required offline input is absent. Failures are recorded then raised.
    """
    dataset = cfg["dataset"]
    if dataset["name"] not in ("gaussian", "fx"):
        raise ValueError("Data preparation supports gaussian or fx")
    options = cfg["data_preparation"]
    if not isinstance(options["download"], bool):
        raise ValueError("data_preparation.download must be boolean")
    root = paper_data_dir("error-qcbm", data_root=cfg["data_root"])
    directory = root / dataset["name"]
    directory.mkdir(parents=True, exist_ok=True)
    input_state = cfg["model"]["input_state"]
    states = collision_free_states(len(input_state), sum(input_state))
    report = {
        "status": "preparing",
        "dataset": dataset["name"],
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "qpu_jobs_submitted": 0,
        "training_started": False,
    }
    report_path = run_dir / "data_preparation.json"
    try:
        if dataset["name"] == "gaussian":
            if options["download"]:
                raise ValueError(
                    "Gaussian is a synthetic target; download must be false"
                )
            target = gaussian_target(
                len(states), dataset["means"], dataset["standard_deviations"]
            )
            arrays = {
                "probabilities": target,
                "coordinates": np.arange(len(states)),
                "states": np.asarray(states),
            }
            metadata = {
                "source": "Synthetic author-defined Gaussian mixture; no finite training dataset",
                "author_commit": AUTHOR_COMMIT,
                "source_url": f"https://github.com/Quandela/photonic-qcbm/blob/{AUTHOR_COMMIT}/src/helpers/utils.py",
                "bins": len(states),
                "means": dataset["means"],
                "standard_deviations": dataset["standard_deviations"],
                "definition": "Sum two normal PDFs at integer coordinates 0..bins-1, then normalize; not bin integrals",
                "state_order": "ascending binary lexicographic collision-free states",
                "training_representation": "known normalized probabilities",
            }
            report["acquisition"] = {"mode": "generated", "network_requested": False}
        else:
            source = acquire_financial_source(
                dataset, directory, options, report, report_path
            )
            target, arrays, metadata = financial_target(source, dataset, states)
            metadata["acquisition"] = report["acquisition"]
        metadata["probability_vector_sha256_float64_le"] = hashlib.sha256(
            np.asarray(target, dtype="<f8").tobytes()
        ).hexdigest()
        # Gaussian and FX share the same NPZ/metadata artifact format.
        save_financial_target(directory, arrays, metadata)
        report.update(
            status="complete",
            bins=len(states),
            target_metadata=metadata,
            target_relative_to_data_root=f"error-qcbm/{dataset['name']}/target.npz",
        )
        write_json(report_path, report)
    except Exception as error:
        report.update(
            status="failed", error_type=type(error).__name__, error=str(error)
        )
        write_json(report_path, report)
        raise
