#!/usr/bin/env python3
"""Plot normal and anomaly-injected RF-RQKS spectrogram examples."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

PAPER_ROOT = Path(__file__).resolve().parents[1]
PANEL_TITLES = {
    "normal": "(a) Normal Spectrogram",
    "chirp": "(b) Chirp Anomaly",
    "barrage": "(c) Barrage Jamming",
    "frequency_hopping": "(d) Frequency Hopping Jamming",
}
DEMO_DURATION_FRACTIONS = {
    "chirp": 0.56,
    "barrage": 0.012,
    "frequency_hopping": 0.95,
}
DEMO_ANOMALY_SEEDS = {"chirp": 237, "barrage": 2, "frequency_hopping": 2}


def parse_arguments() -> argparse.Namespace:
    """Parse example-figure arguments.

    Returns
    -------
    argparse.Namespace
        Input mode, configuration, JSR, seed, and output path.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--iq-path",
        type=Path,
        help="Measured interleaved-IQ .bin file containing at least one segment",
    )
    input_group.add_argument(
        "--synthetic-demo",
        action="store_true",
        help="Use a synthetic LTE-like background for pipeline verification",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PAPER_ROOT / "configs" / "dataset.json",
        help="Dataset JSON configuration",
    )
    parser.add_argument("--jsr-db", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument(
        "--output",
        type=Path,
        default=PAPER_ROOT / "results" / "rf_rqks_dataset_examples.png",
    )
    return parser.parse_args()


def generate_synthetic_lte_like_iq(
    sample_count: int,
    sampling_rate_hz: float,
    seed: int,
) -> np.ndarray:
    """Create a two-subband complex waveform for visual pipeline verification.

    Parameters
    ----------
    sample_count : int
        Number of complex IQ samples to generate.
    sampling_rate_hz : float
        Sampling frequency in hertz.
    seed : int
        Random seed controlling the waveform.

    Returns
    -------
    numpy.ndarray
        Synthetic LTE-like complex baseband waveform.
    """
    rng = np.random.default_rng(seed)
    time_seconds = np.arange(sample_count, dtype=np.float64) / sampling_rate_hz
    filter_taps = signal.firwin(257, 3_500_000.0, fs=sampling_rate_hz)
    waveform = np.zeros(sample_count, dtype=np.complex128)
    for center_frequency_hz in (-5_500_000.0, 5_500_000.0):
        white_noise = rng.standard_normal(sample_count) + 1j * rng.standard_normal(
            sample_count
        )
        subband = signal.lfilter(filter_taps, [1.0], white_noise)
        waveform += subband * np.exp(
            1j * 2.0 * np.pi * center_frequency_hz * time_seconds
        )

    amplitude_envelope = np.ones(sample_count, dtype=np.float64)
    for _ in range(24):
        burst_length = int(rng.integers(sample_count // 300, sample_count // 80))
        burst_start = int(rng.integers(0, sample_count - burst_length))
        amplitude_envelope[burst_start : burst_start + burst_length] *= rng.uniform(
            0.15, 1.8
        )
    waveform *= amplitude_envelope
    return waveform / np.sqrt(np.mean(np.abs(waveform) ** 2))


def main() -> None:
    """Generate the four-panel spectrogram verification figure."""
    sys.path.insert(0, str(PAPER_ROOT))
    from lib.dataset import (
        DatasetConfig,
        IqSegment,
        generate_anomaly,
        iq_to_spectrogram,
        load_iq_segment,
    )

    arguments = parse_arguments()
    config = DatasetConfig.from_json(arguments.config)
    if arguments.synthetic_demo:
        normal_iq = generate_synthetic_lte_like_iq(
            config.iq_points_per_sample,
            config.sampling_rate_hz,
            arguments.seed,
        )
        source_label = "synthetic LTE-like verification background"
    else:
        normal_iq = load_iq_segment(IqSegment(arguments.iq_path, 0), config)
        source_label = arguments.iq_path.name

    spectrograms = {"normal": iq_to_spectrogram(normal_iq, config)}
    realized_parameters: dict[str, dict[str, object]] = {}
    for anomaly_type in ("chirp", "barrage", "frequency_hopping"):
        duration_fraction = DEMO_DURATION_FRACTIONS[anomaly_type]
        anomaly_config = replace(
            config,
            minimum_anomaly_fraction=duration_fraction,
            maximum_anomaly_fraction=duration_fraction,
            hopping_bandwidth_hz=(2_000_000.0, 2_000_000.0),
            hopping_dwell_fraction=(0.02, 0.05),
        )
        anomaly_rng = np.random.default_rng(DEMO_ANOMALY_SEEDS[anomaly_type])
        anomaly_iq, parameters = generate_anomaly(
            normal_iq,
            anomaly_type,
            arguments.jsr_db,
            anomaly_rng,
            anomaly_config,
        )
        spectrograms[anomaly_type] = iq_to_spectrogram(normal_iq + anomaly_iq, config)
        realized_parameters[anomaly_type] = parameters

    figure, axes = plt.subplots(1, 4, figsize=(20, 5.2), constrained_layout=True)
    frequency_extent_mhz = config.bandwidth_hz / 2.0 / 1e6
    duration_ms = config.iq_points_per_sample / config.sampling_rate_hz * 1e3
    for axis, (panel_name, spectrogram) in zip(axes, spectrograms.items()):
        lower_limit, upper_limit = np.percentile(spectrogram, [1.0, 99.8])
        image = axis.imshow(
            spectrogram.T,
            origin="upper",
            aspect="auto",
            extent=[-frequency_extent_mhz, frequency_extent_mhz, duration_ms, 0.0],
            cmap="viridis",
            vmin=lower_limit,
            vmax=upper_limit,
        )
        axis.set_title(PANEL_TITLES[panel_name], fontsize=12)
        axis.set_xlabel("Frequency [MHz]")
        axis.set_ylabel("Time [ms]")
        colorbar = figure.colorbar(image, ax=axis, pad=0.02, fraction=0.046)
        colorbar.set_label("Magnitude [dB]")

    figure.suptitle(
        f"RF-RQKS dataset pipeline verification - {source_label} - JSR {arguments.jsr_db:g} dB",
        fontsize=14,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(arguments.output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(f"Wrote {arguments.output}")
    for anomaly_type, parameters in realized_parameters.items():
        print(f"{anomaly_type}: {parameters}")


if __name__ == "__main__":
    main()
