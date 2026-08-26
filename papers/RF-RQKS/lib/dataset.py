"""Build the paired RF-RQKS spectrogram dataset from measured LTE IQ data."""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage, signal

ANOMALY_TYPES = ("chirp", "barrage", "frequency_hopping")


@dataclass(frozen=True)
class IqSegment:
    """Identify one non-overlapping segment in a measured IQ capture.

    Parameters
    ----------
    path : pathlib.Path
        Path to an interleaved IQ binary file.
    complex_offset : int
        Offset measured in complex IQ samples.
    """

    path: Path
    complex_offset: int


@dataclass(frozen=True)
class DatasetConfig:
    """Validated parameters used to generate an RF-RQKS dataset.

    Parameters
    ----------
    sampling_rate_hz : float
        IQ sampling frequency in hertz.
    iq_points_per_sample : int
        Number of complex IQ points in one source segment.
    bandwidth_hz : float
        Central bandwidth retained in each spectrogram.
    stft_window_length : int
        Hann-window length in IQ points.
    stft_nfft : int
        FFT length used for every STFT frame.
    stft_overlap_ratio : float
        Fraction of adjacent STFT windows that overlaps.
    spectrogram_height : int
        Output frequency-bin count.
    spectrogram_width : int
        Output time-bin count.
    train_source_samples : int
        Number of unique LTE segments assigned to training.
    test_source_samples : int
        Number of unique LTE segments assigned to testing.
    jsr_db_values : tuple[float, ...]
        Permitted jamming-to-signal ratios in decibels.
    minimum_anomaly_fraction : float
        Minimum anomaly duration relative to one IQ segment.
    maximum_anomaly_fraction : float
        Maximum anomaly duration relative to one IQ segment.
    hopping_bandwidth_hz : tuple[float, float]
        Minimum and maximum bandwidth of frequency-hopping noise.
    hopping_dwell_fraction : tuple[float, float]
        Minimum and maximum hop dwell time relative to one IQ segment.
    iq_dtype : str
        NumPy dtype of interleaved I/Q values in the source files.
    output_dtype : str
        NumPy dtype used for spectrogram arrays.
    seed : int
        Master random seed.
    """

    sampling_rate_hz: float
    iq_points_per_sample: int
    bandwidth_hz: float
    stft_window_length: int
    stft_nfft: int
    stft_overlap_ratio: float
    spectrogram_height: int
    spectrogram_width: int
    train_source_samples: int
    test_source_samples: int
    jsr_db_values: tuple[float, ...]
    minimum_anomaly_fraction: float
    maximum_anomaly_fraction: float
    hopping_bandwidth_hz: tuple[float, float]
    hopping_dwell_fraction: tuple[float, float]
    iq_dtype: str
    output_dtype: str
    seed: int

    @classmethod
    def from_json(cls, path: Path) -> DatasetConfig:
        """Load and validate dataset parameters from JSON.

        Parameters
        ----------
        path : pathlib.Path
            Dataset configuration file.

        Returns
        -------
        DatasetConfig
            Validated immutable configuration.

        Raises
        ------
        KeyError
            If a required configuration entry is absent.
        ValueError
            If a configuration value is inconsistent.
        """
        raw_config = json.loads(path.read_text(encoding="utf-8"))
        config = cls(
            sampling_rate_hz=float(raw_config["sampling_rate_hz"]),
            iq_points_per_sample=int(raw_config["iq_points_per_sample"]),
            bandwidth_hz=float(raw_config["bandwidth_hz"]),
            stft_window_length=int(raw_config["stft_window_length"]),
            stft_nfft=int(raw_config["stft_nfft"]),
            stft_overlap_ratio=float(raw_config["stft_overlap_ratio"]),
            spectrogram_height=int(raw_config["spectrogram_height"]),
            spectrogram_width=int(raw_config["spectrogram_width"]),
            train_source_samples=int(raw_config["train_source_samples"]),
            test_source_samples=int(raw_config["test_source_samples"]),
            jsr_db_values=tuple(float(value) for value in raw_config["jsr_db_values"]),
            minimum_anomaly_fraction=float(raw_config["minimum_anomaly_fraction"]),
            maximum_anomaly_fraction=float(raw_config["maximum_anomaly_fraction"]),
            hopping_bandwidth_hz=tuple(
                float(value) for value in raw_config["hopping_bandwidth_hz"]
            ),
            hopping_dwell_fraction=tuple(
                float(value) for value in raw_config["hopping_dwell_fraction"]
            ),
            iq_dtype=str(raw_config["iq_dtype"]),
            output_dtype=str(raw_config["output_dtype"]),
            seed=int(raw_config["seed"]),
        )
        config.validate()
        return config

    def validate(self) -> None:
        """Validate relationships between generation parameters.

        Raises
        ------
        ValueError
            If a parameter is outside its valid range.
        """
        if self.sampling_rate_hz <= 0:
            raise ValueError("sampling_rate_hz must be positive")
        if not 0 < self.bandwidth_hz <= self.sampling_rate_hz:
            raise ValueError("bandwidth_hz must be in (0, sampling_rate_hz]")
        if self.iq_points_per_sample < self.stft_window_length:
            raise ValueError("iq_points_per_sample must be at least stft_window_length")
        if self.stft_nfft < self.stft_window_length:
            raise ValueError("stft_nfft must be at least stft_window_length")
        if not 0 <= self.stft_overlap_ratio < 1:
            raise ValueError("stft_overlap_ratio must be in [0, 1)")
        if self.spectrogram_height <= 0 or self.spectrogram_width <= 0:
            raise ValueError("spectrogram dimensions must be positive")
        if self.train_source_samples <= 0 or self.test_source_samples <= 0:
            raise ValueError(
                "train_source_samples and test_source_samples must be positive"
            )
        if not self.jsr_db_values:
            raise ValueError("jsr_db_values must not be empty")
        if not 0 < self.minimum_anomaly_fraction <= self.maximum_anomaly_fraction <= 1:
            raise ValueError(
                "anomaly fractions must satisfy 0 < minimum <= maximum <= 1"
            )
        if len(self.hopping_bandwidth_hz) != 2 or not (
            0
            < self.hopping_bandwidth_hz[0]
            <= self.hopping_bandwidth_hz[1]
            < self.bandwidth_hz
        ):
            raise ValueError(
                "hopping_bandwidth_hz must be an increasing pair inside bandwidth_hz"
            )
        if len(self.hopping_dwell_fraction) != 2 or not (
            0 < self.hopping_dwell_fraction[0] <= self.hopping_dwell_fraction[1] <= 1
        ):
            raise ValueError(
                "hopping_dwell_fraction must be an increasing pair in (0, 1]"
            )
        if np.dtype(self.iq_dtype).kind not in "iuf":
            raise ValueError("iq_dtype must be a real integer or floating-point dtype")
        if np.dtype(self.output_dtype).kind != "f":
            raise ValueError("output_dtype must be a floating-point dtype")


def discover_iq_segments(input_root: Path, config: DatasetConfig) -> list[IqSegment]:
    """Discover non-overlapping IQ segments in binary captures.

    Parameters
    ----------
    input_root : pathlib.Path
        Directory recursively searched for ``.bin`` files.
    config : DatasetConfig
        Dataset generation parameters.

    Returns
    -------
    list[IqSegment]
        Segments in deterministic path and offset order.

    Raises
    ------
    ValueError
        If a binary file has an odd number of real scalar values.
    """
    scalar_bytes = np.dtype(config.iq_dtype).itemsize
    segments: list[IqSegment] = []
    for path in sorted(input_root.rglob("*.bin")):
        scalar_count, remainder = divmod(path.stat().st_size, scalar_bytes)
        if remainder or scalar_count % 2:
            raise ValueError(f"Invalid interleaved IQ byte count in {path}")
        complex_count = scalar_count // 2
        for complex_offset in range(
            0,
            complex_count - config.iq_points_per_sample + 1,
            config.iq_points_per_sample,
        ):
            segments.append(IqSegment(path=path, complex_offset=complex_offset))
    return segments


def load_iq_segment(
    segment: IqSegment, config: DatasetConfig
) -> NDArray[np.complex128]:
    """Load one interleaved I/Q segment as a complex array.

    Parameters
    ----------
    segment : IqSegment
        Source file and complex-sample offset.
    config : DatasetConfig
        Dataset generation parameters.

    Returns
    -------
    numpy.ndarray
        Complex IQ vector with ``iq_points_per_sample`` entries.

    Raises
    ------
    ValueError
        If the requested segment cannot be read in full.
    """
    raw_values = np.fromfile(
        segment.path,
        dtype=np.dtype(config.iq_dtype),
        count=2 * config.iq_points_per_sample,
        offset=2 * segment.complex_offset * np.dtype(config.iq_dtype).itemsize,
    )
    if raw_values.size != 2 * config.iq_points_per_sample:
        raise ValueError(
            f"Incomplete IQ segment at offset {segment.complex_offset} in {segment.path}"
        )
    return raw_values[::2].astype(np.float64) + 1j * raw_values[1::2].astype(np.float64)


def scale_anomaly_to_jsr(
    lte_iq: NDArray[np.complex128],
    anomaly_iq: NDArray[np.complex128],
    jsr_db: float,
) -> NDArray[np.complex128]:
    """Scale an anomaly so its active-sample power has the requested JSR.

    Parameters
    ----------
    lte_iq : numpy.ndarray
        Measured LTE IQ segment.
    anomaly_iq : numpy.ndarray
        Unscaled, gated anomaly waveform.
    jsr_db : float
        Target anomaly-to-LTE power ratio in decibels.

    Returns
    -------
    numpy.ndarray
        Scaled complex anomaly waveform.

    Raises
    ------
    ValueError
        If either reference power is zero or non-finite.
    """
    active_samples = np.abs(anomaly_iq) > 0
    lte_power = float(np.mean(np.abs(lte_iq[active_samples]) ** 2))
    anomaly_power = float(np.mean(np.abs(anomaly_iq[active_samples]) ** 2))
    if not np.isfinite(lte_power) or lte_power <= 0:
        raise ValueError(
            "LTE signal power must be positive and finite over the anomaly interval"
        )
    if not np.isfinite(anomaly_power) or anomaly_power <= 0:
        raise ValueError("Anomaly signal power must be positive and finite")
    amplitude_scale = math.sqrt(lte_power * 10.0 ** (jsr_db / 10.0) / anomaly_power)
    return anomaly_iq * amplitude_scale


def generate_anomaly(
    lte_iq: NDArray[np.complex128],
    anomaly_type: str,
    jsr_db: float,
    rng: np.random.Generator,
    config: DatasetConfig,
) -> tuple[NDArray[np.complex128], dict[str, Any]]:
    """Generate one gated anomaly and scale it to a requested JSR.

    Parameters
    ----------
    lte_iq : numpy.ndarray
        Measured LTE IQ segment.
    anomaly_type : str
        One of ``chirp``, ``barrage``, or ``frequency_hopping``.
    jsr_db : float
        Requested jamming-to-signal ratio in decibels.
    rng : numpy.random.Generator
        Random generator dedicated to this sample.
    config : DatasetConfig
        Dataset generation parameters.

    Returns
    -------
    tuple[numpy.ndarray, dict]
        Scaled waveform and explicit generation parameters.

    Raises
    ------
    ValueError
        If ``anomaly_type`` is unsupported.
    """
    sample_count = lte_iq.size
    duration_fraction = rng.uniform(
        config.minimum_anomaly_fraction, config.maximum_anomaly_fraction
    )
    duration_samples = max(
        1, min(sample_count, int(round(duration_fraction * sample_count)))
    )
    start_sample = int(rng.integers(0, sample_count - duration_samples + 1))
    stop_sample = start_sample + duration_samples
    time_seconds = (
        np.arange(duration_samples, dtype=np.float64) / config.sampling_rate_hz
    )
    half_bandwidth_hz = config.bandwidth_hz / 2.0
    parameters: dict[str, Any] = {
        "start_sample": start_sample,
        "stop_sample": stop_sample,
        "duration_samples": duration_samples,
    }

    if anomaly_type == "chirp":
        start_frequency_hz, stop_frequency_hz = rng.uniform(
            -half_bandwidth_hz, half_bandwidth_hz, size=2
        )
        chirp_rate = (stop_frequency_hz - start_frequency_hz) / (
            duration_samples / config.sampling_rate_hz
        )
        phase = (
            2.0
            * np.pi
            * (start_frequency_hz * time_seconds + 0.5 * chirp_rate * time_seconds**2)
        )
        active_waveform = np.exp(1j * phase)
        parameters.update(
            start_frequency_hz=float(start_frequency_hz),
            stop_frequency_hz=float(stop_frequency_hz),
        )
    elif anomaly_type == "barrage":
        active_waveform = _band_limited_noise(
            duration_samples,
            config.bandwidth_hz,
            config.sampling_rate_hz,
            rng,
        )
        parameters["bandwidth_hz"] = config.bandwidth_hz
    elif anomaly_type == "frequency_hopping":
        hopping_bandwidth_hz = float(rng.uniform(*config.hopping_bandwidth_hz))
        baseband_noise = _band_limited_noise(
            duration_samples,
            hopping_bandwidth_hz,
            config.sampling_rate_hz,
            rng,
        )
        active_waveform = np.empty(duration_samples, dtype=np.complex128)
        minimum_dwell = max(
            1, int(round(config.hopping_dwell_fraction[0] * sample_count))
        )
        maximum_dwell = max(
            minimum_dwell, int(round(config.hopping_dwell_fraction[1] * sample_count))
        )
        hop_centers_hz: list[float] = []
        hop_ranges: list[list[int]] = []
        hop_start = 0
        while hop_start < duration_samples:
            dwell_samples = int(rng.integers(minimum_dwell, maximum_dwell + 1))
            hop_stop = min(duration_samples, hop_start + dwell_samples)
            maximum_center_hz = half_bandwidth_hz - hopping_bandwidth_hz / 2.0
            center_frequency_hz = float(
                rng.uniform(-maximum_center_hz, maximum_center_hz)
            )
            local_time = (
                np.arange(hop_stop - hop_start, dtype=np.float64)
                / config.sampling_rate_hz
            )
            active_waveform[hop_start:hop_stop] = baseband_noise[
                hop_start:hop_stop
            ] * np.exp(1j * 2.0 * np.pi * center_frequency_hz * local_time)
            hop_centers_hz.append(center_frequency_hz)
            hop_ranges.append([hop_start + start_sample, hop_stop + start_sample])
            hop_start = hop_stop
        parameters.update(
            bandwidth_hz=hopping_bandwidth_hz,
            hop_centers_hz=hop_centers_hz,
            hop_sample_ranges=hop_ranges,
        )
    else:
        raise ValueError(f"Unsupported anomaly type: {anomaly_type}")

    anomaly_iq = np.zeros(sample_count, dtype=np.complex128)
    anomaly_iq[start_sample:stop_sample] = active_waveform
    return scale_anomaly_to_jsr(lte_iq, anomaly_iq, jsr_db), parameters


def iq_to_spectrogram(
    iq: NDArray[np.complex128], config: DatasetConfig
) -> NDArray[np.floating[Any]]:
    """Convert one IQ segment to a cropped 400-by-400 log-magnitude STFT.

    Parameters
    ----------
    iq : numpy.ndarray
        Complex IQ segment.
    config : DatasetConfig
        STFT, bandwidth, and output parameters.

    Returns
    -------
    numpy.ndarray
        Frequency-by-time spectrogram in decibels.

    Raises
    ------
    ValueError
        If the bandwidth cannot be represented by a symmetric FFT crop.
    """
    overlap_samples = int(config.stft_window_length * config.stft_overlap_ratio)
    _, _, stft_values = signal.stft(
        iq,
        fs=config.sampling_rate_hz,
        window="hann",
        nperseg=config.stft_window_length,
        noverlap=overlap_samples,
        nfft=config.stft_nfft,
        boundary=None,
        padded=False,
        return_onesided=False,
    )
    shifted_magnitude = np.abs(np.fft.fftshift(stft_values, axes=0))
    retained_bin_count = int(
        round(config.stft_nfft * config.bandwidth_hz / config.sampling_rate_hz)
    )
    if retained_bin_count <= 0 or retained_bin_count > config.stft_nfft:
        raise ValueError("Configured bandwidth produces an invalid FFT-bin count")
    first_bin = (config.stft_nfft - retained_bin_count) // 2
    cropped_magnitude = shifted_magnitude[first_bin : first_bin + retained_bin_count]
    log_magnitude = 20.0 * np.log10(
        np.maximum(cropped_magnitude, np.finfo(np.float64).tiny)
    )
    zoom_factors = (
        config.spectrogram_height / log_magnitude.shape[0],
        config.spectrogram_width / log_magnitude.shape[1],
    )
    resized = ndimage.zoom(log_magnitude, zoom_factors, order=1, prefilter=False)
    resized = resized[: config.spectrogram_height, : config.spectrogram_width]
    if resized.shape != (config.spectrogram_height, config.spectrogram_width):
        raise ValueError(
            f"Spectrogram resize produced unexpected shape {resized.shape}"
        )
    return resized.astype(config.output_dtype, copy=False)


def build_dataset(input_root: Path, output_root: Path, config: DatasetConfig) -> None:
    """Generate train and test arrays plus provenance metadata on disk.

    Parameters
    ----------
    input_root : pathlib.Path
        Root containing measured interleaved-IQ ``.bin`` files.
    output_root : pathlib.Path
        New directory in which the complete dataset is written.
    config : DatasetConfig
        Validated dataset parameters.

    Raises
    ------
    FileExistsError
        If ``output_root`` or its temporary build directory already exists.
    ValueError
        If too few measured LTE segments are available.
    """
    config.validate()
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing dataset: {output_root}")
    temporary_root = output_root.with_name(
        f".{output_root.name}.building-{os.getpid()}"
    )
    if temporary_root.exists():
        raise FileExistsError(
            f"Temporary build directory already exists: {temporary_root}"
        )

    discovered_segments = discover_iq_segments(input_root, config)
    required_segments = config.train_source_samples + config.test_source_samples
    if len(discovered_segments) < required_segments:
        raise ValueError(
            f"Found {len(discovered_segments)} IQ segments; {required_segments} are required"
        )

    selection_rng = np.random.default_rng(config.seed)
    selected_indices = selection_rng.permutation(len(discovered_segments))[
        :required_segments
    ]
    selected_segments = [discovered_segments[index] for index in selected_indices]
    temporary_root.mkdir(parents=True)
    split_boundaries = {
        "train": (0, config.train_source_samples),
        "test": (config.train_source_samples, required_segments),
    }
    for split_name, (start, stop) in split_boundaries.items():
        _write_split(
            selected_segments[start:stop],
            split_name,
            input_root,
            temporary_root / split_name,
            config,
        )

    manifest = {
        "format_version": 1,
        "configuration": asdict(config),
        "source_root": str(input_root.resolve()),
        "source_segment_count": required_segments,
        "train_samples": 2 * config.train_source_samples,
        "test_samples": 2 * config.test_source_samples,
        "class_labels": {"normal": 0, "anomalous": 1},
        "array_layout": "spectrograms[sample, frequency, time]",
        "spectrogram_units": "dB log magnitude",
        "notes": [
            "The paper specifies the sample/STFT dimensions but not exact anomaly parameter distributions.",
            "The configurable anomaly distributions are explicit in configuration and metadata.",
            "Bilinear interpolation is used to resize the cropped STFT to the target resolution.",
        ],
    }
    plotting_script = (
        Path(__file__).resolve().parents[1] / "utils" / "plot_data_examples.py"
    )
    shutil.copy2(plotting_script, temporary_root / "plot_data_examples.py")
    (temporary_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    temporary_root.rename(output_root)


def _write_split(
    segments: list[IqSegment],
    split_name: str,
    input_root: Path,
    split_root: Path,
    config: DatasetConfig,
) -> None:
    split_root.mkdir()
    sample_count = 2 * len(segments)
    spectrograms = np.lib.format.open_memmap(
        split_root / "spectrograms.npy",
        mode="w+",
        dtype=config.output_dtype,
        shape=(sample_count, config.spectrogram_height, config.spectrogram_width),
    )
    labels = np.lib.format.open_memmap(
        split_root / "labels.npy", mode="w+", dtype=np.uint8, shape=(sample_count,)
    )
    metadata_path = split_root / "metadata.csv"
    fieldnames = [
        "sample_index",
        "pair_index",
        "label",
        "source_path",
        "source_complex_offset",
        "anomaly_type",
        "jsr_db",
        "anomaly_seed",
        "anomaly_parameters",
    ]
    split_seed_offset = 0 if split_name == "train" else config.train_source_samples
    progress_interval = max(1, min(100, len(segments) // 20))
    with metadata_path.open("w", encoding="utf-8", newline="") as metadata_file:
        writer = csv.DictWriter(metadata_file, fieldnames=fieldnames)
        writer.writeheader()
        for pair_index, segment in enumerate(segments):
            lte_iq = load_iq_segment(segment, config)
            normal_index = 2 * pair_index
            anomaly_index = normal_index + 1
            spectrograms[normal_index] = iq_to_spectrogram(lte_iq, config)
            labels[normal_index] = 0
            relative_source_path = str(segment.path.relative_to(input_root))
            writer.writerow(
                {
                    "sample_index": normal_index,
                    "pair_index": pair_index,
                    "label": 0,
                    "source_path": relative_source_path,
                    "source_complex_offset": segment.complex_offset,
                    "anomaly_type": "",
                    "jsr_db": "",
                    "anomaly_seed": "",
                    "anomaly_parameters": "",
                }
            )

            anomaly_seed = config.seed + split_seed_offset + pair_index + 1
            anomaly_rng = np.random.default_rng(anomaly_seed)
            anomaly_type = ANOMALY_TYPES[
                int(anomaly_rng.integers(0, len(ANOMALY_TYPES)))
            ]
            jsr_db = config.jsr_db_values[
                int(anomaly_rng.integers(0, len(config.jsr_db_values)))
            ]
            anomaly_iq, anomaly_parameters = generate_anomaly(
                lte_iq, anomaly_type, jsr_db, anomaly_rng, config
            )
            spectrograms[anomaly_index] = iq_to_spectrogram(lte_iq + anomaly_iq, config)
            labels[anomaly_index] = 1
            writer.writerow(
                {
                    "sample_index": anomaly_index,
                    "pair_index": pair_index,
                    "label": 1,
                    "source_path": relative_source_path,
                    "source_complex_offset": segment.complex_offset,
                    "anomaly_type": anomaly_type,
                    "jsr_db": jsr_db,
                    "anomaly_seed": anomaly_seed,
                    "anomaly_parameters": json.dumps(
                        anomaly_parameters, separators=(",", ":")
                    ),
                }
            )
            processed_pair_count = pair_index + 1
            if (
                processed_pair_count % progress_interval == 0
                or processed_pair_count == len(segments)
            ):
                print(
                    f"[{split_name}] processed {processed_pair_count}/{len(segments)} LTE pairs",
                    flush=True,
                )
    spectrograms.flush()
    labels.flush()


def _band_limited_noise(
    sample_count: int,
    bandwidth_hz: float,
    sampling_rate_hz: float,
    rng: np.random.Generator,
) -> NDArray[np.complex128]:
    white_noise = rng.standard_normal(sample_count) + 1j * rng.standard_normal(
        sample_count
    )
    cutoff_hz = bandwidth_hz / 2.0
    taps = signal.firwin(129, cutoff_hz, fs=sampling_rate_hz)
    return signal.lfilter(taps, [1.0], white_noise)
