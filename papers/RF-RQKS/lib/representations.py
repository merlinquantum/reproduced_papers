"""Create and normalize RF-RQKS input representations without split leakage."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.fft import dctn


@dataclass(frozen=True)
class DctRepresentationConfig:
    """Configure a cached two-dimensional DCT representation.

    Parameters
    ----------
    frequency_coefficients : int
        Number of low-index coefficients retained on the frequency axis.
    time_coefficients : int
        Number of low-index coefficients retained on the time axis.
    dct_type : int
        Discrete cosine transform type.
    normalization : str
        SciPy DCT normalization mode.
    batch_size : int
        Number of spectrograms transformed together.
    output_dtype : str
        Floating-point dtype of the cached feature matrix.
    workers : int
        Maximum number of parallel SciPy FFT workers.
    """

    frequency_coefficients: int
    time_coefficients: int
    dct_type: int
    normalization: str
    batch_size: int
    output_dtype: str
    workers: int

    @property
    def feature_count(self) -> int:
        """Return the flattened representation dimension.

        Returns
        -------
        int
            Product of retained frequency and time coefficients.
        """
        return self.frequency_coefficients * self.time_coefficients

    @property
    def representation_name(self) -> str:
        """Return the retained DCT block's canonical name.

        Returns
        -------
        str
            Name formatted as ``dct<frequency>x<time>``.
        """
        return f"dct{self.frequency_coefficients}x{self.time_coefficients}"

    @classmethod
    def from_json(cls, path: Path) -> DctRepresentationConfig:
        """Load and validate a DCT representation configuration.

        Parameters
        ----------
        path : pathlib.Path
            JSON configuration path.

        Returns
        -------
        DctRepresentationConfig
            Validated immutable configuration.

        Raises
        ------
        KeyError
            If a required configuration entry is absent.
        ValueError
            If a configuration value is invalid.
        """
        raw_config = json.loads(path.read_text(encoding="utf-8"))
        config = cls(
            frequency_coefficients=int(raw_config["frequency_coefficients"]),
            time_coefficients=int(raw_config["time_coefficients"]),
            dct_type=int(raw_config["dct_type"]),
            normalization=str(raw_config["normalization"]),
            batch_size=int(raw_config["batch_size"]),
            output_dtype=str(raw_config["output_dtype"]),
            workers=int(raw_config["workers"]),
        )
        config.validate()
        return config

    def validate(self) -> None:
        """Validate DCT representation parameters.

        Raises
        ------
        ValueError
            If a parameter is outside its supported range.
        """
        if self.frequency_coefficients <= 0 or self.time_coefficients <= 0:
            raise ValueError("DCT coefficient dimensions must be positive")
        if self.dct_type != 2:
            raise ValueError("RF-RQKS uses a type-II DCT")
        if self.normalization != "ortho":
            raise ValueError("RF-RQKS uses orthonormal DCT normalization")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if np.dtype(self.output_dtype).kind != "f":
            raise ValueError("output_dtype must be floating point")
        if self.workers == 0:
            raise ValueError("workers must be nonzero")


def compute_dct_features(
    spectrograms: NDArray[np.floating],
    config: DctRepresentationConfig,
) -> NDArray[np.floating]:
    """Transform spectrograms into flattened low-index DCT blocks.

    Parameters
    ----------
    spectrograms : numpy.ndarray
        Batch with shape ``(samples, frequency, time)``.
    config : DctRepresentationConfig
        DCT and retained-block parameters.

    Returns
    -------
    numpy.ndarray
        Feature matrix with shape ``(samples, feature_count)``.

    Raises
    ------
    ValueError
        If the input is not a three-dimensional batch or is too small.
    """
    if spectrograms.ndim != 3:
        raise ValueError("spectrograms must have shape (samples, frequency, time)")
    if spectrograms.shape[1] < config.frequency_coefficients:
        raise ValueError(
            "Spectrogram frequency dimension is smaller than the DCT block"
        )
    if spectrograms.shape[2] < config.time_coefficients:
        raise ValueError("Spectrogram time dimension is smaller than the DCT block")
    transformed = dctn(
        spectrograms,
        type=config.dct_type,
        norm=config.normalization,
        axes=(-2, -1),
        workers=config.workers,
    )
    retained_block = transformed[
        :, : config.frequency_coefficients, : config.time_coefficients
    ]
    return retained_block.reshape(spectrograms.shape[0], config.feature_count).astype(
        config.output_dtype, copy=False
    )


def fit_feature_standardization(
    features: NDArray[np.floating],
    training_indices: NDArray[np.integer],
    batch_size: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Fit coefficient-wise mean and standard deviation on training rows only.

    Parameters
    ----------
    features : numpy.ndarray
        Unnormalized feature matrix.
    training_indices : numpy.ndarray
        One-dimensional indices belonging to the current training partition.
    batch_size : int
        Number of indexed rows processed together.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Per-feature population mean and standard deviation.

    Raises
    ------
    ValueError
        If indices are empty, malformed, duplicated, out of range, or produce a
        zero-variance feature.
    """
    indices = np.asarray(training_indices)
    if indices.ndim != 1 or indices.size == 0:
        raise ValueError("training_indices must be a non-empty one-dimensional array")
    if not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("training_indices must contain integers")
    if np.unique(indices).size != indices.size:
        raise ValueError("training_indices must not contain duplicates")
    if int(indices.min()) < 0 or int(indices.max()) >= features.shape[0]:
        raise ValueError("training_indices contain an out-of-range row")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    sample_count = 0
    feature_mean = np.zeros(features.shape[1], dtype=np.float64)
    feature_sum_squared_deviations = np.zeros(features.shape[1], dtype=np.float64)
    for start in range(0, indices.size, batch_size):
        batch_indices = indices[start : start + batch_size]
        batch = np.asarray(features[batch_indices], dtype=np.float64)
        batch_count = batch.shape[0]
        batch_mean = batch.mean(axis=0)
        batch_sum_squared_deviations = np.sum((batch - batch_mean) ** 2, axis=0)
        combined_count = sample_count + batch_count
        mean_difference = batch_mean - feature_mean
        feature_mean += mean_difference * batch_count / combined_count
        feature_sum_squared_deviations += batch_sum_squared_deviations
        feature_sum_squared_deviations += (
            mean_difference**2 * sample_count * batch_count / combined_count
        )
        sample_count = combined_count

    feature_standard_deviation = np.sqrt(feature_sum_squared_deviations / sample_count)
    zero_variance_indices = np.flatnonzero(feature_standard_deviation == 0)
    if zero_variance_indices.size:
        raise ValueError(
            "Training partition contains zero-variance DCT features at indices "
            f"{zero_variance_indices.tolist()}"
        )
    return feature_mean, feature_standard_deviation


def standardize_features(
    features: NDArray[np.floating],
    feature_mean: NDArray[np.float64],
    feature_standard_deviation: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Apply training-derived coefficient-wise standardization.

    Parameters
    ----------
    features : numpy.ndarray
        Feature rows to normalize.
    feature_mean : numpy.ndarray
        Mean fitted only on the current training partition.
    feature_standard_deviation : numpy.ndarray
        Standard deviation fitted only on the current training partition.

    Returns
    -------
    numpy.ndarray
        Standardized feature matrix.

    Raises
    ------
    ValueError
        If feature and statistic dimensions differ or a standard deviation is
        not strictly positive.
    """
    if features.ndim != 2:
        raise ValueError("features must be a two-dimensional matrix")
    if feature_mean.shape != (features.shape[1],):
        raise ValueError("feature_mean has an incompatible shape")
    if feature_standard_deviation.shape != (features.shape[1],):
        raise ValueError("feature_standard_deviation has an incompatible shape")
    if np.any(feature_standard_deviation <= 0):
        raise ValueError("feature_standard_deviation must be strictly positive")
    return (features - feature_mean) / feature_standard_deviation


def build_dct_representation_cache(
    input_root: Path,
    output_root: Path,
    config: DctRepresentationConfig,
) -> None:
    """Cache unnormalized train and test DCT representations incrementally.

    Parameters
    ----------
    input_root : pathlib.Path
        Processed RF-RQKS dataset containing train and test spectrogram arrays.
    output_root : pathlib.Path
        New directory for DCT features, labels, metadata, and a manifest.
    config : DctRepresentationConfig
        DCT cache parameters.

    Raises
    ------
    FileExistsError
        If the output or temporary directory already exists.
    FileNotFoundError
        If a required source array or metadata file is absent.
    ValueError
        If source spectrogram or label shapes are inconsistent.
    """
    config.validate()
    if output_root.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing representation: {output_root}"
        )
    temporary_root = output_root.with_name(
        f".{output_root.name}.building-{os.getpid()}"
    )
    if temporary_root.exists():
        raise FileExistsError(
            f"Temporary representation directory already exists: {temporary_root}"
        )

    required_paths = [
        input_root / split_name / filename
        for split_name in ("train", "test")
        for filename in ("spectrograms.npy", "labels.npy", "metadata.csv")
    ]
    missing_paths = [path for path in required_paths if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError(f"Missing required dataset files: {missing_paths}")

    temporary_root.mkdir(parents=True)
    split_sample_counts: dict[str, int] = {}
    for split_name in ("train", "test"):
        input_split_root = input_root / split_name
        output_split_root = temporary_root / split_name
        output_split_root.mkdir()
        spectrograms = np.load(input_split_root / "spectrograms.npy", mmap_mode="r")
        labels = np.load(input_split_root / "labels.npy", mmap_mode="r")
        if spectrograms.ndim != 3:
            raise ValueError(f"{split_name} spectrograms must be three-dimensional")
        if labels.shape != (spectrograms.shape[0],):
            raise ValueError(f"{split_name} labels do not match spectrogram rows")

        features = np.lib.format.open_memmap(
            output_split_root / "features.npy",
            mode="w+",
            dtype=config.output_dtype,
            shape=(spectrograms.shape[0], config.feature_count),
        )
        for start in range(0, spectrograms.shape[0], config.batch_size):
            stop = min(start + config.batch_size, spectrograms.shape[0])
            features[start:stop] = compute_dct_features(
                np.asarray(spectrograms[start:stop]), config
            )
        features.flush()
        np.save(output_split_root / "labels.npy", np.asarray(labels))
        shutil.copy2(
            input_split_root / "metadata.csv", output_split_root / "metadata.csv"
        )
        split_sample_counts[split_name] = int(spectrograms.shape[0])

    manifest = {
        "format_version": 1,
        "representation": config.representation_name,
        "configuration": asdict(config),
        "source_root": str(input_root.resolve()),
        "split_sample_counts": split_sample_counts,
        "array_layout": "features[sample, flattened_frequency_then_time_coefficient]",
        "normalized": False,
        "normalization_policy": (
            "Fit coefficient-wise mean and population standard deviation only on "
            "the training indices of each ablation stage, then apply those statistics "
            "to that stage's training and validation rows. Refit on the full raw "
            "training split for Stage 5."
        ),
    }
    (temporary_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    temporary_root.rename(output_root)
