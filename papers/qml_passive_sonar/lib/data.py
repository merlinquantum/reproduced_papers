"""Data loading and preprocessing for QML passive-sonar reproduction.

This module covers three concerns:

1. **Audio frame extraction.** Real .wav files (if present) are sliced into
   60-second frames with 25 % overlap, DC-removed, and unit-variance
   normalised — matching Sec. 4.2 of the paper.
2. **Spectrogram features.** Each frame is converted to a 2-D image either
   via SAV variance spectra stacked over time, via DEMON envelope
   spectrograms, or via plain STFT magnitudes, then resized to ``224 × 224``
   with channel replication so it matches the CNN input contract.
3. **Synthetic fallback.** When no real audio is available, this module
   procedurally generates propeller-noise-like signals with class-specific
   tonal structure so the rest of the pipeline can run end-to-end.

The real datasets (ShipsEar, DeepShip) require manual download due to
licensing and size constraints — see the paper README for instructions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlretrieve

import numpy as np
import torch
from lib.demon import compute_demon
from lib.sav import compute_sav
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)

# Default classes per dataset (matches paper Sec. 4.1)
SHIPEAR_CLASSES = ("A", "B", "C", "D", "E")
DEEPSHIP_CLASSES = ("F", "G", "H", "I")
DEEPSHIP_GITHUB_BASE_URL = (
    "https://raw.githubusercontent.com/irfankamboh/DeepShip/main"
)
DEEPSHIP_GITHUB_SAMPLE = {
    "F": ("Cargo", ("41.wav", "69.wav")),
    "G": ("Passengership", ("16.wav", "30.wav")),
    "H": ("Tanker", ("19.wav", "50.wav")),
    "I": ("Tug", ("40.wav", "49.wav")),
}


@dataclass
class FrameSpec:
    """Per-frame extraction specification."""

    duration_s: float = 60.0
    overlap: float = 0.25
    target_sr: int = 8000  # downsampled to keep RAM small; paper uses native sr


def _bilinear_resize_2d(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize a 2-D array to ``size`` using torch's bilinear interp (CPU)."""
    tensor = torch.from_numpy(image.astype(np.float32))[None, None]
    out = torch.nn.functional.interpolate(
        tensor, size=size, mode="bilinear", align_corners=False
    )
    return out[0, 0].numpy()


def _frame_to_spectrogram(
    frame: np.ndarray, sr: int, method: str, image_size: int = 224
) -> np.ndarray:
    """Return a ``(3, image_size, image_size)`` float32 spectrogram image."""
    if method == "sav":
        result = compute_sav(frame, sr=sr)
        spec = result.spectrogram
    elif method == "demon":
        # DEMON band must lie strictly below the Nyquist of the downsampled
        # signal, otherwise the band-pass design fails. Cap to 0.45 * sr.
        nyq = 0.5 * sr
        high = min(nyq * 0.9, 10_000.0)
        low = min(high * 0.5, 1_000.0)
        result = compute_demon(frame, sr=sr, band=(low, high))
        spec = result.spectrogram
    elif method == "stft":
        from scipy.signal import stft

        _, _, zxx = stft(frame, fs=sr, nperseg=512, noverlap=256)
        spec = np.abs(zxx)
    else:
        raise ValueError(f"Unknown spectrogram method: {method}")

    # log compression + bilinear resize + per-channel duplicate to 3xHxW
    spec = np.log1p(np.abs(spec))
    if spec.shape[0] < 2 or spec.shape[1] < 2:
        spec = np.pad(spec, ((0, max(0, 2 - spec.shape[0])), (0, max(0, 2 - spec.shape[1]))))
    spec = _bilinear_resize_2d(spec, (image_size, image_size))
    # standardise per-image
    mean, std = float(spec.mean()), float(spec.std() + 1e-8)
    spec = (spec - mean) / std
    return np.stack([spec, spec, spec], axis=0).astype(np.float32)


def _extract_frames(signal: np.ndarray, sr: int, spec: FrameSpec) -> list[np.ndarray]:
    frame_len = int(spec.duration_s * sr)
    if signal.size < frame_len:
        # zero-pad short clips so each one still produces a single frame
        signal = np.concatenate(
            [signal, np.zeros(frame_len - signal.size, dtype=signal.dtype)]
        )
    step = max(1, int(frame_len * (1.0 - spec.overlap)))
    frames = []
    for start in range(0, signal.size - frame_len + 1, step):
        f = signal[start : start + frame_len].astype(np.float64)
        f -= f.mean()  # DC removal
        std = f.std()
        if std > 1e-12:
            f /= std  # unit variance per frame
        frames.append(f)
    if not frames:  # very short clip safety net
        f = signal.astype(np.float64)
        f -= f.mean()
        std = f.std()
        if std > 1e-12:
            f /= std
        frames.append(f)
    return frames


def _generate_synthetic_signal(
    rng: np.random.Generator,
    class_idx: int,
    duration_s: float,
    sr: int,
    tonal_sets: list[tuple[float, ...]],
    background_class: int | None,
) -> np.ndarray:
    """Generate a propeller-noise-like signal for a class.

    Each non-background class is the sum of its base tonal frequencies (with
    a few harmonics each) plus pink-ish coloured noise. The background class
    is pure noise. The class-specific tonals make the classification problem
    learnable and the SAV/DEMON detectors meaningful.
    """
    n = int(duration_s * sr)
    t = np.arange(n) / sr
    signal = np.zeros(n, dtype=np.float64)

    if background_class is None or class_idx != background_class:
        for base in tonal_sets[class_idx]:
            # Each tonal is amplitude-modulated by a low-frequency carrier
            # (mimicking propeller cavitation modulation). This is what makes
            # SAV's temporal-variance detector meaningful: a steady sine has
            # zero STFT-power variance, but a modulated sine has a clear bump.
            phase = rng.uniform(0, 2 * np.pi)
            mod_freq = 0.3 + 0.8 * rng.random()  # Hz — slow drift
            mod_depth = 0.6 + 0.3 * rng.random()
            envelope = 1.0 + mod_depth * np.sin(2 * np.pi * mod_freq * t + rng.uniform(0, 2 * np.pi))
            for k in range(1, 4):
                amp = 1.0 / k * (0.7 + 0.3 * rng.random())
                signal += amp * envelope * np.sin(2 * np.pi * base * k * t + phase)

    # Coloured noise (pink-ish): cumulative sum of white noise, then HP filter.
    noise = rng.standard_normal(n)
    noise = np.cumsum(noise) / np.sqrt(np.arange(1, n + 1))
    noise_amp = 1.5 if (background_class is not None and class_idx == background_class) else 0.5
    signal += noise_amp * noise

    return signal


def _load_real_audio(root: Path, classes: tuple[str, ...]) -> list[tuple[np.ndarray, int, int]]:
    """Return list of ``(signal, sr, class_idx)`` from .wav files under root.

    Files must be organised as ``root/<class_name>/*.wav``.
    """
    try:
        from scipy.io import wavfile
    except ImportError as exc:  # pragma: no cover - scipy is a hard dep
        raise ImportError("scipy.io.wavfile is required for real audio loading") from exc

    items: list[tuple[np.ndarray, int, int]] = []
    for class_idx, class_name in enumerate(classes):
        class_dir = root / class_name
        if not class_dir.exists():
            continue
        for wav in sorted(class_dir.glob("*.wav")):
            sr, data = wavfile.read(wav)
            if data.ndim == 2:
                data = data.mean(axis=1)
            signal = data.astype(np.float64) / max(1.0, float(np.abs(data).max()))
            items.append((signal, int(sr), class_idx))
    return items


def ensure_deepship_github_sample(data_root: Path) -> Path:
    """Download the small public DeepShip GitHub sample into ``data_root``.

    The full DeepShip dataset is not bundled with this reproduction. This
    helper fetches only the tiny subset used for the CPU sanity runs, and only
    when explicitly requested by config or CLI.
    """
    dataset_dir = Path(data_root) / "deepship"
    for class_name, (source_dir, filenames) in DEEPSHIP_GITHUB_SAMPLE.items():
        target_dir = dataset_dir / class_name
        target_dir.mkdir(parents=True, exist_ok=True)
        for filename in filenames:
            target_path = target_dir / filename
            if target_path.exists():
                continue
            url = f"{DEEPSHIP_GITHUB_BASE_URL}/{source_dir}/{filename}"
            tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")
            logger.info("Downloading DeepShip sample %s -> %s", url, target_path)
            try:
                urlretrieve(url, tmp_path)
            except URLError as exc:
                if tmp_path.exists():
                    tmp_path.unlink()
                raise RuntimeError(
                    "Failed to download the DeepShip GitHub sample. "
                    "Retry with network access or place .wav files manually "
                    f"under {dataset_dir}."
                ) from exc
            tmp_path.replace(target_path)
    return dataset_dir


def _resample_naive(signal: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Linear-interpolation resample (good enough for log-magnitude features)."""
    if src_sr == dst_sr:
        return signal
    n_dst = int(round(signal.size * dst_sr / src_sr))
    if n_dst <= 0:
        return signal
    x_src = np.linspace(0.0, 1.0, signal.size, endpoint=False)
    x_dst = np.linspace(0.0, 1.0, n_dst, endpoint=False)
    return np.interp(x_dst, x_src, signal)


def build_dataset(
    dataset_name: str,
    data_root: Path,
    spectrogram_method: str,
    frame_spec: FrameSpec,
    image_size: int = 224,
    samples_per_class_synthetic: int = 32,
    duration_synthetic_s: float = 60.0,
    seed: int = 1337,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build ``(images, labels, class_names)`` arrays for a dataset.

    Falls back to a synthetic substitute when no real .wav files are found
    under ``data_root/<dataset_name>/<class>/``. This keeps the pipeline
    runnable in environments where the original datasets cannot be obtained.

    Parameters
    ----------
    dataset_name : str
        ``"shipear"`` or ``"deepship"``.
    data_root : Path
        Root data directory (typically ``data/qml_passive_sonar``).
    spectrogram_method : str
        ``"sav"``, ``"demon"``, or ``"stft"``.
    frame_spec : FrameSpec
        Frame-extraction parameters.
    image_size : int
        Square image side length. Default ``224``.
    samples_per_class_synthetic : int
        Number of synthetic clips per class to fabricate when no real data
        is found. Default ``32``.
    duration_synthetic_s : float
        Duration of each synthetic clip in seconds.
    seed : int
        RNG seed for the synthetic fallback.

    Returns
    -------
    images : np.ndarray
        Shape ``(N, 3, image_size, image_size)`` float32.
    labels : np.ndarray
        Shape ``(N,)`` int64.
    class_names : list[str]
        Ordered list of class names.
    """
    if dataset_name == "shipear":
        classes = SHIPEAR_CLASSES
        # Distinct tonals per class (Hz). Background class E is pure noise.
        tonal_sets = [
            (47.0, 91.0),
            (110.0, 175.0),
            (63.0, 130.0, 220.0),
            (38.0, 80.0, 150.0, 305.0),
            (),
        ]
        background_class: int | None = 4
    elif dataset_name == "deepship":
        classes = DEEPSHIP_CLASSES
        tonal_sets = [
            (52.0, 100.0, 190.0),
            (120.0, 240.0),
            (40.0, 78.0, 160.0),
            (90.0, 180.0, 270.0),
        ]
        background_class = None
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    dataset_dir = Path(data_root) / dataset_name
    real_items = _load_real_audio(dataset_dir, classes) if dataset_dir.exists() else []

    if real_items:
        logger.info("Loaded %d real audio recordings from %s", len(real_items), dataset_dir)
    else:
        logger.warning(
            "No real audio found under %s — falling back to synthetic data. "
            "Drop ShipsEar/DeepShip .wav files there to use the real dataset.",
            dataset_dir,
        )
        rng = np.random.default_rng(seed)
        real_items = []
        for class_idx, _ in enumerate(classes):
            for _ in range(samples_per_class_synthetic):
                signal = _generate_synthetic_signal(
                    rng,
                    class_idx,
                    duration_synthetic_s,
                    frame_spec.target_sr,
                    list(tonal_sets),
                    background_class,
                )
                real_items.append((signal, frame_spec.target_sr, class_idx))

    images: list[np.ndarray] = []
    labels: list[int] = []
    for signal, src_sr, class_idx in real_items:
        signal_ds = _resample_naive(signal, src_sr, frame_spec.target_sr)
        for frame in _extract_frames(signal_ds, frame_spec.target_sr, frame_spec):
            images.append(
                _frame_to_spectrogram(
                    frame,
                    frame_spec.target_sr,
                    method=spectrogram_method,
                    image_size=image_size,
                )
            )
            labels.append(class_idx)

    return (
        np.stack(images, axis=0) if images else np.zeros((0, 3, image_size, image_size), dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
        list(classes),
    )


def split_dataset(
    images: np.ndarray,
    labels: np.ndarray,
    test_fraction: float = 0.3,
    seed: int = 1337,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stratified train/test split.

    This is a simple stratified split — when real data is supplied, the
    paper-aligned source-stratified split is the user's responsibility (split
    at the recording/vessel level before calling ``build_dataset``). The
    runner reports this caveat in the run log.
    """
    rng = np.random.default_rng(seed)
    train_idx, test_idx = [], []
    for class_idx in np.unique(labels):
        idx = np.where(labels == class_idx)[0]
        rng.shuffle(idx)
        n_test = max(1, int(round(len(idx) * test_fraction)))
        test_idx.extend(idx[:n_test].tolist())
        train_idx.extend(idx[n_test:].tolist())
    train_idx = np.asarray(train_idx, dtype=np.int64)
    test_idx = np.asarray(test_idx, dtype=np.int64)
    return images[train_idx], labels[train_idx], images[test_idx], labels[test_idx]


def make_dataloaders(
    images_train: np.ndarray,
    labels_train: np.ndarray,
    images_test: np.ndarray,
    labels_test: np.ndarray,
    batch_size: int,
) -> tuple[DataLoader, DataLoader]:
    train = TensorDataset(
        torch.from_numpy(images_train), torch.from_numpy(labels_train)
    )
    test = TensorDataset(
        torch.from_numpy(images_test), torch.from_numpy(labels_test)
    )
    # drop_last on the train loader prevents BatchNorm from seeing a
    # single-sample tail batch in training mode (which raises).
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True),
        DataLoader(test, batch_size=batch_size, shuffle=False),
    )
