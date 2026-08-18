"""DEMON (Detection of Envelope Modulation On Noise) baseline.

Classical band-pass → Hilbert envelope → spectrum pipeline used in the paper
as the comparison method against SAV.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, get_window, hilbert, sosfiltfilt


@dataclass
class DEMONResult:
    """Container for DEMON outputs."""

    frequencies: np.ndarray
    spectrum: np.ndarray
    threshold: float
    peak_indices: np.ndarray
    spectrogram: np.ndarray  # envelope spectrogram (F, T)


def _bandpass(signal, sr, low_hz, high_hz, order=4):
    nyq = 0.5 * sr
    low = max(1e-6, low_hz / nyq)
    high = min(0.999, high_hz / nyq)
    sos = butter(order, [low, high], btype="bandpass", output="sos")
    return sosfiltfilt(sos, signal)


def compute_demon(
    signal,
    sr,
    band: tuple[float, float] = (1000.0, 10000.0),
    window_duration: float = 1.0,
    overlap: float = 0.5,
    window: str = "hann",
    threshold_eta: float = 3.0,
) -> DEMONResult:
    """Compute the DEMON envelope spectrum of a 1-D acoustic signal.

    Parameters
    ----------
    signal : array-like
        Mono time-domain signal.
    sr : int
        Sampling rate (Hz).
    band : tuple[float, float]
        Pre-envelope band-pass corners. Default ``(1 kHz, 10 kHz)`` per the
        paper's ShipsEar setup; use ``(1 kHz, 8 kHz)`` for DeepShip.
    window_duration : float
        Analysis window length (seconds) for the envelope DFT.
    overlap : float
        Fractional window overlap (50 % per the paper).
    window : str
        Window type passed to ``scipy.signal.get_window``.
    threshold_eta : float
        Adaptive-threshold scaling factor.

    Returns
    -------
    DEMONResult
    """
    signal = np.asarray(signal, dtype=np.float64).ravel()
    if signal.size == 0:
        raise ValueError("Empty signal passed to compute_demon")

    filtered = _bandpass(signal, sr, band[0], band[1])
    envelope = np.abs(hilbert(filtered))
    envelope = envelope - envelope.mean()

    nperseg = max(8, int(round(window_duration * sr)))
    if nperseg > envelope.size:
        nperseg = envelope.size
    step = max(1, int(round(nperseg * (1.0 - overlap))))
    win = get_window(window, nperseg, fftbins=True)
    win_norm = np.sum(win**2)

    starts = list(range(0, envelope.size - nperseg + 1, step))
    if not starts:
        starts = [0]
        if nperseg > envelope.size:
            nperseg = envelope.size
            win = get_window(window, nperseg, fftbins=True)
            win_norm = np.sum(win**2)

    spectrogram = []
    for s in starts:
        seg = envelope[s : s + nperseg] * win
        spec = np.abs(np.fft.rfft(seg)) ** 2 / win_norm
        spectrogram.append(spec)
    spectrogram = np.stack(spectrogram, axis=1)  # (F, T)
    freqs = np.fft.rfftfreq(nperseg, d=1.0 / sr)
    spectrum = spectrogram.mean(axis=1)  # RMS-like power average

    threshold = float(spectrum.mean() + threshold_eta * spectrum.std())
    peaks = []
    for i in range(1, spectrum.size - 1):
        v = spectrum[i]
        if v > threshold and v > spectrum[i - 1] and v > spectrum[i + 1]:
            peaks.append(i)

    return DEMONResult(
        frequencies=freqs,
        spectrum=spectrum,
        threshold=threshold,
        peak_indices=np.asarray(peaks, dtype=np.int64),
        spectrogram=spectrogram,
    )
