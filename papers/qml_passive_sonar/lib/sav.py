"""Spectral Amplitude Variation (SAV) preprocessing.

Implements the algorithm from Bach & Nguyen (Ocean Eng., 2025), Sec. 3 and
Eqs. (4) and (13). The detector computes the temporal *variance* of the STFT
power across short overlapping frames inside a long segment, then averages
those variance estimates over ``L`` consecutive segments (the "stacking"
step). A local-maxima search with adaptive threshold
``gamma = mu_V + eta * sigma_V`` returns the detected tonal frequencies.

The paper-recommended parameters are: ``T_s = 45 s``, ``rho = 1/3``,
``Delta_f = 1 Hz``, ``eta = 3.0``, ``L = 4``, Blackman window.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import get_window, stft


@dataclass
class SAVResult:
    """Container for SAV outputs.

    Attributes
    ----------
    frequencies : np.ndarray
        Frequency bins in Hz, shape ``(F,)``.
    v_stack : np.ndarray
        Stacked variance estimate ``V_stack(f)``, shape ``(F,)``.
    threshold : float
        Adaptive detection threshold ``gamma``.
    peak_indices : np.ndarray
        Indices into ``frequencies`` for detected tonal peaks.
    spectrogram : np.ndarray
        Raw STFT power spectrogram ``P(f, t)`` (last segment), shape
        ``(F, T)`` — useful as input to the CNN backbone.
    """

    frequencies: np.ndarray
    v_stack: np.ndarray
    threshold: float
    peak_indices: np.ndarray
    spectrogram: np.ndarray


def _stft_power(signal, sr, freq_resolution, window_name):
    nperseg = int(round(sr / freq_resolution))
    if nperseg < 8:
        nperseg = 8
    window = get_window(window_name, nperseg, fftbins=True)
    noverlap = nperseg // 2  # 50 % overlap for the inner STFT
    f, _, zxx = stft(
        signal,
        fs=sr,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        boundary=None,
        padded=False,
    )
    power = np.abs(zxx) ** 2
    return f, power


def _segment_indices(signal_length, sr, segment_duration, overlap):
    """Return (start, stop) sample indices for each long segment."""
    seg_samples = int(round(segment_duration * sr))
    if seg_samples > signal_length:
        # Fall back to a single segment covering the whole signal.
        return [(0, signal_length)]
    step = max(1, int(round(seg_samples * (1.0 - overlap))))
    indices = []
    start = 0
    while start + seg_samples <= signal_length:
        indices.append((start, start + seg_samples))
        start += step
    if not indices:
        indices.append((0, seg_samples))
    return indices


def compute_sav(
    signal,
    sr,
    segment_duration: float = 45.0,
    overlap: float = 1.0 / 3.0,
    freq_resolution: float = 1.0,
    stacking: int = 4,
    window: str = "blackman",
    threshold_eta: float = 3.0,
) -> SAVResult:
    """Compute the SAV detection statistic for a 1-D acoustic signal.

    Parameters
    ----------
    signal : array-like
        Mono time-domain signal.
    sr : int
        Sampling rate in Hz.
    segment_duration : float
        ``T_s`` in seconds. Default ``45.0``.
    overlap : float
        Fractional segment overlap ``rho``. Default ``1/3``.
    freq_resolution : float
        Desired STFT frequency resolution ``Delta_f`` (Hz). Default ``1.0``.
    stacking : int
        Number of consecutive variance estimates to average (``L``).
        Default ``4``.
    window : str
        Name passed to ``scipy.signal.get_window``. Default ``"blackman"``.
    threshold_eta : float
        Adaptive-threshold scaling ``eta``. Default ``3.0``.

    Returns
    -------
    SAVResult
    """
    signal = np.asarray(signal, dtype=np.float64).ravel()
    if signal.size == 0:
        raise ValueError("Empty signal passed to compute_sav")

    seg_bounds = _segment_indices(signal.size, sr, segment_duration, overlap)

    # Eq. (4): per-segment temporal variance of STFT power.
    variance_estimates = []
    last_power = None
    last_freq = None
    for start, stop in seg_bounds:
        freq, power = _stft_power(signal[start:stop], sr, freq_resolution, window)
        # variance across the time axis for each frequency bin
        variance_estimates.append(power.var(axis=1))
        last_power = power
        last_freq = freq

    variance_estimates = np.stack(variance_estimates, axis=0)  # (M, F)

    # Eq. (13): stack L variance estimates and average.
    if stacking > 1 and variance_estimates.shape[0] >= stacking:
        n_full = (variance_estimates.shape[0] // stacking) * stacking
        v_stack = (
            variance_estimates[:n_full]
            .reshape(-1, stacking, variance_estimates.shape[1])
            .mean(axis=1)
            .mean(axis=0)
        )
    else:
        v_stack = variance_estimates.mean(axis=0)

    mu = float(v_stack.mean())
    sigma = float(v_stack.std())
    threshold = mu + threshold_eta * sigma

    peaks = _local_maxima(v_stack, threshold)

    return SAVResult(
        frequencies=last_freq,
        v_stack=v_stack,
        threshold=threshold,
        peak_indices=np.asarray(peaks, dtype=np.int64),
        spectrogram=last_power,
    )


def _local_maxima(values, threshold):
    """Return indices where ``values`` is a strict local maximum and above threshold."""
    idx = []
    n = values.size
    for i in range(1, n - 1):
        v = values[i]
        if v > threshold and v > values[i - 1] and v > values[i + 1]:
            idx.append(i)
    return idx
