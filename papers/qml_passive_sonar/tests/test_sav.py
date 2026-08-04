"""Algorithmic checks for SAV and DEMON on synthetic tonal signals."""

from __future__ import annotations

import numpy as np
from lib.demon import compute_demon
from lib.sav import compute_sav


def _tonal_signal(freqs, duration=4.0, sr=4000, noise=0.2, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(duration * sr)) / sr
    signal = np.zeros_like(t)
    for f in freqs:
        signal += np.sin(2 * np.pi * f * t)
    signal += noise * rng.standard_normal(t.size)
    return signal, sr


def test_sav_detects_known_tonals():
    freqs = [120.0, 311.0]
    signal, sr = _tonal_signal(freqs, duration=8.0, sr=4000, noise=0.5)
    result = compute_sav(signal, sr=sr, segment_duration=2.0, stacking=2)
    detected = result.frequencies[result.peak_indices]
    # Every expected tonal should sit within +/- 3 Hz of a detected peak.
    for f in freqs:
        assert detected.size > 0
        assert np.min(np.abs(detected - f)) < 3.0


def test_sav_threshold_above_mean():
    signal, sr = _tonal_signal([100.0], duration=4.0, sr=4000)
    result = compute_sav(signal, sr=sr, segment_duration=2.0)
    assert result.threshold >= result.v_stack.mean()


def test_demon_runs_and_produces_spectrum():
    signal, sr = _tonal_signal([50.0], duration=4.0, sr=4000, noise=0.5)
    result = compute_demon(signal, sr=sr, band=(20.0, 1500.0), window_duration=0.5)
    assert result.spectrum.size > 0
    assert np.isfinite(result.threshold)
