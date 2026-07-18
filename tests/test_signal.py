"""Pure-numpy signal primitives (savgol_filter / find_peaks) vs known results.

The implementations were cross-validated against scipy.signal (200 random
savgol cases, 2500 find_peaks cases, exact agreement); these tests pin the
behaviour with scipy-free analytic and hand-computed cases.
"""

from __future__ import annotations

import numpy as np

from src.geometry.signal import find_peaks, peak_prominences, savgol_filter


def test_savgol_reproduces_polynomial_exactly():
    # A degree-2 polynomial is invariant under a polyorder-2 Savitzky-Golay fit,
    # including the mode='interp' edges.
    t = np.linspace(0, 4, 40)
    x = 3.0 * t**2 - 2.0 * t + 1.0
    for window in (5, 9, 15):
        np.testing.assert_allclose(savgol_filter(x, window, 2), x, atol=1e-9)


def test_savgol_smooths_noise():
    rng = np.random.default_rng(0)
    x = np.ones(100) + rng.normal(0, 1.0, 100)
    sm = savgol_filter(x, 15, 2)
    assert sm.std() < 0.5 * x.std()


def test_savgol_window_clamps_to_short_signal():
    x = np.array([0.0, 1.0, 4.0, 9.0, 16.0])  # t^2 on 5 samples
    np.testing.assert_allclose(savgol_filter(x, 15, 2), x, atol=1e-9)


def test_find_peaks_basic_and_prominence():
    x = np.array([0, 2, 0, 5, 0, 1, 0], dtype=float)
    np.testing.assert_array_equal(find_peaks(x), [1, 3, 5])
    np.testing.assert_array_equal(find_peaks(x, prominence=1.5), [1, 3])
    np.testing.assert_array_equal(find_peaks(x, prominence=3.0), [3])


def test_find_peaks_plateau_midpoint():
    x = np.array([0, 1, 3, 3, 3, 1, 0, 5, 5, 0, 2, 0], dtype=float)
    # scipy resolves a plateau to its midpoint: [3,3,3] -> 3, [5,5] -> 7
    np.testing.assert_array_equal(find_peaks(x, prominence=0.1), [3, 7, 10])


def test_find_peaks_distance_keeps_higher_peak():
    x = np.zeros(16)
    x[1], x[3], x[13] = 4.0, 6.0, 3.0
    # peaks at 1, 3, 13; distance=10 drops 1 (within 10 of the higher peak 3)
    # but keeps 13 (|13 - 3| = 10, not < 10)
    np.testing.assert_array_equal(find_peaks(x, distance=10), [3, 13])


def test_peak_prominence_values():
    x = np.array([0, 2, 1, 5, 1, 3, 0], dtype=float)
    peaks = find_peaks(x)
    np.testing.assert_array_equal(peaks, [1, 3, 5])
    # peak 3 is the global max: prominence = height - max(edge minima) = 5 - 0
    proms = peak_prominences(x, peaks)
    np.testing.assert_allclose(proms, [1.0, 5.0, 2.0])
