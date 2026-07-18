"""Pure-numpy signal primitives: Savitzky-Golay smoothing and peak finding.

These are minimal re-implementations of the two ``scipy.signal`` functions the
ELASTIC event-detection algorithm relies on (``savgol_filter`` and
``find_peaks`` with prominence/distance), so the geometry & moments plane stays
importable and unit-testable with numpy alone. Behaviour matches scipy for the
subset of options used by ``geometry.key_moments``:

- ``savgol_filter(x, window_length, polyorder)`` with scipy's default
  ``mode='interp'`` edge handling (fit a polynomial to the first/last window
  and evaluate it at the edge positions).
- ``find_peaks(x, prominence=..., distance=...)`` — local maxima (plateau
  midpoint), distance condition applied before prominence, scipy's
  prominence definition.
"""

from __future__ import annotations

import numpy as np


def savgol_filter(x: np.ndarray, window_length: int, polyorder: int) -> np.ndarray:
    """Savitzky-Golay smoothing, matching ``scipy.signal.savgol_filter`` defaults.

    ``window_length`` must be odd and > ``polyorder``; it is clamped to the
    signal length (largest odd value <= len(x)). Signals shorter than
    ``polyorder + 1`` are returned unchanged.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n = len(x)
    if window_length % 2 == 0:
        raise ValueError("window_length must be odd")
    if window_length > n:
        window_length = n if n % 2 == 1 else n - 1
    if window_length <= polyorder or n <= polyorder:
        return x.copy()

    half = window_length // 2
    # Projection weights for the window centre: value of the LSQ-fit polynomial
    # at offset 0 is row 0 of pinv(A), A_ij = offset_i ** j.
    offsets = np.arange(-half, half + 1, dtype=np.float64)
    A = np.vander(offsets, polyorder + 1, increasing=True)
    center_weights = np.linalg.pinv(A)[0]

    out = np.convolve(x, center_weights[::-1], mode="same")

    # mode='interp': fit a polynomial to the first/last full window and
    # evaluate it at the edge sample positions.
    head_coeffs = np.linalg.pinv(A) @ x[:window_length]
    tail_coeffs = np.linalg.pinv(A) @ x[n - window_length:]
    edge = np.arange(half, dtype=np.float64)
    out[:half] = np.vander(edge - half, polyorder + 1, increasing=True) @ head_coeffs
    out[n - half:] = np.vander(edge + 1, polyorder + 1, increasing=True) @ tail_coeffs
    return out


def _local_maxima(x: np.ndarray) -> np.ndarray:
    """Indices of local maxima, plateau resolved to its midpoint (scipy rule)."""
    peaks = []
    i, n = 1, len(x)
    while i < n - 1:
        if x[i - 1] < x[i]:
            i_ahead = i + 1
            while i_ahead < n - 1 and x[i_ahead] == x[i]:
                i_ahead += 1
            if x[i_ahead] < x[i]:
                peaks.append((i + i_ahead - 1) // 2)
                i = i_ahead
                continue
        i += 1
    return np.array(peaks, dtype=np.intp)


def peak_prominences(x: np.ndarray, peaks: np.ndarray) -> np.ndarray:
    """Prominence of each peak, per scipy's definition.

    Extend left/right from the peak until the signal exceeds the peak height
    (or the end is reached); the base on each side is the minimum over that
    stretch; prominence = peak height - max(left base, right base).
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    proms = np.empty(len(peaks), dtype=np.float64)
    for k, p in enumerate(peaks):
        left_min = x[p]
        i = p - 1
        while i >= 0 and x[i] <= x[p]:
            left_min = min(left_min, x[i])
            i -= 1
        right_min = x[p]
        i = p + 1
        while i < len(x) and x[i] <= x[p]:
            right_min = min(right_min, x[i])
            i += 1
        proms[k] = x[p] - max(left_min, right_min)
    return proms


def find_peaks(
    x: np.ndarray,
    prominence: float | None = None,
    distance: int | None = None,
) -> np.ndarray:
    """Local-maxima indices filtered by ``distance`` then ``prominence`` (scipy order)."""
    x = np.asarray(x, dtype=np.float64).ravel()
    peaks = _local_maxima(x)
    if distance is not None and len(peaks) > 1:
        # Keep higher peaks first; drop any peak within `distance` of a kept one.
        keep = np.ones(len(peaks), dtype=bool)
        priority = np.argsort(x[peaks])[::-1]
        for idx in priority:
            if not keep[idx]:
                continue
            too_close = np.abs(peaks - peaks[idx]) < distance
            too_close[idx] = False
            keep &= ~too_close
        peaks = peaks[keep]
    if prominence is not None and len(peaks) > 0:
        peaks = peaks[peak_prominences(x, peaks) >= prominence]
    return peaks
