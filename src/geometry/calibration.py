"""Homography application: map pixel points through a calibration to metric coords.

The calibration itself (the pixel -> metre homography) is produced by the PnLCalib
model (see ``pnl_calibration``). This module holds the small, pure helper that
*applies* a homography to pixel points -- used to push ball/foot points onto the
metric pitch. It stays dependency-free (pure numpy) so it unit-tests without cv2.
"""

from __future__ import annotations

import numpy as np


def apply_homography(h: np.ndarray, points_px: np.ndarray) -> np.ndarray:
    """Map pixel points (N, 2) through ``h`` to metric points (N, 2)."""
    pts = np.asarray(points_px, dtype=np.float64).reshape(-1, 2)
    if len(pts) == 0:
        return pts
    homog = np.c_[pts, np.ones(len(pts))]              # (N, 3)
    mapped = (h @ homog.T).T                            # (N, 3)
    w = mapped[:, 2:3]
    w[np.abs(w) < 1e-12] = 1e-12
    return mapped[:, :2] / w
