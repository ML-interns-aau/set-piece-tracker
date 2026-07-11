"""Homography application helper (calibration.apply_homography)."""

from __future__ import annotations

import numpy as np

from src.geometry.calibration import apply_homography

# a fixed, mildly-perspective homography that maps pixel -> metre
H_TRUE = np.array(
    [[0.045, 0.0020, -6.0],
     [0.0018, 0.050, -4.0],
     [0.00012, 0.00021, 1.0]],
    dtype=np.float64,
)


def test_apply_homography_matches_manual_projection():
    pts = np.array([[123.0, 456.0], [800.0, 200.0], [10.0, 10.0]])
    got = apply_homography(H_TRUE, pts)
    for (u, v), (x, y) in zip(pts, got):
        p = H_TRUE @ np.array([u, v, 1.0])
        np.testing.assert_allclose([x, y], p[:2] / p[2], atol=1e-9)


def test_apply_homography_round_trips_through_inverse():
    pts = np.array([[123.0, 456.0], [640.0, 360.0]])
    metric = apply_homography(H_TRUE, pts)
    back = apply_homography(np.linalg.inv(H_TRUE), metric)
    np.testing.assert_allclose(back, pts, atol=1e-6)


def test_apply_homography_empty_input():
    out = apply_homography(H_TRUE, np.empty((0, 2)))
    assert out.shape == (0, 2)
