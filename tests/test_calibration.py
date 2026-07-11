"""Task 2 - penalty-area calibration from named FIFA points."""

from __future__ import annotations

import numpy as np
import pytest

from src.domain.models import Source
from src.domain.pitch import reference_points_metric
from src.geometry.calibration import (
    apply_homography,
    calibrate_from_markings,
    reprojection_error,
    solve_homography,
    solve_homography_hybrid,
)

NAMES = [
    "near_post", "far_post",
    "goal_area_front_left", "goal_area_front_right",
    "pen_area_front_left", "pen_area_front_right",
]

# a fixed, mildly-perspective homography that maps pixel -> metre
H_TRUE = np.array(
    [[0.045, 0.0020, -6.0],
     [0.0018, 0.050, -4.0],
     [0.00012, 0.00021, 1.0]],
    dtype=np.float64,
)


def _pixels_for(names: list[str]) -> np.ndarray:
    """Pixel coords that H_TRUE maps to the metric reference points."""
    metric = reference_points_metric(names)
    return apply_homography(np.linalg.inv(H_TRUE), metric)


def test_solver_recovers_known_homography():
    pixels = _pixels_for(NAMES)
    metric = reference_points_metric(NAMES)
    h = solve_homography(pixels, metric)

    # a brand-new pixel point maps the same way under the recovered and true H
    probe = np.array([[123.0, 456.0]])
    np.testing.assert_allclose(
        apply_homography(h, probe), apply_homography(H_TRUE, probe), atol=1e-6
    )


def test_calibrate_from_markings_is_near_exact():
    pixels = _pixels_for(NAMES)
    points = {n: tuple(px) for n, px in zip(NAMES, pixels)}
    calib = calibrate_from_markings(points)

    assert calib.points_used == len(NAMES)
    assert calib.source == Source.AUTO
    assert calib.reprojection_error_m < 1e-6
    # maps pixels back to their metric targets
    np.testing.assert_allclose(
        apply_homography(calib.H, pixels), reference_points_metric(NAMES), atol=1e-6
    )


def test_hybrid_solver_recovers_homography_from_lines_and_points():
    """Wall-clear features recover H: goalposts + arc apex + 3 longitudinal lines.

    Goalposts alone are collinear (both on x=0), so the arc apex (20.15, 34), which
    lies off the goal line, supplies the missing cross-direction constraint.
    """
    Hinv = np.linalg.inv(H_TRUE)

    points_m = np.array([[0.0, 30.34], [0.0, 37.66], [20.15, 34.0]])  # posts + arc apex
    points_px = apply_homography(Hinv, points_m)

    line_x = [0.0, 5.5, 16.5]
    segs_px = np.array([
        apply_homography(Hinv, np.array([[c, 10.0], [c, 58.0]])) for c in line_x
    ])

    h = solve_homography_hybrid(points_px, points_m, segs_px, np.array(line_x))

    probe = np.array([[321.0, 210.0], [800.0, 400.0]])
    np.testing.assert_allclose(
        apply_homography(h, probe), apply_homography(H_TRUE, probe), atol=1e-5
    )


def test_hybrid_solver_rejects_underconstrained():
    """Two collinear points + fewer than the needed lines is refused."""
    Hinv = np.linalg.inv(H_TRUE)
    posts_m = np.array([[0.0, 30.34], [0.0, 37.66]])
    posts_px = apply_homography(Hinv, posts_m)
    segs_px = np.array([apply_homography(Hinv, np.array([[0.0, 10.0], [0.0, 58.0]]))])
    with pytest.raises(ValueError):
        solve_homography_hybrid(posts_px, posts_m, segs_px, np.array([0.0]))


def test_reprojection_error_grows_with_noise():
    rng = np.random.default_rng(0)
    pixels = _pixels_for(NAMES)
    noisy = pixels + rng.normal(0, 1.5, pixels.shape)   # ~1.5 px jitter
    metric = reference_points_metric(NAMES)
    h = solve_homography(noisy, metric)
    err = reprojection_error(h, noisy, metric)
    assert 0.0 < err < 2.0   # small but non-zero (metres)


def test_too_few_points_raises():
    pixels = _pixels_for(NAMES[:3])
    points = {n: tuple(px) for n, px in zip(NAMES[:3], pixels)}
    with pytest.raises(ValueError):
        calibrate_from_markings(points)


def test_unknown_marking_raises():
    with pytest.raises(KeyError):
        calibrate_from_markings({"not_a_marking": (1.0, 2.0), "near_post": (3.0, 4.0)})
