"""Task 1 - orientation normalization (pure) and corner-arc detection (cv2)."""

from __future__ import annotations

import numpy as np

from src.domain.models import CornerSide
from src.domain.pitch import REFERENCE_POINTS
from src.geometry.orientation import (
    detect_corner_side,
    side_from_metric_corner,
    to_canonical,
)


def test_left_is_identity():
    pts = np.array([REFERENCE_POINTS["near_post"], REFERENCE_POINTS["far_post"]])
    np.testing.assert_allclose(to_canonical(pts, CornerSide.LEFT), pts)


def test_right_mirrors_near_and_far_post():
    near = REFERENCE_POINTS["near_post"]   # (0, 30.34)
    far = REFERENCE_POINTS["far_post"]     # (0, 37.66)
    out = to_canonical(np.array([near, far]), CornerSide.RIGHT)
    # mirroring swaps near/far y across the mid-line (34)
    np.testing.assert_allclose(out[0], far, atol=1e-9)
    np.testing.assert_allclose(out[1], near, atol=1e-9)


def test_side_from_metric_corner():
    assert side_from_metric_corner(1.5) == CornerSide.LEFT
    assert side_from_metric_corner(66.0) == CornerSide.RIGHT


def test_detect_corner_side_on_synthetic_arc():
    import cv2

    # black frame with a clear white arc (circle outline) in the LEFT edge band
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    cv2.circle(frame, (70, 300), 28, (255, 255, 255), 2)

    result = detect_corner_side(frame)
    assert result.corner_side == CornerSide.LEFT
    assert result.confidence > 0.0
