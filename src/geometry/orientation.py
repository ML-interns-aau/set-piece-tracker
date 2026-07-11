"""Corner-side detection and orientation normalization (FR-007).

Two responsibilities:

1. **Detect** which side the corner is taken from, from the distinctive corner
   arc near a frame edge at clip start (best-effort, cv2; thresholds are
   pilot-tunable and a manual override is expected for ambiguous clips).
2. **Normalize** all metric positions into one canonical orientation so that
   "near post" / "far post" mean the same geometric thing across every clip.

Canonical frame (see ``domain.pitch``): the corner is taken from the y = 0 side
(``CornerSide.LEFT``); the near post is (0, 30.34). A clip whose corner is on
the y = 68 side (``CornerSide.RIGHT``) is mirrored (y -> width - y).

The normalization functions are pure and unit-tested; only ``detect_corner_side``
touches OpenCV.
"""

from __future__ import annotations

import numpy as np

from src.domain.models import CornerSide, CornerSideResult, Source
from src.domain.pitch import PITCH_WIDTH_M, mirror_y


# --- pure: orientation normalization ----------------------------------------
def to_canonical(points_m: np.ndarray, corner_side: CornerSide) -> np.ndarray:
    """Bring metric points into the canonical (LEFT / y=0) orientation.

    RIGHT-side corners are mirrored across the pitch mid-line; LEFT is identity.
    """
    pts = np.asarray(points_m, dtype=np.float64).reshape(-1, 2)
    if corner_side == CornerSide.RIGHT:
        return mirror_y(pts)
    return pts.copy()


def side_from_metric_corner(corner_y_m: float) -> CornerSide:
    """Infer corner side from the corner's *metric* y (robust, calibration-based).

    Once calibrated, the taker's corner sits near y = 0 or y = 68; this decides
    the side without relying on the broadcast-frame heuristic.
    """
    return CornerSide.RIGHT if corner_y_m > PITCH_WIDTH_M / 2.0 else CornerSide.LEFT


# --- impure edge: corner-arc detection (cv2) --------------------------------
def detect_corner_side(
    frame: np.ndarray,
    white_thresh: int = 200,
    min_radius: int = 6,
    max_radius: int = 80,
    edge_band: float = 0.30,
) -> CornerSideResult:
    """Best-effort corner-side detection from the corner arc near a frame edge.

    Builds a white-line mask, then runs a Hough circle search in the left and
    right edge bands; the band with the stronger circular (arc) response wins.
    Confidence is the normalized dominance of the winning side -- treat a low
    value as "confirm/override manually" (a first-class catalog action, FR-007).

    Note: broadcast left/right is mapped to CornerSide.LEFT/RIGHT by convention;
    per-camera setups may need the manual override or ``side_from_metric_corner``.
    """
    import cv2  # local import keeps the pure API importable without a display build

    if frame is None or frame.ndim != 3:
        raise ValueError("expected a BGR image frame")
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    white = cv2.threshold(gray, white_thresh, 255, cv2.THRESH_BINARY)[1]

    band = int(w * edge_band)

    def _arc_score(mask_region: np.ndarray) -> float:
        circles = cv2.HoughCircles(
            mask_region, cv2.HOUGH_GRADIENT, dp=1.5, minDist=max(1, min_radius),
            param1=100, param2=18, minRadius=min_radius, maxRadius=max_radius,
        )
        if circles is None:
            return 0.0
        # weight by inverse radius: the corner arc is small and near the edge
        return float(sum(1.0 / (1.0 + r) for _, _, r in circles[0]))

    left_score = _arc_score(white[:, :band])
    right_score = _arc_score(white[:, w - band:])

    total = left_score + right_score
    if total <= 0.0:
        # nothing detected -> default LEFT, zero confidence -> manual pass
        return CornerSideResult(CornerSide.LEFT, Source.AUTO, 0.0)

    if left_score >= right_score:
        side = CornerSide.LEFT
        confidence = (left_score - right_score) / total
    else:
        side = CornerSide.RIGHT
        confidence = (right_score - left_score) / total
    return CornerSideResult(side, Source.AUTO, round(confidence, 3))


def manual_corner_side(side: CornerSide) -> CornerSideResult:
    """Catalog override: assert the corner side by hand (confidence 1.0)."""
    return CornerSideResult(side, Source.MANUAL, 1.0)
