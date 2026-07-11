"""FIFA-standard pitch model and named reference points (pure, no cv2).

Coordinate convention (the *reference frame* the homography maps into):

    - Origin at a pitch corner. Units: metres.
    - x runs from the analysed goal line (x = 0) toward the halfway line (x -> 105).
    - y runs across the pitch width, 0 .. 68.
    - The analysed goal is centred at y = 34; posts at y = 30.34 and y = 37.66.

Corner footage only ever shows one penalty area, so we always calibrate that
visible area to the x = 0 end. Which *side* the corner is taken from (near the
(0, 0) corner vs. the (0, 68) corner) is handled separately by orientation
normalisation (see ``geometry.orientation``) so that "near post" / "far post"
mean the same thing across every clip.

Canonical orientation: the corner is taken from the y = 0 corner, so the
**near post** is (0, 30.34) and the **far post** is (0, 37.66). A clip whose
corner is on the y = 68 side is mirrored (y -> 68 - y) into this frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# --- FIFA standard dimensions (metres) --------------------------------------
PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0
PENALTY_AREA_DEPTH_M = 16.5
PENALTY_AREA_WIDTH_M = 40.32
GOAL_AREA_DEPTH_M = 5.5
GOAL_AREA_WIDTH_M = 18.32
GOAL_WIDTH_M = 7.32
PENALTY_SPOT_DIST_M = 11.0
CORNER_ARC_RADIUS_M = 0.9144  # 1 yard

_CENTRE_Y = PITCH_WIDTH_M / 2.0  # 34.0


@dataclass(frozen=True)
class PitchModel:
    """Immutable bundle of the dimensions above, for injection/overrides."""

    length_m: float = PITCH_LENGTH_M
    width_m: float = PITCH_WIDTH_M
    penalty_area_depth_m: float = PENALTY_AREA_DEPTH_M
    penalty_area_width_m: float = PENALTY_AREA_WIDTH_M
    goal_area_depth_m: float = GOAL_AREA_DEPTH_M
    goal_area_width_m: float = GOAL_AREA_WIDTH_M
    goal_width_m: float = GOAL_WIDTH_M
    penalty_spot_dist_m: float = PENALTY_SPOT_DIST_M


PITCH = PitchModel()

# --- Named reference points in the reference frame (metres) ------------------
# These are the standard markings a corner broadcast view can plausibly show.
# Any >= 4 of them, matched to their pixel locations, calibrate a clip (FR-008).
REFERENCE_POINTS: dict[str, tuple[float, float]] = {
    # goal posts (on the goal line, x = 0)
    "near_post": (0.0, _CENTRE_Y - GOAL_WIDTH_M / 2.0),          # (0, 30.34)
    "far_post": (0.0, _CENTRE_Y + GOAL_WIDTH_M / 2.0),           # (0, 37.66)
    # penalty-area corners: two on the goal line, two on the 16.5 m line
    "pen_area_gl_left": (0.0, _CENTRE_Y - PENALTY_AREA_WIDTH_M / 2.0),   # (0, 13.84)
    "pen_area_gl_right": (0.0, _CENTRE_Y + PENALTY_AREA_WIDTH_M / 2.0),  # (0, 54.16)
    "pen_area_front_left": (PENALTY_AREA_DEPTH_M, _CENTRE_Y - PENALTY_AREA_WIDTH_M / 2.0),   # (16.5, 13.84)
    "pen_area_front_right": (PENALTY_AREA_DEPTH_M, _CENTRE_Y + PENALTY_AREA_WIDTH_M / 2.0),  # (16.5, 54.16)
    # goal-area (6-yard box) corners
    "goal_area_gl_left": (0.0, _CENTRE_Y - GOAL_AREA_WIDTH_M / 2.0),     # (0, 24.84)
    "goal_area_gl_right": (0.0, _CENTRE_Y + GOAL_AREA_WIDTH_M / 2.0),    # (0, 43.16)
    "goal_area_front_left": (GOAL_AREA_DEPTH_M, _CENTRE_Y - GOAL_AREA_WIDTH_M / 2.0),   # (5.5, 24.84)
    "goal_area_front_right": (GOAL_AREA_DEPTH_M, _CENTRE_Y + GOAL_AREA_WIDTH_M / 2.0),  # (5.5, 43.16)
    # penalty spot
    "penalty_spot": (PENALTY_SPOT_DIST_M, _CENTRE_Y),           # (11, 34)
    # corner arcs (also used by orientation detection)
    "corner_near": (0.0, 0.0),
    "corner_far": (0.0, PITCH_WIDTH_M),
}


def reference_points_metric(names: list[str]) -> np.ndarray:
    """Metric coords (N, 2) for the given reference-point names, in order.

    Raises KeyError if a name is unknown.
    """
    return np.array([REFERENCE_POINTS[n] for n in names], dtype=np.float64)


def mirror_y(points_m: np.ndarray) -> np.ndarray:
    """Mirror metric points across the pitch mid-line (y -> width - y).

    Used to bring a y = 68-side corner into the canonical y = 0-side frame.
    """
    pts = np.asarray(points_m, dtype=np.float64).reshape(-1, 2).copy()
    pts[:, 1] = PITCH_WIDTH_M - pts[:, 1]
    return pts
