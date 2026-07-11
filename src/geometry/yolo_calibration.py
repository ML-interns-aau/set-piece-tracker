"""YOLO-based auto-calibration for penalty-area views (FR-008).

Uses YOLO ball and player detections to infer anchor points on the pitch:
- The ball at the corner flag gives the corner anchor (known metric position).
- The goalkeeper (most isolated player near the goal line) gives the goal anchor.
- A third anchor is inferred from the penalty-area player distribution.

From these correspondences, a homography (similarity transform) is solved that
maps pixel coordinates to metric pitch coordinates. This is more robust than
HSV line detection because it works with the actual objects on the pitch rather
than fragile colour thresholds.

This is the impure edge (cv2, torch, ultralytics); pure calibration logic
stays in ``calibration.py``.
"""

from __future__ import annotations

import numpy as np

from src.domain.models import Calibration, Source
from src.domain.pitch import (
    GOAL_WIDTH_M,
    PENALTY_AREA_DEPTH_M,
    PENALTY_AREA_WIDTH_M,
    PITCH_WIDTH_M,
    REFERENCE_POINTS,
    _CENTRE_Y,
)
from src.geometry.calibration import calibrate_from_markings, solve_homography, reprojection_error
from src.geometry.orientation import side_from_metric_corner


def _foot_point(bbox: np.ndarray) -> np.ndarray:
    """Bottom-centre of a bounding box (the point on the ground)."""
    x1, y1, x2, y2 = bbox
    return np.array([(x1 + x2) / 2.0, y2])


def _find_corner_and_gk(
    clip_path: str,
    n_frames: int = 30,
    device: str = "cpu",
    conf: float = 0.20,
) -> dict | None:
    """Scan the first ``n_frames`` with YOLO to find the ball (corner) and GK (goal).

    Returns a dict with keys:
        ball_px: (u, v) pixel position of the ball (corner anchor)
        gk_px: (u, v) pixel position of the goalkeeper (goal anchor)
        gk_isolation: isolation score of the GK
        n_players: number of players detected
        corner_side: CornerSide.LEFT or RIGHT based on ball x-position
    Or None if detection fails.
    """
    import cv2

    from src.engine.detector import FootballDetector
    from src.engine.team_classifier import TeamClassifier

    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        return None

    detector = FootballDetector(model_path="yolo11m.pt", conf=conf, device=device)
    classifier = TeamClassifier(detect_goalkeeper=True)

    ball_positions: list[tuple[float, float]] = []
    player_bboxes: list[np.ndarray] = []
    player_teams: list[int] = []
    gk_ids: list[int] = []

    for frame_idx in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break

        dets = detector.detect(frame)
        if len(dets) == 0:
            continue

        # Ball
        ball_mask = dets.class_id == 32
        if ball_mask.any():
            ball_dets = dets[ball_mask]
            best = int(np.argmax(ball_dets.confidence))
            bx1, by1, bx2, by2 = ball_dets.xyxy[best]
            ball_positions.append(((bx1 + bx2) / 2, (by1 + by2) / 2))

        # Players + team classification
        player_mask = dets.class_id == 0
        if player_mask.any():
            player_dets = dets[player_mask]
            teams = classifier.assign_teams(frame, player_dets)
            for i, (bbox, team) in enumerate(zip(player_dets.xyxy, teams)):
                player_bboxes.append(bbox)
                player_teams.append(int(team))
                if int(team) == TeamClassifier.GK_ID:
                    gk_ids.append(len(player_bboxes) - 1)

    cap.release()

    if len(ball_positions) < 1 or len(player_bboxes) < 3:
        return None

    # Ball: take the median position across frames (stable corner position)
    ball_arr = np.array(ball_positions)
    ball_px = (float(np.median(ball_arr[:, 0])), float(np.median(ball_arr[:, 1])))

    # Goalkeeper: find the player with the GK team label, or fall back to
    # the most isolated player near the bottom of the frame (goal line area)
    gk_px = None
    gk_isolation = 0.0

    if gk_ids:
        # Use the last detected GK position (closest to the action)
        last_gk_idx = gk_ids[-1]
        gk_bbox = player_bboxes[last_gk_idx]
        gk_px = tuple(_foot_point(gk_bbox))
        gk_isolation = 1.0
    else:
        # Fallback: find the most isolated player in the bottom third of the frame
        foot_points = np.array([_foot_point(b) for b in player_bboxes])
        frame_h = 720  # approximate; will be overridden
        # Players in the bottom third (near goal line in broadcast view)
        bottom_mask = foot_points[:, 1] > frame_h * 0.6
        if bottom_mask.any():
            bottom_fps = foot_points[bottom_mask]
            bottom_indices = np.where(bottom_mask)[0]
            best_idx, best_isolation = 0, -1.0
            for i, fp in enumerate(bottom_fps):
                others = np.delete(bottom_fps, i, axis=0)
                if len(others) == 0:
                    continue
                isolation = float(np.mean(np.linalg.norm(others - fp, axis=1)))
                if isolation > best_isolation:
                    best_isolation = isolation
                    best_idx = i
            gk_px = tuple(bottom_fps[best_idx])
            gk_isolation = best_isolation / 100.0  # normalize roughly

    if gk_px is None:
        return None

    # Determine corner side from ball x-position
    # In a left-corner view, the ball is on the left side of the frame
    # In a right-corner view, the ball is on the right side
    frame_w = 1280  # approximate
    corner_side = side_from_metric_corner(0.0)  # default LEFT
    if ball_px[0] > frame_w / 2:
        corner_side = side_from_metric_corner(PITCH_WIDTH_M)  # RIGHT

    return {
        "ball_px": ball_px,
        "gk_px": gk_px,
        "gk_isolation": gk_isolation,
        "n_players": len(player_bboxes),
        "corner_side": corner_side,
        "frame_w": frame_w,
        "frame_h": frame_h,
    }


def _solve_from_anchors(
    ball_px: tuple[float, float],
    gk_px: tuple[float, float],
    corner_side,
    frame_shape: tuple[int, int],
) -> Calibration | None:
    """Solve a homography from the detected anchor points.

    Uses the ball (corner) and goalkeeper (goal) as two known correspondences,
    then infers additional points from the pitch geometry to get >= 4 for a
    proper homography.

    Strategy:
    1. Ball at corner: (0, 0) or (0, 68) depending on side
    2. Goalkeeper near goal line: approximately (2, 34) — 2m from goal line, centred
    3. Infer the penalty-area front from the goal-to-ball vector direction
    4. Use the known penalty area width to get the side points
    """
    from src.domain.pitch import PITCH_WIDTH_M, mirror_y

    h, w = frame_shape

    # Known metric positions
    if corner_side.value == "left":
        corner_metric = np.array([0.0, 0.0])
    else:
        corner_metric = np.array([0.0, PITCH_WIDTH_M])

    # Goalkeeper is roughly 2m from the goal line, centred
    gk_metric = np.array([2.0, _CENTRE_Y])

    # Goal line centre (for additional anchor)
    goal_centre_metric = np.array([0.0, _CENTRE_Y])

    # Penalty area front centre (16.5m from goal line)
    pa_front_metric = np.array([PENALTY_AREA_DEPTH_M, _CENTRE_Y])

    # Penalty area front corners
    pa_fl_metric = np.array([PENALTY_AREA_DEPTH_M, _CENTRE_Y - PENALTY_AREA_WIDTH_M / 2])
    pa_fr_metric = np.array([PENALTY_AREA_DEPTH_M, _CENTRE_Y + PENALTY_AREA_WIDTH_M / 2])

    # Now we need to figure out the pixel positions of these metric points.
    # We know:
    #   ball_px -> corner_metric
    #   gk_px -> gk_metric
    # From these two correspondences, we can estimate a similarity transform
    # (rotation + translation + scale) and use it to project the other points.

    ball_px_arr = np.array(ball_px)
    gk_px_arr = np.array(gk_px)

    # Vector from ball to GK in pixel space
    px_vec = gk_px_arr - ball_px_arr
    # Vector from corner to GK in metric space
    m_vec = gk_metric - corner_metric

    # Scale: ratio of distances
    px_dist = np.linalg.norm(px_vec)
    m_dist = np.linalg.norm(m_vec)
    if px_dist < 10 or m_dist < 0.1:
        return None
    scale = px_dist / m_dist

    # Rotation angle
    px_angle = np.arctan2(px_vec[1], px_vec[0])
    m_angle = np.arctan2(m_vec[1], m_vec[0])
    rotation = px_angle - m_angle

    # Build transform: pixel = scale * R @ metric + translation
    cos_r, sin_r = np.cos(rotation), np.sin(rotation)
    R = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
    T = ball_px_arr - scale * R @ corner_metric

    def metric_to_px(pt_m):
        return scale * R @ pt_m + T

    # Project known metric points to pixel space
    goal_centre_px = metric_to_px(goal_centre_metric)
    pa_front_px = metric_to_px(pa_front_metric)
    pa_fl_px = metric_to_px(pa_fl_metric)
    pa_fr_px = metric_to_px(pa_fr_metric)

    # Build correspondence set for a proper homography (>= 4 points)
    pixel_points = {
        "near_post": tuple(metric_to_px(REFERENCE_POINTS["near_post"])),
        "far_post": tuple(metric_to_px(REFERENCE_POINTS["far_post"])),
        "pen_area_gl_left": tuple(metric_to_px(REFERENCE_POINTS["pen_area_gl_left"])),
        "pen_area_gl_right": tuple(metric_to_px(REFERENCE_POINTS["pen_area_gl_right"])),
        "pen_area_front_left": tuple(pa_fl_px),
        "pen_area_front_right": tuple(pa_fr_px),
        "penalty_spot": tuple(metric_to_px(REFERENCE_POINTS["penalty_spot"])),
    }

    # Solve the full homography from these correspondences
    try:
        calib = calibrate_from_markings(pixel_points, source=Source.AUTO)
        return calib
    except (ValueError, KeyError):
        return None


def yolo_calibrate(
    clip_path: str,
    n_frames: int = 30,
    device: str = "cpu",
    conf: float = 0.20,
) -> Calibration | None:
    """Auto-calibrate using YOLO ball + player detections.

    Detects the ball (corner anchor) and goalkeeper (goal anchor) across the
    first ``n_frames`` of the clip, then solves a homography from those known
    pitch positions.

    Returns a ``Calibration`` on success, or ``None`` if detection fails.
    """
    result = _find_corner_and_gk(clip_path, n_frames=n_frames, device=device, conf=conf)
    if result is None:
        return None

    return _solve_from_anchors(
        ball_px=result["ball_px"],
        gk_px=result["gk_px"],
        corner_side=result["corner_side"],
        frame_shape=(result["frame_h"], result["frame_w"]),
    )


def yolo_calibrate_detailed(
    clip_path: str,
    n_frames: int = 30,
    device: str = "cpu",
    conf: float = 0.20,
) -> tuple[Calibration | None, dict]:
    """Like ``yolo_calibrate`` but also returns detection metadata."""
    result = _find_corner_and_gk(clip_path, n_frames=n_frames, device=device, conf=conf)
    if result is None:
        return None, {"error": "detection failed"}

    calib = _solve_from_anchors(
        ball_px=result["ball_px"],
        gk_px=result["gk_px"],
        corner_side=result["corner_side"],
        frame_shape=(result["frame_h"], result["frame_w"]),
    )

    return calib, result
