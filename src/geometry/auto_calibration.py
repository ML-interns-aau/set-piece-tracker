"""Automatic pitch-marking detection for penalty-area calibration (FR-008).

Pipeline (all classical CV -- no learned model, no heavy deps beyond cv2):

    1. Isolate the playing field with a green HSV mask (largest component) so
       ad boards, stands and crowd stop polluting the edge map.
    2. Canny edges *inside the field* -> Hough segments -> merge collinear
       fragments into consolidated infinite lines (players and the net split a
       real pitch line into pieces; merging puts it back together).
    3. Split lines into two orientation families: longitudinal (goal line, the
       5.5 m goal-area line, the 16.5 m penalty-area line -- roughly parallel)
       and transverse (the penalty/goal-area *side* lines, which recede toward a
       vanishing point and are therefore skewed, not axis-aligned).
    4. Do NOT assume a rectangle. Corner correspondences are the **intersections
       of the actual detected segments** (longitudinal x transverse), which are
       perspective-correct by construction. Which line is which is resolved by
       trying the few consistent label assignments and keeping the one under
       which every detected line best lands on the known metric pitch grid.
    5. That grid-consistency score doubles as the validation gate: a frame with
       no coherent penalty area (e.g. a pre-kick close-up) scores badly and
       returns ``None`` so the clip routes to the manual fallback.

This is the impure edge (cv2); the pure homography solver stays in
``calibration.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.domain.models import Calibration, Source
from src.domain.pitch import (
    GOAL_AREA_DEPTH_M,
    GOAL_AREA_WIDTH_M,
    PENALTY_AREA_DEPTH_M,
    PENALTY_AREA_WIDTH_M,
    _CENTRE_Y,
)
from src.geometry.calibration import (
    reprojection_error,
    solve_homography,
    solve_homography_hybrid,
)

# A pitch line is stored as (x1, y1, x2, y2, angle_deg, support_length) -- the
# same tuple shape the debug drawing and _line_intersection expect. (x1,y1)-(x2,y2)
# are the extreme endpoints of the (possibly merged) line; support_length is the
# total length of the raw segments that back it.
Line = tuple

# Known metric line positions the detected lines must fall onto (FIFA standard).
LONGITUDINAL_X_M = (0.0, GOAL_AREA_DEPTH_M, PENALTY_AREA_DEPTH_M)          # 0, 5.5, 16.5
_PA_HALF = PENALTY_AREA_WIDTH_M / 2.0                                       # 20.16
_GA_HALF = GOAL_AREA_WIDTH_M / 2.0                                         # 9.16
TRANSVERSE_Y_M = (
    _CENTRE_Y - _PA_HALF, _CENTRE_Y - _GA_HALF,
    _CENTRE_Y + _GA_HALF, _CENTRE_Y + _PA_HALF,
)                                                                          # 13.84, 24.84, 43.16, 54.16
PA_SIDE_Y_M = (_CENTRE_Y - _PA_HALF, _CENTRE_Y + _PA_HALF)                 # 13.84, 54.16

# Tuning knobs (pilot-phase defaults; not final).
CONSISTENCY_MAX_M = 1.5        # reject a calibration whose lines miss the grid by > this
REPROJ_MAX_FIT_M = 2.5         # reject a label hypothesis whose own fit residual exceeds this
MIN_CORRESPONDENCES = 4        # the DLT solver needs at least 4 points
TOPHAT_KERNEL_PX = 17          # white top-hat kernel; wider than a pitch line, narrower than a torso
TOPHAT_THRESH = 25             # min top-hat response to count as a line pixel


# ---------------------------------------------------------------------------
# 1. Field isolation
# ---------------------------------------------------------------------------

def _field_mask(
    frame: np.ndarray,
    hsv_low: tuple[int, int, int] = (30, 20, 20),
    hsv_high: tuple[int, int, int] = (95, 255, 255),
) -> np.ndarray:
    """Binary mask of the grass field: green HSV threshold, largest component."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, np.array(hsv_low), np.array(hsv_high))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(green, 8)
    if n > 1:
        biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        field = (labels == biggest).astype(np.uint8) * 255
    else:
        field = green

    # Close holes (players/ball on the grass) then dilate so a white line sitting
    # just inside the field boundary is not clipped away.
    field = cv2.morphologyEx(field, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25)))
    field = cv2.dilate(field, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))
    return field


def _line_mask(frame: np.ndarray, field: np.ndarray | None = None) -> np.ndarray:
    """Binary mask of white pitch lines inside the field.

    A morphological white top-hat keeps thin bright structures (the painted
    lines) while suppressing broad bright regions (player kits, the net) and
    low-contrast texture (mown-grass stripes) that a raw Canny pass picks up.
    """
    if field is None:
        field = _field_mask(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kern = cv2.getStructuringElement(cv2.MORPH_RECT, (TOPHAT_KERNEL_PX, TOPHAT_KERNEL_PX))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kern)
    mask = (tophat > TOPHAT_THRESH).astype(np.uint8) * 255
    mask = cv2.bitwise_and(mask, field)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    return mask


# ---------------------------------------------------------------------------
# 2. Line detection: Hough segments -> merged collinear lines
# ---------------------------------------------------------------------------

def _angle_of(x1: float, y1: float, x2: float, y2: float) -> float:
    return float(np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180.0)


def _angle_diff(a: float, b: float) -> float:
    """Smallest difference between two orientations in [0, 180)."""
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def _line_midpoint(line: Line) -> tuple[float, float]:
    return ((line[0] + line[2]) / 2.0, (line[1] + line[3]) / 2.0)


def _line_direction(line: Line) -> np.ndarray:
    """Unit direction vector of a line."""
    dx, dy = line[2] - line[0], line[3] - line[1]
    length = np.hypot(dx, dy)
    if length < 1e-9:
        return np.array([1.0, 0.0])
    return np.array([dx, dy]) / length


def _line_intersection(l1: Line, l2: Line) -> tuple[float, float] | None:
    """Intersection of two segments extended to infinite lines (or None if parallel)."""
    x1, y1, x2, y2 = l1[:4]
    x3, y3, x4, y4 = l2[:4]
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-9:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))


def _detect_segments(mask: np.ndarray, min_length_frac: float = 0.05) -> np.ndarray:
    """Raw Hough line segments as an (N, 4) array of (x1, y1, x2, y2)."""
    h, w = mask.shape[:2]
    min_len = int(min_length_frac * max(h, w))
    segs = cv2.HoughLinesP(mask, 1, np.pi / 180, threshold=40,
                           minLineLength=min_len, maxLineGap=30)
    if segs is None:
        return np.empty((0, 4), dtype=np.float64)
    return segs.reshape(-1, 4).astype(np.float64)


def _merge_collinear(
    segments: np.ndarray,
    angle_tol_deg: float = 4.0,
    rho_tol_px: float = 12.0,
) -> list[Line]:
    """Merge collinear Hough fragments into consolidated lines.

    Segments are grouped in Hough normal form ``(theta, rho)`` with *global*
    coordinates -- so a fragment joins a cluster only if its own orientation and
    perpendicular offset match, never by chaining off a drifting centroid. This
    reconstructs a pitch line broken up by players/net without fusing distinct
    parallel lines together. ``rho`` is the signed distance from the image origin
    to the line; support length is the summed length of the member fragments.
    """
    clusters: list[dict] = []
    for x1, y1, x2, y2 in segments:
        a = _angle_of(x1, y1, x2, y2)
        length = float(np.hypot(x2 - x1, y2 - y1))
        mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        rad = np.radians(a)
        rho = -np.sin(rad) * mx + np.cos(rad) * my  # normal-form offset

        match = None
        for cl in clusters:
            if _angle_diff(a, cl["a"]) <= angle_tol_deg and abs(rho - cl["rho"]) <= rho_tol_px:
                match = cl
                break

        if match is None:
            clusters.append({
                "a": a, "rho": rho, "w": length,
                "s2": np.sin(2 * rad) * length, "c2": np.cos(2 * rad) * length,
                "rw": rho * length, "support": length,
                "pts": [[x1, y1], [x2, y2]],
            })
        else:
            match["w"] += length
            match["s2"] += np.sin(2 * rad) * length
            match["c2"] += np.cos(2 * rad) * length
            match["rw"] += rho * length
            match["support"] += length
            match["pts"] += [[x1, y1], [x2, y2]]
            match["a"] = (np.degrees(np.arctan2(match["s2"], match["c2"])) / 2.0) % 180.0
            match["rho"] = match["rw"] / match["w"]

    lines: list[Line] = []
    for cl in clusters:
        pts = np.array(cl["pts"], dtype=np.float64)
        rad = np.radians(cl["a"])
        d = np.array([np.cos(rad), np.sin(rad)])
        centroid = pts.mean(axis=0)
        t = (pts - centroid) @ d
        p_lo, p_hi = centroid + t.min() * d, centroid + t.max() * d
        lines.append((float(p_lo[0]), float(p_lo[1]), float(p_hi[0]), float(p_hi[1]),
                      cl["a"], float(cl["support"])))

    lines.sort(key=lambda ln: ln[5], reverse=True)
    return lines


# ---------------------------------------------------------------------------
# 3. Orientation families
# ---------------------------------------------------------------------------

def _orientation_families(
    lines: list[Line],
    family_tol_deg: float = 18.0,
) -> tuple[list[Line], list[Line]]:
    """Split lines into (longitudinal, transverse).

    Longitudinal = the largest set of mutually near-parallel lines, weighted by
    support length (the goal line and the 5.5/16.5 m lines are long and roughly
    parallel). Transverse = everything else (the skewed side lines, plus arc and
    residual noise, which are filtered later by length).
    """
    if not lines:
        return [], []

    best_anchor, best_score = 0, -1.0
    for i, anchor in enumerate(lines):
        score = sum(ln[5] for ln in lines if _angle_diff(ln[4], anchor[4]) <= family_tol_deg)
        if score > best_score:
            best_score, best_anchor = score, i

    anchor_angle = lines[best_anchor][4]
    longitudinal, transverse = [], []
    for ln in lines:
        (longitudinal if _angle_diff(ln[4], anchor_angle) <= family_tol_deg
         else transverse).append(ln)
    return longitudinal, transverse


# ---------------------------------------------------------------------------
# 4. Label assignment by homography consistency
# ---------------------------------------------------------------------------

@dataclass
class _Hypothesis:
    """One candidate calibration: 3 longitudinal lines + the penalty arc."""
    H: np.ndarray
    point_corr: list[tuple[tuple[float, float], tuple[float, float]]]  # arc pts (pixel, metric)
    line_segs: list[Line]      # longitudinal lines used to solve H
    line_x: list[float]        # metric x assigned to each of those lines
    arc_ellipse: tuple | None  # cv2 ellipse ((cx,cy),(MA,ma),angle)
    consistency_m: float
    reprojection_m: float


def _grid_consistency(
    H: np.ndarray,
    longitudinal: list[Line],
    transverse: list[Line],
    cap_m: float = 3.0,
) -> float:
    """Support-weighted distance (m) from *all* detected lines to the metric grid.

    A correct homography maps longitudinal lines onto x in {0, 5.5, 16.5} and
    transverse lines onto y in {13.84, 24.84, 43.16, 54.16}. Scoring over every
    detected line -- not just the four used to solve H -- is what makes this a
    real (non-circular) quality signal: a homography fit to a player edge plus
    one real line places all the *other* real lines off-grid and scores badly.
    Each line's error is capped (so one noise line can't dominate) and weighted
    by its support length (so the strong goal/16.5 m lines carry the score).
    """
    num = den = 0.0
    for ln in longitudinal:
        pts = np.array([[ln[0], ln[1]], [ln[2], ln[3]], _line_midpoint(ln)])
        xs = _apply(H, pts)[:, 0]
        err = float(np.mean([min(abs(x - gx) for gx in LONGITUDINAL_X_M) for x in xs]))
        num += ln[5] * min(err, cap_m)
        den += ln[5]
    for ln in transverse:
        pts = np.array([[ln[0], ln[1]], [ln[2], ln[3]], _line_midpoint(ln)])
        ys = _apply(H, pts)[:, 1]
        err = float(np.mean([min(abs(y - gy) for gy in TRANSVERSE_Y_M) for y in ys]))
        num += ln[5] * min(err, cap_m)
        den += ln[5]
    return num / den if den else float("inf")


def _apply(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Map pixel points (N,2) through H to metric (N,2)."""
    homog = np.c_[pts, np.ones(len(pts))]
    mapped = (H @ homog.T).T
    w = mapped[:, 2:3]
    w[np.abs(w) < 1e-12] = 1e-12
    return mapped[:, :2] / w


def _plausible(H: np.ndarray, frame_shape: tuple[int, int]) -> bool:
    """Reject degenerate homographies that could still fool the grid score.

    A collapsing (near-singular) H maps the frame to a sliver near one grid line,
    scoring a deceptively low consistency. Guard by mapping the frame corners to
    metres and requiring a finite, sensibly-bounded, non-collapsed pitch region.
    """
    h, w = frame_shape
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64)
    m = _apply(H, corners)
    if not np.all(np.isfinite(m)):
        return False
    if m[:, 0].min() < -40 or m[:, 0].max() > 130:   # metric x well outside a half-pitch
        return False
    if m[:, 1].min() < -40 or m[:, 1].max() > 110:   # metric y well outside the pitch
        return False
    area = 0.5 * abs(
        sum(m[i, 0] * m[(i + 1) % 4, 1] - m[(i + 1) % 4, 0] * m[i, 1] for i in range(4))
    )
    return area >= 200.0   # the visible penalty area alone is ~665 m^2


def _sample_ellipse(ellipse: tuple, n: int = 720) -> np.ndarray:
    """Sample ``n`` points (n, 2) around a cv2 ellipse ((cx,cy),(MA,ma),angle_deg)."""
    (cx, cy), (major, minor), ang = ellipse
    a, b = major / 2.0, minor / 2.0
    th = np.radians(ang)
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x, y = a * np.cos(t), b * np.sin(t)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    return (R @ np.vstack([x, y])).T + np.array([cx, cy])


def _extract_arc(
    line_mask: np.ndarray,
    front_line: Line,
    inward_pt: tuple[float, float],
    band_px: float = 200.0,
    min_pts: int = 60,
) -> tuple | None:
    """Fit an ellipse to the penalty arc: white pixels just beyond the 16.5 m line.

    The arc is the one distinctive marking that clears the corner-kick player wall
    (defenders pack the box *inside* the front line; the arc bows out past it). We
    keep white pixels on the midfield side of ``front_line`` within a band, take
    the largest connected blob, and fit an ellipse (the perspective image of the
    circular arc). Returns the cv2 ellipse, or ``None`` if too little curve is found.
    """
    d = _line_direction(front_line)
    n = np.array([-d[1], d[0]])
    p0 = np.array([front_line[0], front_line[1]])
    sign = -np.sign(np.dot(np.array(inward_pt) - p0, n)) or 1.0  # +side = away from goal

    ys, xs = np.where(line_mask > 0)
    if len(xs) == 0:
        return None
    pts = np.column_stack([xs, ys]).astype(np.float64)
    sd = ((pts - p0) @ n) * sign
    keep = (sd > 12.0) & (sd < band_px)
    if keep.sum() < min_pts:
        return None

    blob = np.zeros_like(line_mask)
    kp = pts[keep].astype(np.int32)
    blob[kp[:, 1], kp[:, 0]] = 255
    blob = cv2.dilate(blob, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    ncc, lab, stats, _ = cv2.connectedComponentsWithStats(blob, 8)
    if ncc <= 1:
        return None
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    cc = np.column_stack(np.where(lab == biggest))[:, ::-1]  # (x, y)
    if len(cc) < min_pts:
        return None
    try:
        return cv2.fitEllipse(cc.astype(np.int32))
    except cv2.error:
        return None


ARC_APEX_M = (PENALTY_AREA_DEPTH_M + 3.65, _CENTRE_Y)          # (~20.15, 34) arc apex
_ARC_HALF = float(np.sqrt(9.15 ** 2 - (PENALTY_AREA_DEPTH_M - 11.0) ** 2))   # 7.31 m
ARC_END_Y_M = (_CENTRE_Y - _ARC_HALF, _CENTRE_Y + _ARC_HALF)   # arc meets 16.5 m line


def _search_labelling(
    longitudinal: list[Line],
    line_mask: np.ndarray,
    frame_shape: tuple[int, int],
) -> _Hypothesis | None:
    """Calibrate from the 3 longitudinal lines + the penalty arc.

    Side lines are occluded in corner footage, so instead of box corners we anchor
    on the arc: its apex (~20.15 m, 34) is off the goal line and its two crossings
    of the 16.5 m line pin the cross-direction. For each x-ordering of the ordered
    longitudinal lines we take the line labelled 16.5 m as the front, fit the arc
    beyond it, derive the three arc points, solve the hybrid line+point homography,
    and keep the assignment that best fits every detected line onto the pitch grid.
    """
    longs = sorted(longitudinal, key=lambda ln: ln[5], reverse=True)[:3]
    if len(longs) < 2:
        return None
    score_long = sorted(longitudinal, key=lambda ln: ln[5], reverse=True)[:8]

    anchor_dir = _line_direction(longs[0])
    across = np.array([-anchor_dir[1], anchor_dir[0]])
    longs.sort(key=lambda ln: np.dot(_line_midpoint(ln), across))

    if len(longs) == 3:
        x_options = [(0.0, GOAL_AREA_DEPTH_M, PENALTY_AREA_DEPTH_M),
                     (PENALTY_AREA_DEPTH_M, GOAL_AREA_DEPTH_M, 0.0)]
    else:
        x_options = [(0.0, PENALTY_AREA_DEPTH_M), (PENALTY_AREA_DEPTH_M, 0.0)]

    best: _Hypothesis | None = None
    for xs in x_options:
        front = longs[xs.index(PENALTY_AREA_DEPTH_M)]
        goal = longs[xs.index(0.0)]
        ellipse = _extract_arc(line_mask, front, _line_midpoint(goal))
        if ellipse is None:
            continue

        samp = _sample_ellipse(ellipse)
        d = _line_direction(front)
        n = np.array([-d[1], d[0]])
        p0 = np.array([front[0], front[1]])
        sign = -np.sign(np.dot(np.array(_line_midpoint(goal)) - p0, n)) or 1.0
        sd = ((samp - p0) @ n) * sign
        along = (samp - p0) @ d

        apex = samp[int(np.argmax(sd))]
        near_front = samp[np.abs(sd) < 4.0]
        near_along = along[np.abs(sd) < 4.0]
        if len(near_front) < 2:
            continue
        end_a = near_front[int(np.argmin(near_along))]
        end_b = near_front[int(np.argmax(near_along))]

        for y_lo, y_hi in (ARC_END_Y_M, ARC_END_Y_M[::-1]):
            point_corr = [
                (tuple(apex), ARC_APEX_M),
                (tuple(end_a), (PENALTY_AREA_DEPTH_M, y_lo)),
                (tuple(end_b), (PENALTY_AREA_DEPTH_M, y_hi)),
            ]
            seg_arr = np.array([[[ln[0], ln[1]], [ln[2], ln[3]]] for ln in longs])
            pts_px = np.array([c[0] for c in point_corr])
            pts_m = np.array([c[1] for c in point_corr])
            try:
                H = solve_homography_hybrid(pts_px, pts_m, seg_arr, np.array(xs))
            except (ValueError, np.linalg.LinAlgError):
                continue
            reproj = float(np.sqrt(((_apply(H, pts_px) - pts_m) ** 2).sum(axis=1)).mean())
            if reproj > REPROJ_MAX_FIT_M or not _plausible(H, frame_shape):
                continue
            cons = _grid_consistency(H, score_long, [])
            cand = _Hypothesis(H, point_corr, longs, list(xs), ellipse, cons, reproj)
            if best is None or cand.consistency_m < best.consistency_m:
                best = cand
    return best


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def auto_calibrate(frame: np.ndarray, source: Source = Source.AUTO) -> Calibration | None:
    """Attempt automatic calibration from a single frame (None if it can't be trusted)."""
    calib, _ = auto_calibrate_detailed(frame, source)
    return calib


def auto_calibrate_detailed(
    frame: np.ndarray,
    source: Source = Source.AUTO,
) -> tuple[Calibration | None, dict]:
    """Like :func:`auto_calibrate` but also returns detection metadata for debugging.

    The ``info`` dict carries ``field_mask``, ``edge_mask``, the merged
    ``longitudinal`` / ``transverse`` lines, the arc ``point_corr`` (pixel ->
    metric) and fitted ``arc_ellipse``, the longitudinal lines used, and the
    ``consistency_m`` score.
    """
    field = _field_mask(frame)
    edges = _line_mask(frame, field)

    segments = _detect_segments(edges)
    lines = _merge_collinear(segments)
    longitudinal, transverse = _orientation_families(lines)

    info: dict = {
        "field_mask": field,
        "edge_mask": edges,
        "n_segments": len(segments),
        "n_lines": len(lines),
        "longitudinal": longitudinal,
        "transverse": transverse,
        "point_corr": None,
        "line_segs": None,
        "arc_ellipse": None,
        "consistency_m": None,
        "reprojection_error_m": None,
    }

    hyp = _search_labelling(longitudinal, edges, frame.shape[:2])
    if hyp is None:
        return None, info

    info["point_corr"] = hyp.point_corr
    info["line_segs"] = hyp.line_segs
    info["arc_ellipse"] = hyp.arc_ellipse
    info["consistency_m"] = hyp.consistency_m
    info["reprojection_error_m"] = hyp.reprojection_m

    if hyp.consistency_m > CONSISTENCY_MAX_M:
        return None, info

    calib = Calibration(
        H=hyp.H,
        reprojection_error_m=hyp.reprojection_m,
        points_used=len(hyp.point_corr) + len(hyp.line_segs),
        source=source,
    )
    return calib, info
