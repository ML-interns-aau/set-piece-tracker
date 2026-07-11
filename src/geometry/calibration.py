"""Penalty-area calibration: pixel -> metric homography from named FIFA points.

Corner footage never shows all four pitch corners, so we solve the homography
from whatever standard markings are visible (penalty-area corners, goal-area
corners, goalposts, penalty spot, corner arcs) -- any >= 4 give a
well-conditioned correspondence to known metric coordinates (FR-008).

The solver is a normalized DLT implemented in pure numpy, so it unit-tests
without OpenCV. The interactive manual fallback lives in ``manual_calibration``
(it needs a cv2 window).
"""

from __future__ import annotations

import numpy as np

from src.domain.models import Calibration, Source
from src.domain.pitch import REFERENCE_POINTS, reference_points_metric

MIN_POINTS = 4


def _normalization_matrix(pts: np.ndarray) -> np.ndarray:
    """Hartley normalization: translate centroid to origin, scale mean dist to sqrt(2)."""
    centroid = pts.mean(axis=0)
    shifted = pts - centroid
    mean_dist = np.sqrt((shifted ** 2).sum(axis=1)).mean()
    if mean_dist < 1e-12:
        raise ValueError("degenerate point set (all points coincident)")
    scale = np.sqrt(2.0) / mean_dist
    return np.array(
        [[scale, 0.0, -scale * centroid[0]],
         [0.0, scale, -scale * centroid[1]],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def solve_homography(src_px: np.ndarray, dst_m: np.ndarray) -> np.ndarray:
    """Solve H mapping ``src_px`` (pixels) -> ``dst_m`` (metres) via normalized DLT.

    Requires >= 4 non-degenerate correspondences. Returns a 3x3 matrix with
    H[2, 2] normalized to 1.
    """
    src = np.asarray(src_px, dtype=np.float64).reshape(-1, 2)
    dst = np.asarray(dst_m, dtype=np.float64).reshape(-1, 2)
    if src.shape[0] != dst.shape[0]:
        raise ValueError("src and dst must have the same number of points")
    if src.shape[0] < MIN_POINTS:
        raise ValueError(f"need >= {MIN_POINTS} points, got {src.shape[0]}")

    t_src = _normalization_matrix(src)
    t_dst = _normalization_matrix(dst)
    src_n = (t_src @ np.c_[src, np.ones(len(src))].T).T[:, :2]
    dst_n = (t_dst @ np.c_[dst, np.ones(len(dst))].T).T[:, :2]

    rows = []
    for (u, v), (x, y) in zip(src_n, dst_n):
        rows.append([-u, -v, -1, 0, 0, 0, u * x, v * x, x])
        rows.append([0, 0, 0, -u, -v, -1, u * y, v * y, y])
    a = np.array(rows, dtype=np.float64)

    _, _, vh = np.linalg.svd(a)
    h_norm = vh[-1].reshape(3, 3)

    # denormalize: H = T_dst^-1 @ H_norm @ T_src
    h = np.linalg.inv(t_dst) @ h_norm @ t_src
    if abs(h[2, 2]) < 1e-12:
        raise ValueError("degenerate homography (H[2,2] ~ 0)")
    return h / h[2, 2]


def _line_through(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """Homogeneous line (a, b, c) through two homogeneous points (a u + b v + c = 0)."""
    return np.cross(p1, p2)


def solve_homography_hybrid(
    points_px: np.ndarray,
    points_m: np.ndarray,
    line_segs_px: np.ndarray,
    line_metric_x: np.ndarray,
) -> np.ndarray:
    """Solve H (pixel -> metre) from a mix of point and vertical-line correspondences.

    Corner footage occludes the penalty-area *side* lines (players stand on them),
    so we cannot get four box corners. We can, however, see the *longitudinal*
    lines -- the goal line and the 5.5 m / 16.5 m lines, each a metric line
    ``x = c`` -- plus a couple of clear points (goalpost bases). This solver takes
    both: point correspondences constrain H the usual way, and each pixel line is
    constrained to be the image of its metric line ``x = c`` via ``l ~ H^T m``.

    Args:
        points_px: (N, 2) pixel points.
        points_m:  (N, 2) their metre coordinates.
        line_segs_px: (M, 2, 2) two pixel endpoints per detected longitudinal line.
        line_metric_x: (M,) the metric x each of those lines sits at (0, 5.5, 16.5).

    Needs ``2*N + 2*M >= 8`` independent constraints. Returns a 3x3 H (H[2,2]=1).
    """
    points_px = np.asarray(points_px, dtype=np.float64).reshape(-1, 2)
    points_m = np.asarray(points_m, dtype=np.float64).reshape(-1, 2)
    line_segs_px = np.asarray(line_segs_px, dtype=np.float64).reshape(-1, 2, 2)
    line_metric_x = np.asarray(line_metric_x, dtype=np.float64).reshape(-1)

    n_constraints = 2 * len(points_px) + 2 * len(line_segs_px)
    if n_constraints < 8:
        raise ValueError(f"need >= 8 constraints, got {n_constraints}")

    # Two metre points on each longitudinal line x = c (arbitrary distinct y) so a
    # line can be normalised and built as the join of two points, like the others.
    line_m_pts = np.array([[(c, 0.0), (c, 68.0)] for c in line_metric_x], dtype=np.float64)

    # Hartley normalisation from all points involved in each space.
    px_all = np.vstack([points_px, line_segs_px.reshape(-1, 2)]) if len(line_segs_px) else points_px
    m_all = np.vstack([points_m, line_m_pts.reshape(-1, 2)]) if len(line_metric_x) else points_m
    t_px = _normalization_matrix(px_all)
    t_m = _normalization_matrix(m_all)

    def hp(pt: np.ndarray, T: np.ndarray) -> np.ndarray:
        return T @ np.array([pt[0], pt[1], 1.0])

    rows: list[list[float]] = []

    for (u, v), (x, y) in zip(points_px, points_m):
        p = hp((u, v), t_px)
        q = hp((x, y), t_m)
        p1, p2, p3 = p
        q1, q2, q3 = q
        rows.append([0, 0, 0, -q3 * p1, -q3 * p2, -q3 * p3, q2 * p1, q2 * p2, q2 * p3])
        rows.append([q3 * p1, q3 * p2, q3 * p3, 0, 0, 0, -q1 * p1, -q1 * p2, -q1 * p3])

    for seg, (mp0, mp1) in zip(line_segs_px, line_m_pts):
        lpx = _line_through(hp(seg[0], t_px), hp(seg[1], t_px))
        lm = _line_through(hp(mp0, t_m), hp(mp1, t_m))
        l1, l2, l3 = lpx
        m1, m2, m3 = lm
        # l x (H^T m) = 0 -> two independent rows, linear in vec(H).
        rows.append([0, -l3 * m1, l2 * m1, 0, -l3 * m2, l2 * m2, 0, -l3 * m3, l2 * m3])
        rows.append([l3 * m1, 0, -l1 * m1, l3 * m2, 0, -l1 * m2, l3 * m3, 0, -l1 * m3])

    a = np.array(rows, dtype=np.float64)
    _, _, vh = np.linalg.svd(a)
    h_norm = vh[-1].reshape(3, 3)

    h = np.linalg.inv(t_m) @ h_norm @ t_px
    if abs(h[2, 2]) < 1e-12:
        raise ValueError("degenerate homography (H[2,2] ~ 0)")
    return h / h[2, 2]


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


def reprojection_error(h: np.ndarray, src_px: np.ndarray, dst_m: np.ndarray) -> float:
    """Mean Euclidean re-projection error (metres) of ``src_px`` mapped vs ``dst_m``."""
    projected = apply_homography(h, src_px)
    dst = np.asarray(dst_m, dtype=np.float64).reshape(-1, 2)
    return float(np.sqrt(((projected - dst) ** 2).sum(axis=1)).mean())


def calibrate_from_markings(
    pixel_points: dict[str, tuple[float, float]],
    source: Source = Source.AUTO,
) -> Calibration:
    """Build a :class:`Calibration` from named marking pixel locations.

    ``pixel_points`` maps reference-point names (keys of
    :data:`domain.pitch.REFERENCE_POINTS`) to their (u, v) pixel coordinates in
    the clip. Unknown names raise KeyError; fewer than 4 known points raise
    ValueError.
    """
    names = [n for n in pixel_points if n in REFERENCE_POINTS]
    unknown = set(pixel_points) - set(REFERENCE_POINTS)
    if unknown:
        raise KeyError(f"unknown reference point(s): {sorted(unknown)}")
    if len(names) < MIN_POINTS:
        raise ValueError(f"need >= {MIN_POINTS} known markings, got {len(names)}")

    src = np.array([pixel_points[n] for n in names], dtype=np.float64)
    dst = reference_points_metric(names)
    h = solve_homography(src, dst)
    err = reprojection_error(h, src, dst)
    return Calibration(H=h, reprojection_error_m=err, points_used=len(names), source=source)
