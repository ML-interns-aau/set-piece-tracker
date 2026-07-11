"""Ball delivery trajectory: projectile-model fit (FR-009, features 11-13).

Physics model (constant horizontal velocity, gravity-only vertical):

    s(t) = s0 + vs * t                         # horizontal distance along delivery
    z(t) = z0 + vz * t - 0.5 * g * t**2        # height above the pitch plane

``fit_projectile`` is a pure least-squares fit (g fixed, so both components are
linear in their parameters) and is unit-tested against analytic synthetic
trajectories. ``simulate_projectile`` is the forward model used to generate
those tests and to document the equations.

Monocular height is uncertain (FR-009): per-frame heights must come from a
height estimator (camera-pose dependent, out of scope here). When no heights
are supplied the fit still returns the horizontal metric (max speed) and leaves
height fields ``None`` rather than fabricating them. The fit reports a
confidence interval on height, never just a point value.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

from src.domain.models import BallSample, ProjectileFit

GRAVITY = 9.81


def simulate_projectile(
    launch_speed_ms: float,
    launch_angle_deg: float,
    n: int = 25,
    duration_s: float | None = None,
    z0_m: float = 0.0,
    g: float = GRAVITY,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Forward projectile model -> (times, horizontal_m, height_m).

    If ``duration_s`` is None it runs until the ball returns to ``z0_m``.
    """
    angle = np.radians(launch_angle_deg)
    vs = launch_speed_ms * np.cos(angle)
    vz = launch_speed_ms * np.sin(angle)
    if duration_s is None:
        duration_s = (2.0 * vz / g) if vz > 0 else 1.0
    t = np.linspace(0.0, duration_s, n)
    horizontal = vs * t
    height = z0_m + vz * t - 0.5 * g * t ** 2
    return t, horizontal, height


def fit_projectile(
    times_s: Sequence[float],
    horizontal_m: Sequence[float],
    height_m: Sequence[float] | None = None,
    target_horizontal_m: float | None = None,
    g: float = GRAVITY,
) -> ProjectileFit:
    """Least-squares fit of the projectile model to sampled points.

    ``height_m`` may be ``None`` (monocular height unavailable) -- then only the
    horizontal metric is returned. Requires >= 2 samples (>= 3 for a meaningful
    height confidence interval).
    """
    t = np.asarray(times_s, dtype=np.float64).ravel()
    s = np.asarray(horizontal_m, dtype=np.float64).ravel()
    if t.shape != s.shape:
        raise ValueError("times and horizontal must have the same length")
    if len(t) < 2:
        raise ValueError("need >= 2 samples to fit a trajectory")

    design = np.c_[np.ones_like(t), t]                       # [1, t]
    (s0, vs), *_ = np.linalg.lstsq(design, s, rcond=None)
    s_res = s - design @ np.array([s0, vs])

    if height_m is None:
        rmse = float(np.sqrt(np.mean(s_res ** 2)))
        return ProjectileFit(
            max_speed_ms=abs(float(vs)),
            launch_speed_ms=abs(float(vs)),
            launch_angle_deg=None,
            max_height_m=None,
            height_at_target_m=None,
            height_ci_m=None,
            rmse_m=rmse,
            n_samples=len(t),
        )

    z = np.asarray(height_m, dtype=np.float64).ravel()
    if z.shape != t.shape:
        raise ValueError("times and height must have the same length")
    # linear fit with g fixed:  z + 0.5*g*t^2 = z0 + vz*t
    y = z + 0.5 * g * t ** 2
    (z0, vz), *_ = np.linalg.lstsq(design, y, rcond=None)
    z_pred = z0 + vz * t - 0.5 * g * t ** 2
    z_res = z - z_pred

    # apex height (clamped to the observed window if the apex is outside it)
    t_apex = vz / g if g > 0 else 0.0
    if t.min() <= t_apex <= t.max():
        max_height = z0 + vz * t_apex - 0.5 * g * t_apex ** 2
    else:
        max_height = float(max(z_pred.max(), z0))

    # max speed over the observed window (speed grows away from the apex)
    grid = np.linspace(t.min(), t.max(), 200)
    speeds = np.sqrt(vs ** 2 + (vz - g * grid) ** 2)
    max_speed = float(speeds.max())
    launch_speed = float(np.hypot(vs, vz))
    launch_angle = float(np.degrees(np.arctan2(vz, vs)))

    height_at_target = None
    if target_horizontal_m is not None and abs(vs) > 1e-9:
        t_tgt = (target_horizontal_m - s0) / vs
        height_at_target = float(z0 + vz * t_tgt - 0.5 * g * t_tgt ** 2)

    # confidence interval on the height estimate from the vertical residual std
    dof = max(1, len(t) - 2)
    sigma = float(np.sqrt((z_res ** 2).sum() / dof))
    ci = (float(max_height - 1.96 * sigma), float(max_height + 1.96 * sigma))
    rmse = float(np.sqrt(np.mean(s_res ** 2 + z_res ** 2)))

    return ProjectileFit(
        max_speed_ms=max_speed,
        launch_speed_ms=launch_speed,
        launch_angle_deg=launch_angle,
        max_height_m=float(max_height),
        height_at_target_m=height_at_target,
        height_ci_m=ci,
        rmse_m=rmse,
        n_samples=len(t),
    )


def _as_frames_xy(ball_track: Iterable) -> tuple[np.ndarray, np.ndarray]:
    """Normalize a ball track (BallSample list or (frame_idx, x, y) rows) -> arrays."""
    frames, xy = [], []
    for item in ball_track:
        if isinstance(item, BallSample):
            frames.append(item.frame_idx)
            xy.append((item.x_m, item.y_m))
        else:
            f, x, y = item
            frames.append(f)
            xy.append((x, y))
    if len(frames) < 2:
        raise ValueError("need >= 2 ball samples")
    return np.asarray(frames, dtype=np.float64), np.asarray(xy, dtype=np.float64)


def project_to_horizontal(track_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project a (near-straight) ground track onto its delivery direction.

    Returns (horizontal_distance_from_start, unit_direction). For a projectile
    the ground track is a straight line, so distance-along-direction is the
    horizontal coordinate the model needs.
    """
    p = np.asarray(track_xy, dtype=np.float64).reshape(-1, 2)
    start = p[0]
    direction = p[-1] - p[0]
    norm = float(np.linalg.norm(direction))
    if norm < 1e-9:
        return np.linalg.norm(p - start, axis=1), np.array([1.0, 0.0])
    u = direction / norm
    return (p - start) @ u, u


def reconstruct_trajectory(
    ball_track: Iterable,
    fps: float,
    heights_m: Sequence[float] | None = None,
    target_xy: tuple[float, float] | None = None,
    g: float = GRAVITY,
) -> ProjectileFit:
    """Fit the projectile model to an I8-style pitch-plane ball track.

    ``ball_track`` is BallSample objects or (frame_idx, x, y) rows in metres.
    Times come from frame indices and ``fps``; horizontal distance is measured
    along the delivery direction. ``heights_m`` (optional, aligned to the track)
    feeds the vertical fit; ``target_xy`` is the delivery target for
    height-at-target.
    """
    if fps <= 0:
        raise ValueError("fps must be positive")
    frames, xy = _as_frames_xy(ball_track)
    times = (frames - frames[0]) / fps
    horizontal, u = project_to_horizontal(xy)
    target_h = None
    if target_xy is not None:
        target_h = float((np.asarray(target_xy, dtype=np.float64) - xy[0]) @ u)
    return fit_projectile(times, horizontal, heights_m, target_horizontal_m=target_h, g=g)
