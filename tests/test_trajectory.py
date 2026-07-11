"""Task 3 - projectile-model fit against synthetic trajectories."""

from __future__ import annotations

import numpy as np

from src.geometry.trajectory import (
    GRAVITY,
    fit_projectile,
    reconstruct_trajectory,
    simulate_projectile,
)


def _analytic_max_height(v: float, angle_deg: float, g: float = GRAVITY) -> float:
    vz = v * np.sin(np.radians(angle_deg))
    return vz ** 2 / (2 * g)


def test_fit_recovers_clean_projectile():
    v, angle = 20.0, 45.0
    t, s, z = simulate_projectile(v, angle, n=30)
    fit = fit_projectile(t, s, z, target_horizontal_m=float(s[-1]))

    np.testing.assert_allclose(fit.max_height_m, _analytic_max_height(v, angle), atol=0.05)
    np.testing.assert_allclose(fit.launch_speed_ms, v, atol=0.05)
    np.testing.assert_allclose(fit.launch_angle_deg, angle, atol=0.5)
    # symmetric flight: speed at the endpoints equals launch speed -> that's the max
    np.testing.assert_allclose(fit.max_speed_ms, v, atol=0.1)
    # target = full range -> ball back on the ground
    assert abs(fit.height_at_target_m) < 0.05


def test_fit_is_robust_to_noise_and_ci_contains_truth():
    rng = np.random.default_rng(1)
    v, angle = 25.0, 38.0
    t, s, z = simulate_projectile(v, angle, n=40)
    z_noisy = z + rng.normal(0, 0.15, z.shape)   # ~15 cm height noise
    s_noisy = s + rng.normal(0, 0.10, s.shape)

    fit = fit_projectile(t, s_noisy, z_noisy)
    truth = _analytic_max_height(v, angle)
    np.testing.assert_allclose(fit.max_height_m, truth, atol=0.6)
    lo, hi = fit.height_ci_m
    assert lo <= fit.max_height_m <= hi


def test_height_none_returns_horizontal_only():
    v, angle = 18.0, 40.0
    t, s, _ = simulate_projectile(v, angle, n=20)
    fit = fit_projectile(t, s, None)
    assert fit.max_height_m is None
    assert fit.height_at_target_m is None
    assert fit.height_ci_m is None
    # horizontal speed component
    np.testing.assert_allclose(fit.max_speed_ms, v * np.cos(np.radians(angle)), atol=0.05)


def test_reconstruct_from_pitch_track():
    v, angle, fps = 22.0, 42.0, 25.0
    t, s, z = simulate_projectile(v, angle, n=30)
    # lay the ground track along a diagonal delivery direction
    direction = np.array([0.8, 0.6])
    start = np.array([0.5, 2.0])
    xy = start + np.outer(s, direction)
    frames = np.arange(len(t)) * (fps * (t[1] - t[0]))   # frame indices for these times
    track = [(int(round(f)), float(x), float(y)) for f, (x, y) in zip(frames, xy)]

    fit = reconstruct_trajectory(track, fps, heights_m=z, target_xy=tuple(xy[-1]))
    np.testing.assert_allclose(fit.max_height_m, _analytic_max_height(v, angle), atol=0.1)
    np.testing.assert_allclose(fit.launch_speed_ms, v, atol=0.3)
    assert abs(fit.height_at_target_m) < 0.1
