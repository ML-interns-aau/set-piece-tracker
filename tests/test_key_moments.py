"""Task 4 - t_kick / t_contact detection off a manufactured ball-speed signal."""

from __future__ import annotations

import numpy as np

from src.geometry.key_moments import (
    compute_ball_accel,
    compute_delta_v,
    compute_speed,
    compute_velocity,
    detect_key_moments,
    detect_t_contact,
    detect_t_contact_elastic,
    detect_t_kick,
    detect_t_kick_elastic,
    min_player_distance,
    smooth_signal,
)

FPS = 25.0


def corner_scenario(kick=50, contact=110, n=150, jitter=0.03, seed=7):
    """Synthetic corner at 25 fps with realistic detector jitter.

    Ball dead at the corner arc, kicked toward the near-post area at ``kick``,
    trapped by a receiver at ``contact``. Returns (positions, taker_dist,
    nearest_player_dist) shaped like the real pipeline's inputs.
    """
    rng = np.random.default_rng(seed)
    pos = np.zeros((n, 2))
    pos[:kick] = [0.3, 0.3]
    target = np.array([6.0, 30.0])
    v = (target - [0.3, 0.3]) / ((contact - kick) / FPS)
    for i in range(kick, contact):
        pos[i] = [0.3, 0.3] + v * (i - kick) / FPS
    v_after = np.array([1.0, 0.5])  # slow roll after the trap
    for i in range(contact, n):
        pos[i] = pos[contact - 1] + v_after * (i - contact + 1) / FPS
    pos += rng.normal(0, jitter, pos.shape)
    taker_dist = np.zeros(n)
    taker_dist[:kick] = np.linspace(3.0, 0.4, kick)  # taker walks up to the ball
    for i in range(kick, n):
        taker_dist[i] = np.linalg.norm(pos[i] - [0.5, 0.5])  # taker stays at the corner
    receiver = np.array([6.2, 30.2])
    nearest_dist = np.minimum(taker_dist, np.linalg.norm(pos - receiver, axis=1))
    return pos, taker_dist, nearest_dist


def test_compute_speed_constant_motion():
    # move 1 m per frame at 10 fps -> 10 m/s
    positions = np.array([[i, 0.0] for i in range(6)])
    speed = compute_speed(positions, fps=10.0)
    assert len(speed) == 6
    np.testing.assert_allclose(speed, 10.0)


def test_detect_t_kick_from_dead_ball():
    speed = np.array([0.2, 0.1, 0.0, 0.1, 0.0, 5.0, 6.0, 6.1, 6.0])
    assert detect_t_kick(speed) == 5


def test_no_kick_when_ball_never_moves():
    speed = np.zeros(20)
    assert detect_t_kick(speed) is None


def test_transient_spike_is_rejected():
    # single-frame blip then back to rest -> not a sustained kick
    speed = np.array([0.0, 0.0, 0.0, 8.0, 0.0, 0.0, 0.0])
    assert detect_t_kick(speed, sustain_frames=2) is None


def test_detect_t_contact_on_speed_discontinuity():
    # kick at 5, smooth flight ~6 m/s, abrupt stop (trap) at index 11
    speed = np.array([0, 0, 0, 0, 0, 6, 6, 6, 6, 6, 6, 1.0, 1.0], dtype=float)
    kick = detect_t_kick(speed)
    assert kick == 5
    assert detect_t_contact(speed, kick) == 11


def test_t_contact_none_when_no_touch():
    speed = np.array([0, 0, 0, 0, 0, 6, 6, 6, 6, 6, 6], dtype=float)
    assert detect_t_contact(speed, kick_frame=5) is None


def test_t_contact_player_gating():
    speed = np.array([0, 0, 0, 0, 0, 6, 6, 6, 6, 6, 6, 1.0, 1.0, 8.0], dtype=float)
    # a player is only near the ball at the later discontinuity (index 13)
    near = np.zeros(len(speed), dtype=bool)
    near[13] = True
    assert detect_t_contact(speed, kick_frame=5, player_near_ball=near) == 13


def test_detect_key_moments_end_to_end():
    # 5 dead-ball frames, then move 0.6 m/frame at 10 fps (=6 m/s), then stop
    pts = [[0.0, 0.0]] * 5
    for i in range(1, 7):
        pts.append([0.6 * i, 0.0])
    pts.append([3.6, 0.0])   # stopped -> contact
    positions = np.array(pts)

    km = detect_key_moments(positions, fps=10.0, frame_offset=100)
    assert km is not None
    assert km.t_kick_frame == 105
    assert km.t_contact_frame == 111


def _redirect_track():
    """5 dead-ball frames, +x at 5 m/s, then redirected to +y at the same 5 m/s.

    The redirect (frame 10) changes heading but not speed magnitude -- a
    scalar |delta speed| diff sees nothing, a vector delta_v does.
    """
    pts = [[0.0, 0.0]] * 5
    for i in range(1, 6):
        pts.append([0.5 * i, 0.0])
    for i in range(1, 4):
        pts.append([2.5, 0.5 * i])
    return np.array(pts)


def test_compute_velocity_and_delta_v_catch_direction_change():
    positions = _redirect_track()
    speed = compute_speed(positions, fps=10.0)
    velocity = compute_velocity(positions, fps=10.0)
    delta_v = compute_delta_v(velocity)

    # speed magnitude is constant (~5 m/s) straight through the redirect
    np.testing.assert_allclose(speed[5:], 5.0)
    # but the velocity vector -- and hence delta_v -- spikes exactly at the turn
    assert delta_v[10] > 5.0
    assert delta_v[9] < 1.0
    assert delta_v[11] < 1.0


def test_detect_t_contact_direction_change_only():
    positions = _redirect_track()
    speed = compute_speed(positions, fps=10.0)
    kick = detect_t_kick(speed)
    assert kick == 5

    # a scalar speed diff misses a pure direction change entirely
    assert detect_t_contact(speed, kick) is None

    # the vector-based delta_v catches it at the turn frame
    delta_v = compute_delta_v(compute_velocity(positions, fps=10.0))
    assert detect_t_contact(speed, kick, delta_v=delta_v) == 10


def test_detect_key_moments_uses_delta_v_by_default():
    positions = _redirect_track()
    km = detect_key_moments(positions, fps=10.0, frame_offset=0)
    assert km is not None
    assert km.t_kick_frame == 5
    assert km.t_contact_frame == 10

    # gating on a player that's never near the ball suppresses the contact
    km_far_player = detect_key_moments(
        positions,
        fps=10.0,
        frame_offset=0,
        player_dist_m=np.full(len(positions), 10.0),
        contact_gate_m=2.0,
    )
    assert km_far_player.t_contact_frame is None


def test_detect_t_kick_taker_cross_check_rejects_spurious_motion():
    # an early spurious blip (e.g. detector/calibration noise) followed by the
    # real kick onset later in the clip
    speed = np.array([0.1] * 5 + [5.0, 6.0] + [0.1] * 5 + [5.0, 6.0, 6.0])

    # without a taker cross-check, the earlier spurious blip wins
    assert detect_t_kick(speed) == 5

    taker_dist_m = np.full(len(speed), 5.0)
    taker_dist_m[10:13] = 0.5  # taker's foot is only near the ball for the real kick
    assert detect_t_kick(speed, taker_dist_m=taker_dist_m, taker_gate_m=2.5) == 12


def test_min_player_distance():
    ball = np.array([0.0, 0.0])
    assert min_player_distance(ball, [[3.0, 4.0], [1.0, 0.0]]) == 1.0
    assert min_player_distance(ball, []) == float("inf")
    assert min_player_distance(ball, None) == float("inf")


def test_smooth_signal_rejects_single_frame_outlier():
    x = np.array([0.1, 0.1, 0.1, 9.0, 0.1, 0.1, 0.1])
    smoothed = smooth_signal(x, window=3)
    assert smoothed[3] < 1.0
    np.testing.assert_allclose(smoothed[[0, 1, 2, 4, 5, 6]], 0.1)


def test_smooth_signal_passthrough_when_window_too_small_or_signal_short():
    x = np.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(smooth_signal(x, window=1), x)
    np.testing.assert_allclose(smooth_signal(np.array([1.0]), window=5), [1.0])


# --- ELASTIC-style detectors (see module docstring for attribution) -----------

def test_compute_ball_accel_spikes_at_kick_and_contact():
    pos, _, _ = corner_scenario()
    accel = compute_ball_accel(pos, FPS)
    assert len(accel) == len(pos)
    assert accel[50] > 30.0          # kick: dead ball -> ~8 m/s
    assert accel[109] > 30.0         # trap: ~8 m/s -> ~1 m/s
    assert accel[20:40].max() < 10.0  # dead-ball stretch stays quiet
    assert accel[70:100].max() < 10.0  # smooth flight stays quiet


def test_elastic_kick_on_synthetic_corner():
    pos, taker, _ = corner_scenario()
    kick = detect_t_kick_elastic(pos, FPS, taker_dist_m=taker)
    assert kick is not None and abs(kick - 50) <= 1


def test_elastic_kick_works_without_taker_signal():
    pos, _, _ = corner_scenario()
    kick = detect_t_kick_elastic(pos, FPS)
    assert kick is not None and abs(kick - 50) <= 1


def test_elastic_kick_gated_when_taker_never_near():
    pos, _, _ = corner_scenario()
    assert detect_t_kick_elastic(pos, FPS, taker_dist_m=np.full(len(pos), 9.0)) is None


def test_elastic_contact_on_synthetic_corner():
    pos, taker, near = corner_scenario()
    kick = detect_t_kick_elastic(pos, FPS, taker_dist_m=taker)
    contact = detect_t_contact_elastic(pos, FPS, kick, player_dist_m=near)
    assert contact is not None and abs(contact - 110) <= 2


def test_elastic_contact_none_when_ball_reaches_no_player():
    pos, taker, _ = corner_scenario()
    near = taker.copy()
    near[52:] = 8.0  # nobody near the ball after the kick
    assert detect_t_contact_elastic(pos, FPS, 50, player_dist_m=near) is None


def test_elastic_contact_ignores_midflight_detector_glitch():
    pos, _, near = corner_scenario()
    pos[80] += [0.35, -0.3]  # single-frame ball-detector glitch, no player near
    contact = detect_t_contact_elastic(pos, FPS, 50, player_dist_m=near)
    assert contact is not None and abs(contact - 110) <= 2


def test_elastic_contact_catches_pure_deflection():
    # flick-on at constant speed: heading changes 90 degrees, magnitude doesn't
    pos, taker, _ = corner_scenario(jitter=0.0)
    v = (np.array([6.0, 30.0]) - [0.3, 0.3]) / (60 / FPS)
    new_v = np.array([-v[1], v[0]])  # rotate 90 degrees, same speed
    for i in range(110, len(pos)):
        pos[i] = pos[109] + new_v * (i - 109) / FPS
    pos += np.random.default_rng(1).normal(0, 0.03, pos.shape)
    near = np.minimum(taker, np.linalg.norm(pos - pos[109], axis=1))
    contact = detect_t_contact_elastic(pos, FPS, 50, player_dist_m=near)
    assert contact is not None and abs(contact - 110) <= 2


def test_elastic_contact_prefers_real_touch_over_near_miss():
    # a player stands near the flight path at ~frame 85 (no trajectory change);
    # the real trap is at 110 -- proximity alone must not win
    pos, taker, _ = corner_scenario(jitter=0.02)
    near = np.minimum(taker, np.linalg.norm(pos - pos[84], axis=1))
    near = np.minimum(near, np.linalg.norm(pos - pos[109], axis=1))
    contact = detect_t_contact_elastic(pos, FPS, 50, player_dist_m=near)
    assert contact is not None and abs(contact - 110) <= 2


def test_detect_key_moments_routes_to_elastic_on_long_tracks():
    pos, taker, near = corner_scenario()
    km = detect_key_moments(pos, FPS, frame_offset=1000, taker_dist_m=taker, player_dist_m=near)
    assert km is not None
    assert abs(km.t_kick_frame - 1050) <= 1
    assert km.t_contact_frame is not None and abs(km.t_contact_frame - 1110) <= 2
