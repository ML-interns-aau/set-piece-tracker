"""Task 4 - t_kick / t_contact detection off a manufactured ball-speed signal."""

from __future__ import annotations

import numpy as np

from src.geometry.key_moments import (
    compute_speed,
    detect_key_moments,
    detect_t_contact,
    detect_t_kick,
)


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
