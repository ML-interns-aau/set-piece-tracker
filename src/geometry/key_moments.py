"""Key-moment detection: t_kick and t_contact (FR-010, FR-011).

Operates on the ball's ground-plane speed signal:

- **t_kick** -- the ball's motion onset: the first frame speed crosses a minimum
  threshold out of a near-zero (dead-ball) baseline, sustained for a few frames.
- **t_contact** -- the first post-kick discontinuity (abrupt speed change) in the
  otherwise smooth delivery, optionally gated on a player being near the ball to
  distinguish a genuine touch from a near-miss. ``None`` if the ball reaches no
  player (FR-011).

The detectors are pure functions over numpy arrays, so they unit-test directly
against a manufactured speed signal. Thresholds default to sensible values and
are pilot-tunable. When ``detect_key_moments`` cannot find a kick it returns
``None`` -- the signal to fall back to manual moment-tagging.
"""

from __future__ import annotations

import numpy as np

from src.domain.models import KeyMoments, Source


def compute_speed(positions_m: np.ndarray, fps: float) -> np.ndarray:
    """Ground-plane speed (m/s) per frame from metric positions (N, 2).

    ``speed[0]`` is set equal to ``speed[1]`` (no earlier frame to difference).
    """
    if fps <= 0:
        raise ValueError("fps must be positive")
    p = np.asarray(positions_m, dtype=np.float64).reshape(-1, 2)
    if len(p) < 2:
        return np.zeros(len(p), dtype=np.float64)
    step = np.linalg.norm(np.diff(p, axis=0), axis=1) * fps
    return np.concatenate([[step[0]], step])


def detect_t_kick(
    speed: np.ndarray,
    speed_thresh_ms: float = 3.0,
    baseline_max_ms: float = 1.0,
    baseline_frames: int = 5,
    sustain_frames: int = 2,
) -> int | None:
    """First frame where speed crosses ``speed_thresh_ms`` out of a dead-ball baseline.

    Requires the preceding ``baseline_frames`` to average at or below
    ``baseline_max_ms`` (the ball was resting) and the motion to persist for
    ``sustain_frames`` (rejects one-frame detection jitter). Returns the frame
    index into ``speed``, or ``None`` if no valid onset is found.
    """
    s = np.asarray(speed, dtype=np.float64).ravel()
    n = len(s)
    for i in range(1, n):
        if s[i] < speed_thresh_ms:
            continue
        pre = s[max(0, i - baseline_frames):i]
        if pre.size and pre.mean() > baseline_max_ms:
            continue  # ball was not at rest before this frame
        hi = min(n, i + sustain_frames)
        if np.all(s[i:hi] >= speed_thresh_ms * 0.5):
            return i
    return None


def detect_t_contact(
    speed: np.ndarray,
    kick_frame: int,
    jerk_thresh_ms: float = 2.5,
    min_frames_after: int = 2,
    player_near_ball: np.ndarray | None = None,
) -> int | None:
    """First abrupt speed change after the kick -- a touch (FR-011).

    The delivery's ground-plane speed is smooth (roughly constant horizontal
    velocity), so a contact shows up as a large frame-to-frame change (a trap
    slows it, a redirect/flick changes it). If ``player_near_ball`` (a per-frame
    boolean array) is given, the change must coincide with a player. Returns the
    frame index, or ``None`` if no contact is found (ball reaches no player).
    """
    s = np.asarray(speed, dtype=np.float64).ravel()
    n = len(s)
    start = max(kick_frame + min_frames_after, 1)
    for i in range(start, n):
        if abs(s[i] - s[i - 1]) >= jerk_thresh_ms:
            if player_near_ball is None or bool(np.asarray(player_near_ball).ravel()[i]):
                return i
    return None


def detect_key_moments(
    positions_m: np.ndarray,
    fps: float,
    frame_offset: int = 0,
    player_near_ball: np.ndarray | None = None,
    speed_thresh_ms: float = 3.0,
    jerk_thresh_ms: float = 2.5,
) -> KeyMoments | None:
    """Convenience: metric ball track -> :class:`KeyMoments` in absolute frames.

    ``frame_offset`` is the absolute frame index of ``positions_m[0]``. Returns
    ``None`` when no kick is detected (caller falls back to manual tagging);
    ``t_contact_frame`` is ``None`` when no contact is found.
    """
    speed = compute_speed(positions_m, fps)
    kick = detect_t_kick(speed, speed_thresh_ms=speed_thresh_ms)
    if kick is None:
        return None
    contact = detect_t_contact(
        speed, kick, jerk_thresh_ms=jerk_thresh_ms, player_near_ball=player_near_ball
    )
    return KeyMoments(
        t_kick_frame=frame_offset + kick,
        t_contact_frame=(frame_offset + contact) if contact is not None else None,
        t_kick_source=Source.AUTO,
        t_contact_source=Source.AUTO,
    )
