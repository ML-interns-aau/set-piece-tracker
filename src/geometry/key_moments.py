"""Key-moment detection: t_kick and t_contact (FR-010, FR-011).

Operates on the ball's ground-plane speed signal:

- **t_kick** -- the ball's motion onset: the first frame speed crosses a minimum
  threshold out of a near-zero (dead-ball) baseline, sustained for a few frames.
  Optionally cross-checked against the taker's foot being near the ball
  (``taker_dist_m``) to reject a motion blip caused by detector/calibration
  noise rather than an actual kick.
- **t_contact** -- the first post-kick discontinuity in the otherwise smooth
  delivery, optionally gated on a player being near the ball to distinguish a
  genuine touch from a near-miss. ``None`` if the ball reaches no player
  (FR-011). The discontinuity is measured as a change in the *velocity
  vector* (``compute_delta_v``), not just its magnitude, so a deflection that
  changes heading at constant speed is still caught.

Two detector families live here:

- **Threshold detectors** (``detect_t_kick`` / ``detect_t_contact``): the
  original simple detectors -- first threshold crossing out of a dead-ball
  baseline, first discontinuity after the kick. Robust on short/clean signals;
  used as the fallback.
- **ELASTIC-style detectors** (``detect_t_kick_elastic`` /
  ``detect_t_contact_elastic``): a numpy port of the candidate-generation +
  scoring approach from **ELASTIC** (Kim et al., MLSA 2025,
  https://github.com/hyunsungkim-ds/elastic, MPL-2.0 -- reference copy
  vendored under ``third_party/ELASTIC``): Savitzky-Golay-smoothed vector
  acceleration, candidate frames from acceleration peaks and player-distance
  valleys, +-3-frame refinement, hard gates (player proximity, dead-ball
  baseline), then a weighted linear score picks the best candidate. Adapted
  for this pipeline: no event data (candidates come from the signals alone),
  no ball height (monocular), anonymous nearest-player distances instead of
  named receivers.

The detectors are pure functions over numpy arrays, so they unit-test directly
against a manufactured speed signal. Thresholds default to sensible values and
are pilot-tunable. When ``detect_key_moments`` cannot find a kick it returns
``None`` -- the signal to fall back to manual moment-tagging.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from src.domain.models import KeyMoments, Source
from src.geometry.signal import find_peaks, savgol_filter


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


def compute_velocity(positions_m: np.ndarray, fps: float) -> np.ndarray:
    """Ground-plane velocity vector (m/s) per frame from metric positions (N, 2).

    ``velocity[0]`` mirrors ``velocity[1]``, matching :func:`compute_speed`.
    """
    if fps <= 0:
        raise ValueError("fps must be positive")
    p = np.asarray(positions_m, dtype=np.float64).reshape(-1, 2)
    if len(p) < 2:
        return np.zeros((len(p), 2), dtype=np.float64)
    step = np.diff(p, axis=0) * fps
    return np.vstack([step[0], step])


def compute_delta_v(velocity_m_s: np.ndarray) -> np.ndarray:
    """Per-frame ``|v[i] - v[i-1]|`` from velocity vectors (N, 2).

    Unlike a scalar speed diff, this is nonzero for a pure direction change at
    constant speed (a deflection/flick), which is exactly the case a touch at
    ``t_contact`` needs to catch. ``delta_v[0]`` mirrors ``delta_v[1]``.
    """
    v = np.asarray(velocity_m_s, dtype=np.float64).reshape(-1, 2)
    if len(v) < 2:
        return np.zeros(len(v), dtype=np.float64)
    d = np.linalg.norm(np.diff(v, axis=0), axis=1)
    return np.concatenate([[d[0]], d])


def smooth_signal(x: np.ndarray, window: int = 3) -> np.ndarray:
    """Centered moving-median filter (numpy only) to reject single-frame jitter.

    A real detector's speed/velocity signal is noisier than the synthetic
    arrays this module is unit-tested against; running it through this filter
    before thresholding rejects one-off detector jitter without a new
    dependency. ``window < 2`` or too short a signal returns ``x`` unchanged.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    if window < 2 or len(x) < window:
        return x.copy()
    pad = window // 2
    xp = np.pad(x, pad, mode="edge")
    return np.array([np.median(xp[i:i + window]) for i in range(len(x))])


def compute_ball_accel(positions_m: np.ndarray, fps: float) -> np.ndarray:
    """Smoothed vector-acceleration magnitude (m/s^2) per frame, ELASTIC-style.

    Velocity components are Savitzky-Golay-smoothed (window 15, order 2), then
    differentiated and smoothed again (window 9, order 2); the result is the
    norm of the acceleration *vector*, so it spikes on both speed changes and
    direction-only deflections. Windows clamp down on short signals. Length
    matches ``positions_m`` (edge values are repeated, as in the reference
    implementation).
    """
    if fps <= 0:
        raise ValueError("fps must be positive")
    p = np.asarray(positions_m, dtype=np.float64).reshape(-1, 2)
    n = len(p)
    if n < 4:
        return np.zeros(n, dtype=np.float64)
    v = np.diff(p, axis=0) * fps                                  # (n-1, 2)
    vx = savgol_filter(v[:, 0], 15, 2)
    vy = savgol_filter(v[:, 1], 15, 2)
    ax = savgol_filter(np.diff(vx) * fps, 9, 2)                   # (n-2,)
    ay = savgol_filter(np.diff(vy) * fps, 9, 2)
    accel = np.sqrt(ax**2 + ay**2)
    return np.concatenate([[accel[0]], accel, [accel[-1]]])


def _clamp01(x: np.ndarray | float) -> np.ndarray | float:
    return np.clip(x, 0.0, 1.0)


def detect_t_kick_elastic(
    positions_m: np.ndarray,
    fps: float,
    taker_dist_m: np.ndarray | None = None,
    accel_prominence_ms2: float = 10.0,
    taker_gate_m: float = 3.0,
    baseline_max_ms: float = 1.0,
    baseline_s: float = 0.5,
    refine_frames: int = 3,
    kick_dist_window_s: float = 2.0,
    frame_delay_full_s: float = 5.0,
) -> int | None:
    """t_kick via ELASTIC-style candidate generation + scoring (FR-010).

    Candidates are peaks of the smoothed vector acceleration (a kick is a large
    acceleration out of a dead ball). Each candidate is refined over a
    +-``refine_frames`` window and must pass two hard gates:

    - **dead-ball baseline** -- mean raw speed over the preceding ``baseline_s``
      seconds at or below ``baseline_max_ms`` (this is a corner: the ball is at
      rest before the kick);
    - **taker proximity** -- if ``taker_dist_m`` is given, the nearest player
      must come within ``taker_gate_m`` around the candidate.

    Survivors are scored with equal-weight linear terms (each clamped 0-1, per
    the reference implementation's scoring): acceleration magnitude (/20),
    taker closeness (/3, when available), post-kick ball displacement
    (``kick_dist``, /5 -- after a real kick the ball departs), and earliness
    (a real corner kick is the first strong candidate). Returns the
    best-scoring frame index, or ``None`` (fall back to the threshold detector
    or manual tagging).
    """
    p = np.asarray(positions_m, dtype=np.float64).reshape(-1, 2)
    n = len(p)
    if n < 4:
        return None
    # Savitzky-Golay-smoothed speed for the dead-ball baseline: raw
    # frame-to-frame speed of a resting ball is dominated by detector jitter.
    v = np.diff(p, axis=0) * fps
    vx, vy = savgol_filter(v[:, 0], 15, 2), savgol_filter(v[:, 1], 15, 2)
    speed_smooth = np.concatenate([[0.0], np.sqrt(vx**2 + vy**2)])
    accel = compute_ball_accel(p, fps)
    td = np.asarray(taker_dist_m, dtype=np.float64).ravel() if taker_dist_m is not None else None

    candidates = find_peaks(accel, prominence=accel_prominence_ms2, distance=10)
    if len(candidates) == 0:
        return None

    baseline_frames = max(1, int(round(baseline_s * fps)))
    baseline_guard = 8  # half the savgol window: keep kick-onset smear out of the baseline
    kick_window = max(1, int(round(kick_dist_window_s * fps)))
    delay_full = max(1, int(round(frame_delay_full_s * fps)))

    best_frame, best_score = None, -np.inf
    for i in candidates:
        lo, hi = max(0, i - refine_frames), min(n, i + refine_frames + 1)
        pre = speed_smooth[max(0, i - baseline_frames - baseline_guard):max(0, i - baseline_guard)]
        if pre.size and pre.mean() > baseline_max_ms:
            continue
        # taker-proximity gate + closeness score
        if td is not None:
            near = td[lo:hi].min()
            if near > taker_gate_m:
                continue
            dist_score = 1.0 - _clamp01(near / 3.0)
        else:
            dist_score = 0.5  # neutral when no player signal is available
        accel_score = _clamp01(accel[lo:hi].max() / 20.0)
        depart = np.linalg.norm(p[i:min(n, i + kick_window)] - p[i], axis=1).max()
        kick_dist_score = _clamp01(depart / 5.0)
        delay_score = 1.0 - _clamp01(i / delay_full)
        score = accel_score + dist_score + kick_dist_score + delay_score
        if score > best_score:
            best_frame, best_score = int(i), score
    return best_frame


def detect_t_contact_elastic(
    positions_m: np.ndarray,
    fps: float,
    kick_frame: int,
    player_dist_m: np.ndarray,
    min_frames_after: int = 2,
    dist_prominence_m: float = 0.5,
    accel_prominence_ms2: float = 10.0,
    contact_gate_m: float = 2.0,
    refine_frames: int = 3,
) -> int | None:
    """t_contact via ELASTIC-style receive detection (FR-011).

    Port of the reference implementation's receive detector: candidate frames
    are valleys of the nearest-player distance and peaks of the smoothed
    vector acceleration after the kick (plus the final frame -- the ball can
    reach a player as the clip ends). Candidates are refined over
    +-``refine_frames`` and hard-gated on the nearest player being within
    ``contact_gate_m``. With several survivors, the score combines
    acceleration magnitude (/20), player closeness (/3), and *approach*
    (how far the ball travelled toward the eventual receiver since the
    previous candidate, /5). Returns ``None`` when no candidate passes the
    gate -- the ball reached no player.
    """
    p = np.asarray(positions_m, dtype=np.float64).reshape(-1, 2)
    n = len(p)
    dist = np.asarray(player_dist_m, dtype=np.float64).ravel()
    if len(dist) != n:
        raise ValueError("player_dist_m must have one value per position")
    start = max(kick_frame + min_frames_after, 1)
    if n - start < 2:
        return None

    accel = compute_ball_accel(p, fps)
    w_dist, w_accel = dist[start:], accel[start:]

    candidates = set(find_peaks(-w_dist, prominence=dist_prominence_m).tolist())
    for i in find_peaks(w_accel, prominence=accel_prominence_ms2, distance=10):
        if not candidates & set(range(i - refine_frames, i + refine_frames + 1)):
            candidates.add(int(i))
    candidates.add(len(w_dist) - 1)
    candidates = np.sort(np.fromiter(candidates, dtype=np.intp))

    refined = []
    for i in candidates:
        lo, hi = max(0, i - refine_frames), min(len(w_dist), i + refine_frames + 1)
        near = w_dist[lo:hi].min()
        if near > contact_gate_m:
            continue
        refined.append((int(i), near, w_accel[lo:hi].max()))
    if not refined:
        return None
    if len(refined) == 1:
        return start + refined[0][0]

    best_frame, best_score = None, -np.inf
    prev = 0
    for i, near, acc in refined:
        approach = w_dist[prev:i + 1].max() - w_dist[i]
        prev = i
        score = (
            _clamp01(acc / 20.0)
            + (1.0 - _clamp01(near / 3.0))
            + _clamp01(approach / 5.0)
        )
        if score > best_score:
            best_frame, best_score = start + i, score
    return best_frame


def min_player_distance(
    ball_xy: np.ndarray, player_xy_list: Sequence[Sequence[float]] | None
) -> float:
    """Distance (m) from the ball to the nearest of a frame's player points.

    ``player_xy_list`` is whatever (variable-length, possibly empty) list of
    per-frame player foot-points a caller has for that frame -- this module
    doesn't need to know how they were produced. Returns ``inf`` when the
    list is empty/``None``, so the caller's gate simply never passes.
    """
    if player_xy_list is None or len(player_xy_list) == 0:
        return float("inf")
    pts = np.asarray(player_xy_list, dtype=np.float64).reshape(-1, 2)
    return float(np.min(np.linalg.norm(pts - np.asarray(ball_xy, dtype=np.float64), axis=1)))


def detect_t_kick(
    speed: np.ndarray,
    speed_thresh_ms: float = 3.0,
    baseline_max_ms: float = 1.0,
    baseline_frames: int = 5,
    sustain_frames: int = 2,
    taker_dist_m: np.ndarray | None = None,
    taker_gate_m: float = 2.5,
) -> int | None:
    """First frame where speed crosses ``speed_thresh_ms`` out of a dead-ball baseline.

    Requires the preceding ``baseline_frames`` to average at or below
    ``baseline_max_ms`` (the ball was resting) and the motion to persist for
    ``sustain_frames`` (rejects one-frame detection jitter). Returns the frame
    index into ``speed``, or ``None`` if no valid onset is found.

    If ``taker_dist_m`` (per-frame distance from the ball to the nearest
    player, e.g. from :func:`min_player_distance`) is given, an onset also
    requires the taker's foot to have been within ``taker_gate_m`` at some
    point in the ``sustain_frames`` window leading up to it -- a cross-check
    against a motion blip from detector/calibration noise. A small window
    (not an exact-frame match) accounts for foot contact leading the ball
    centre's measured motion by a frame or two.
    """
    s = np.asarray(speed, dtype=np.float64).ravel()
    n = len(s)
    td = np.asarray(taker_dist_m, dtype=np.float64).ravel() if taker_dist_m is not None else None
    for i in range(1, n):
        if s[i] < speed_thresh_ms:
            continue
        pre = s[max(0, i - baseline_frames):i]
        if pre.size and pre.mean() > baseline_max_ms:
            continue  # ball was not at rest before this frame
        hi = min(n, i + sustain_frames)
        if not np.all(s[i:hi] >= speed_thresh_ms * 0.5):
            continue
        if td is not None and td[max(0, i - sustain_frames):i + 1].min() > taker_gate_m:
            continue  # motion onset isn't near the taker -- likely noise
        return i
    return None


def detect_t_contact(
    speed: np.ndarray,
    kick_frame: int,
    jerk_thresh_ms: float = 2.5,
    min_frames_after: int = 2,
    player_near_ball: np.ndarray | None = None,
    delta_v: np.ndarray | None = None,
) -> int | None:
    """First abrupt discontinuity after the kick -- a touch (FR-011).

    The delivery is otherwise smooth (roughly constant horizontal velocity),
    so a contact shows up as a large frame-to-frame change. By default that's
    measured as ``|Δspeed|``; passing ``delta_v`` (from :func:`compute_delta_v`)
    measures the velocity-*vector* change instead, which also catches a
    deflection that changes heading without changing speed. If
    ``player_near_ball`` (a per-frame boolean array) is given, the change must
    coincide with a player. Returns the frame index, or ``None`` if no contact
    is found (ball reaches no player).
    """
    s = np.asarray(speed, dtype=np.float64).ravel()
    n = len(s)
    if delta_v is not None:
        disc = np.asarray(delta_v, dtype=np.float64).ravel()
    else:
        disc = np.abs(np.diff(s, prepend=s[0]))
    start = max(kick_frame + min_frames_after, 1)
    for i in range(start, n):
        if disc[i] >= jerk_thresh_ms:
            if player_near_ball is None or bool(np.asarray(player_near_ball).ravel()[i]):
                return i
    return None


def detect_key_moments(
    positions_m: np.ndarray,
    fps: float,
    frame_offset: int = 0,
    player_near_ball: np.ndarray | None = None,
    player_dist_m: np.ndarray | None = None,
    taker_dist_m: np.ndarray | None = None,
    speed_thresh_ms: float = 3.0,
    jerk_thresh_ms: float = 2.5,
    taker_gate_m: float = 2.5,
    contact_gate_m: float = 2.0,
    use_delta_v: bool = True,
    elastic_min_frames: int = 25,
) -> KeyMoments | None:
    """Convenience: metric ball track -> :class:`KeyMoments` in absolute frames.

    ``frame_offset`` is the absolute frame index of ``positions_m[0]``. Returns
    ``None`` when no kick is detected (caller falls back to manual tagging);
    ``t_contact_frame`` is ``None`` when no contact is found.

    Tracks of at least ``elastic_min_frames`` frames use the ELASTIC-style
    detectors (:func:`detect_t_kick_elastic` / :func:`detect_t_contact_elastic`
    -- the contact detector additionally needs ``player_dist_m``, the per-frame
    nearest-player distance, e.g. built with :func:`min_player_distance`).
    Shorter tracks -- and any case where the ELASTIC kick detector finds no
    candidate -- fall back to the threshold detectors, where ``taker_dist_m``
    cross-checks ``t_kick`` (see :func:`detect_t_kick`), ``player_dist_m`` is
    thresholded by ``contact_gate_m`` into the ``player_near_ball`` gate, and
    ``use_delta_v`` measures the contact discontinuity as a velocity-*vector*
    change (:func:`compute_delta_v`).
    """
    positions_m = np.asarray(positions_m, dtype=np.float64).reshape(-1, 2)
    elastic = len(positions_m) >= elastic_min_frames

    kick = None
    if elastic:
        kick = detect_t_kick_elastic(positions_m, fps, taker_dist_m=taker_dist_m)
    if kick is None:
        speed = compute_speed(positions_m, fps)
        kick = detect_t_kick(
            speed, speed_thresh_ms=speed_thresh_ms,
            taker_dist_m=taker_dist_m, taker_gate_m=taker_gate_m,
        )
        if kick is None:
            return None

    if elastic and player_dist_m is not None:
        contact = detect_t_contact_elastic(
            positions_m, fps, kick,
            player_dist_m=player_dist_m, contact_gate_m=contact_gate_m,
        )
    else:
        speed = compute_speed(positions_m, fps)
        delta_v = None
        if use_delta_v:
            delta_v = compute_delta_v(compute_velocity(positions_m, fps))
        if player_near_ball is None and player_dist_m is not None:
            player_near_ball = np.asarray(player_dist_m, dtype=np.float64) <= contact_gate_m
        contact = detect_t_contact(
            speed,
            kick,
            jerk_thresh_ms=jerk_thresh_ms,
            player_near_ball=player_near_ball,
            delta_v=delta_v,
        )
    return KeyMoments(
        t_kick_frame=frame_offset + kick,
        t_contact_frame=(frame_offset + contact) if contact is not None else None,
        t_kick_source=Source.AUTO,
        t_contact_source=Source.AUTO,
    )
