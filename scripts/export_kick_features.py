"""Export the first corner kick (`t_kick`): annotated snapshot + full feature JSON.

Runs the perception + geometry planes on one corner-kick clip, locates the first
kick (`t_kick`, ball motion onset), and at that moment produces:

  * `kick_moment.json` -- a self-describing record: clip metadata, corner side,
    calibration quality, the `t_kick`/`t_contact` frames+times, the corner
    taker, all 13 Appendix-A features, and the per-player position rows the
    features were computed from (interface I10 -> I11).
  * `t_kick.png` -- the `t_kick` frame with team-colored player boxes, GK
    highlight, ball marker, zone overlays, and a burned-in `t_kick` marker, so a
    human can confirm the moment and spot-check the counts in ~1-2 min (FR-017).

This wires already-built pieces: FootballDetector / FootballTracker /
TeamClassifier (perception), PnLCalib calibration + orientation + ball smoother +
key moments + trajectory (geometry), build_player_positions (I10),
build_feature_row (I11), and render_frame (verification overlay). It is a focused
single-clip tool, not the batch runner.

Honest status of the delivery features at t_kick: `pass_speed_max_in_ms` comes
from the trajectory fit; `pass_max_height_in_m` and `pass_hight_in_m_at_target`
are `null` -- they need monocular ball *height*, which is deferred (see the
plan / FR-009). Zone-occupancy features 1-10 are fully computed.

Needs the perception env (torch + ultralytics + `yolo11m.pt`) and the PnLCalib
weights (auto-downloaded on first use). Example:

    python scripts/export_kick_features.py --clip data/samples/sample.mp4
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root on path

from src.domain.models import KeyMoments, Moment, Source
from src.features.feature_rows import build_feature_row
from src.geometry.ball_smoother import BallSmoother
from src.geometry.calibration import apply_homography
from src.geometry.key_moments import detect_key_moments
from src.geometry.orientation import detect_corner_side, to_canonical
from src.geometry.positions import (
    GK_CODE,
    build_player_positions,
    resolve_attacking_side,
)
from src.geometry.trajectory import reconstruct_trajectory


def clip_metadata(cap: cv2.VideoCapture) -> tuple[float, int, tuple[int, int]]:
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return fps, n_frames, (w, h)


def read_frame(cap: cv2.VideoCapture, index: int) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"could not read frame {index}")
    return frame


def _best_ball_bbox(detections) -> np.ndarray | None:
    balls = detections[detections.class_id == 32]
    if len(balls) == 0:
        return None
    return balls.xyxy[int(np.argmax(balls.confidence))]


def run_perception(cap, device, conf, max_frames):
    """One forward pass: ball px track + per-frame player tracks + team-code history.

    Returns:
        ball_track:      list of (frame_idx, u, v, predicted)
        tracks_by_frame: {frame_idx: [(track_id, bbox_xyxy, det_conf), ...]}
        code_history:    {track_id: [team_code, ...]}  (per-frame classifier votes)
    """
    from src.engine.detector import FootballDetector  # lazy: needs torch/ultralytics
    from src.engine.team_classifier import TeamClassifier
    from src.engine.tracker import FootballTracker

    detector = FootballDetector(model_path="yolo11m.pt", conf=conf, device=device)
    tracker = FootballTracker()
    team_clf = TeamClassifier(detect_goalkeeper=True)
    smoother = BallSmoother()

    ball_track: list[tuple[int, float, float, bool]] = []
    tracks_by_frame: dict[int, list[tuple[int, tuple, float]]] = {}
    code_history: dict[int, list[int]] = defaultdict(list)
    started = False

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    idx = 0
    while True:
        if max_frames and idx >= max_frames:
            break
        ok, frame = cap.read()
        if not ok:
            break

        detections = detector.detect(frame)

        bbox = _best_ball_bbox(detections)
        cx, cy, predicted = smoother.update(frame, bbox)
        started = started or (bbox is not None)
        if started:
            ball_track.append((idx, cx, cy, predicted))

        tracked = tracker.update(detections)
        codes = team_clf.assign_teams(frame, tracked)
        players: list[tuple[int, tuple, float]] = []
        if tracked.tracker_id is None:
            tracks_by_frame[idx] = players
            idx += 1
            continue
        conf_arr = tracked.confidence
        for i, (bb, cid, tid) in enumerate(
            zip(tracked.xyxy, tracked.class_id, tracked.tracker_id)
        ):
            if int(cid) != 0 or tid is None:
                continue
            tid = int(tid)
            det_conf = float(conf_arr[i]) if conf_arr is not None else 1.0
            players.append((tid, (float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])), det_conf))
            code_history[tid].append(int(codes[i]))
        tracks_by_frame[idx] = players
        idx += 1

    return ball_track, tracks_by_frame, code_history


def gather_moment_tracks(tracks_by_frame, moment_frame, window):
    """Player tracks at ``moment_frame`` (nearest within +/-window), with gaps.

    Returns ``(tracks, gap_frames)`` where ``tracks`` is
    ``[(track_id, bbox, conf), ...]`` and ``gap_frames`` maps a track id to the
    frame distance used (0 = detected exactly at the moment).
    """
    per_track: dict[int, dict[int, tuple]] = defaultdict(dict)
    for f, plist in tracks_by_frame.items():
        for tid, bbox, det_conf in plist:
            per_track[tid][f] = (bbox, det_conf)

    tracks: list[tuple[int, tuple, float]] = []
    gaps: dict[int, float] = {}
    for tid, fmap in per_track.items():
        if moment_frame in fmap:
            bbox, det_conf = fmap[moment_frame]
            gap = 0
        else:
            cand = [f for f in fmap if abs(f - moment_frame) <= window]
            if not cand:
                continue
            nf = min(cand, key=lambda f: abs(f - moment_frame))
            bbox, det_conf = fmap[nf]
            gap = abs(nf - moment_frame)
        tracks.append((tid, bbox, det_conf))
        gaps[tid] = float(gap)
    return tracks, gaps


def ball_px_at(ball_track, frame_idx):
    """Ball pixel (u, v) at ``frame_idx`` (nearest available), or None."""
    if not ball_track:
        return None
    entry = min(ball_track, key=lambda e: abs(e[0] - frame_idx))
    return (entry[1], entry[2])


def render_snapshot(frame, tracks_at_kick, positions, ball_px, key_moments, calib,
                    clip_id, out_path):
    """Draw + write the annotated t_kick PNG (players, GK, ball, zones, marker)."""
    from src.verification.bridge import events_from_key_moments
    from src.verification.visualizer import FrameTracks, TrackedBox, render_frame

    pos_by_id = {p.player_id: p for p in positions}
    boxes: list[TrackedBox] = []
    for tid, bbox, _conf in tracks_at_kick:
        p = pos_by_id.get(tid)
        if p is None:
            continue
        boxes.append(TrackedBox(
            track_id=tid, bbox_xyxy=bbox, team=p.team,
            is_goalkeeper=p.is_goalkeeper, is_ball=False,
        ))
    if ball_px is not None:
        u, v = ball_px
        boxes.append(TrackedBox(
            track_id=-99, bbox_xyxy=(u - 6, v - 6, u + 6, v + 6),
            team=None, is_goalkeeper=False, is_ball=True,
        ))

    tracks = FrameTracks(frame_idx=key_moments.t_kick_frame, boxes=tuple(boxes))
    events = events_from_key_moments(
        clip_id, key_moments, positions,
        known_player_ids=frozenset(p.player_id for p in positions),
    )
    h_inv = np.linalg.inv(calib.H) if calib is not None else None
    rendered = render_frame(frame, tracks, events.events, h_inv)
    cv2.imwrite(str(out_path), rendered)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clip", required=True, help="path to a corner-kick clip")
    ap.add_argument("--frame", type=int, default=0, help="frame for corner-side detection")
    ap.add_argument("--device", default="cpu", help="YOLO device (cpu / 0 / cuda:0)")
    ap.add_argument("--conf", type=float, default=0.25, help="detection confidence")
    ap.add_argument("--max-frames", type=int, default=0, help="cap frames processed (0 = whole clip)")
    ap.add_argument("--out", default="outputs/kick_moment", help="output directory")
    args = ap.parse_args()

    clip_path = Path(args.clip)
    if not clip_path.exists():
        raise SystemExit(f"clip not found: {clip_path}")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    clip_id = clip_path.stem

    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        raise SystemExit(f"could not open clip: {clip_path}")
    fps, n_frames, (w, h) = clip_metadata(cap)
    print(f"clip      : {clip_path.name}  {w}x{h}  {fps:.2f} fps  {n_frames} frames")

    side = detect_corner_side(read_frame(cap, args.frame))
    print(f"corner    : {side.corner_side.value}  (source={side.side_source.value}, "
          f"confidence={side.confidence})")

    summary: dict = {
        "export": "kick_moment",
        "export_version": "0.1",
        "clip_id": clip_id,
        "fps": fps,
        "resolution": [w, h],
        "units": {"position": "m", "velocity": "m/s", "time": "s"},
        "coordinate_convention": (
            "canonical corner orientation; FIFA 105x68 model; analysed goal at x=0; meters"
        ),
        "corner_side": side.corner_side.value,
        "corner_side_source": side.side_source.value,
        "corner_side_confidence": side.confidence,
    }

    # --- perception pass (needs torch/ultralytics) ---
    try:
        ball_track, tracks_by_frame, code_history = run_perception(
            cap, args.device, args.conf, args.max_frames
        )
    except ImportError as e:
        cap.release()
        raise SystemExit(
            f"perception needs torch+ultralytics ({e}). Run with the perception venv's python."
        )

    # --- calibration (per-frame PnLCalib) ---
    from src.geometry.pnl_calibration import build_calibration_track
    upper = args.max_frames if args.max_frames else n_frames
    calib_track = build_calibration_track(clip_path, frame_indices=range(upper))
    if calib_track is not None:
        summary["calibration"] = {
            "reprojection_error_m": calib_track.mean_reprojection_error_m,
            "static": calib_track.static,
            "discontinuity_frames": list(calib_track.discontinuity_frames),
            "source": Source.AUTO.value,
        }
        print(f"calib     : per-frame track  static={calib_track.static}  "
              f"mean_reproj={calib_track.mean_reprojection_error_m:.3f} m")
    else:
        summary["calibration"] = None
        print("calib     : PnLCalib could not calibrate -- positions will lack metric coords")

    # --- ball metric track + key moments + trajectory ---
    if len(ball_track) < 2:
        summary["t_kick"] = None
        summary["status"] = "no_ball_track"
        _write(out_dir, summary)
        cap.release()
        raise SystemExit("too few ball detections -- try a lower --conf or another clip.")

    frame_ids = [e[0] for e in ball_track]
    px = np.array([[e[1], e[2]] for e in ball_track], dtype=np.float64)
    if calib_track is not None:
        mapped = np.vstack([
            apply_homography((calib_track.at(fid) or calib_track.at(frame_ids[0])).H,
                             px[i:i + 1])[0]
            for i, fid in enumerate(frame_ids)
        ])
        metric = to_canonical(mapped, side.corner_side)
    else:
        metric = None

    km = detect_key_moments(metric[:, :2], fps, frame_offset=frame_ids[0]) if metric is not None else None
    if km is None:
        summary["t_kick"] = None
        summary["status"] = "no_kick_detected"
        _write(out_dir, summary)
        cap.release()
        raise SystemExit("no kick detected (or no calibration) -- fall back to manual moment tagging.")

    t_c = km.t_contact_frame
    print(f"moments   : t_kick=frame {km.t_kick_frame} ({km.t_kick_frame/fps:.2f}s)  "
          f"t_contact={('frame ' + str(t_c)) if t_c is not None else 'None'}")

    # trajectory fit over kick -> contact (heights None -> max speed only)
    metric_track = [(frame_ids[i], float(metric[i, 0]), float(metric[i, 1]))
                    for i in range(len(frame_ids))]
    lo = frame_ids.index(km.t_kick_frame) if km.t_kick_frame in frame_ids else 0
    hi = (frame_ids.index(t_c) + 1) if (t_c in frame_ids) else len(metric_track)
    window = metric_track[lo:hi]
    fit = reconstruct_trajectory(window, fps) if len(window) >= 2 else None

    # --- positions at t_kick (interface I10) ---
    calib_at_kick = calib_track.at(km.t_kick_frame) if calib_track is not None else None
    tracks_at_kick, gaps = gather_moment_tracks(tracks_by_frame, km.t_kick_frame, window=int(fps // 2))
    team_codes = {tid: Counter(codes).most_common(1)[0][0] for tid, codes in code_history.items()}
    ballpx = ball_px_at(ball_track, km.t_kick_frame)
    attacking_code, taker_id = resolve_attacking_side(tracks_at_kick, ballpx, team_codes)
    gk_ids = frozenset(tid for tid, c in team_codes.items() if c == GK_CODE)

    positions = build_player_positions(
        tracks_at_kick, calib_at_kick, side.corner_side,
        clip_id=clip_id, moment=Moment.T_KICK,
        attacking_code=attacking_code, team_codes=team_codes,
        gk_track_ids=gk_ids, gap_frames=gaps,
    )
    print(f"positions : {len(positions)} players at t_kick "
          f"(attacking_code={attacking_code}, taker={taker_id})")

    # --- feature row (interface I11) ---
    row = build_feature_row(
        positions, fit,
        clip_id=clip_id, corner_side=side.corner_side,
        t_kick_frame=km.t_kick_frame, t_contact_frame=t_c,
        taker_player_id=taker_id,
    )
    features = {k: v for k, v in row.as_row().items()
                if k not in ("clip_id", "corner_side", "t_kick_frame",
                             "t_contact_frame", "zone_geometry_version")}

    summary.update({
        "zone_geometry_version": row.zone_geometry_version,
        "provisional_zones": ["NEAR", "EDGE", "SHORT_PASS"],
        "t_kick": {"frame": km.t_kick_frame, "time_s": round(km.t_kick_frame / fps, 3),
                   "source": km.t_kick_source.value},
        "t_contact": (None if t_c is None else
                      {"frame": t_c, "time_s": round(t_c / fps, 3),
                       "source": km.t_contact_source.value}),
        "taker_player_id": taker_id,
        "features": features,
        "delivery_features_status": "max speed only; heights null pending the 3D height estimator",
        "positions": [p.as_row() for p in positions],
    })
    _write(out_dir, summary)

    # --- annotated snapshot PNG ---
    kick_frame_img = read_frame(cap, km.t_kick_frame)
    render_snapshot(kick_frame_img, tracks_at_kick, positions, ballpx, km,
                    calib_at_kick, clip_id, out_dir / "t_kick.png")
    print(f"wrote {out_dir/'kick_moment.json'} and {out_dir/'t_kick.png'}")
    cap.release()


def _write(out_dir: Path, summary: dict) -> None:
    (out_dir / "kick_moment.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
