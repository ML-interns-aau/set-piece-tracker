"""Run the geometry & moments plane on a single corner-kick clip.

This is a DEMO glue script for the plane that is implemented so far (calibration,
corner-side/orientation, ball smoothing, trajectory fit, key moments). It is not
the real pipeline runner -- there is no ingestion/catalog, team/GK, zones,
features or overlay-verification yet.

Pipeline it runs:
    clip -> (calibration) -> corner-side detection
         -> ball detection (YOLO) -> Kalman/optical-flow smoothing (pixel track)
         -> map to metric pitch coords -> normalize orientation
         -> key moments (t_kick / t_contact) -> projectile trajectory fit
    -> prints a summary and writes ball_track.csv + summary.json (+ optional overlay.mp4)

Calibration is always the vendored PnLCalib model (SoccerNet HRNet keypoint/line
detection) -- robust to the corner-kick player wall -- producing a per-frame
CalibrationTrack that tracks camera pan/zoom. If the ~506 MB weights are missing they
are downloaded automatically on first use (scripts/fetch_pnlcalib_weights.sh).

Ball detection needs torch+ultralytics + yolo11m.pt. Use --no-detect to skip it
(corner side + PnLCalib calibration only).

Examples:
    # per-frame PnLCalib calibration + overlay, first 40 frames
    python scripts/demo_geometry.py --clip data/raw/clips/XXXX.mp4 --overlay --max-frames 40
    # calibration only (no ball detection)
    python scripts/demo_geometry.py --clip data/raw/clips/XXXX.mp4 --no-detect --overlay
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root on path

from src.domain.models import Calibration
from src.domain.pitch import (
    GOAL_AREA_DEPTH_M,
    GOAL_AREA_WIDTH_M,
    GOAL_WIDTH_M,
    PENALTY_AREA_DEPTH_M,
    PENALTY_AREA_WIDTH_M,
    PITCH_WIDTH_M,
    _CENTRE_Y,
)
from src.geometry.calibration import apply_homography
from src.geometry.ball_smoother import BallSmoother
from src.geometry.key_moments import detect_key_moments, min_player_distance, smooth_signal
from src.geometry.orientation import detect_corner_side, to_canonical
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


def detect_ball_track(cap, device, conf, max_frames=0, ball_model=None):
    """Detect + smooth the ball across the clip.

    Uses the tiered BallFinder (full-frame -> ROI re-check -> tiled sweep, with a
    single-ball temporal gate) instead of a plain full-frame YOLO pass -- see
    ``src/engine/ball_finder.py``. With ``ball_model`` (a dedicated soccer-ball
    fine-tune, fetch via ``scripts/fetch_ball_weights.sh``), the ball tiers use
    it while the stock model keeps detecting players.

    Returns ``(track, player_feet_px, tier_counts)``: the smoothed pixel ball
    track ``(frame_idx, u, v, predicted)``, per-frame player foot points (pixel
    bottom-centre of each person bbox, for the key-moment taker/contact
    cross-checks), and the per-tier detection counters for diagnostics.
    """
    from src.engine.ball_finder import BallFinder  # lazy: needs torch/ultralytics

    finder = BallFinder(
        model_path="yolo11m.pt", ball_model_path=ball_model,
        device=device, ball_conf=conf,
    )
    smoother = BallSmoother()
    track: list[tuple[int, float, float, bool]] = []
    player_feet_px: dict[int, list[tuple[float, float]]] = {}
    started = False

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    idx = 0
    while True:
        if max_frames and idx >= max_frames:
            break
        ok, frame = cap.read()
        if not ok:
            break
        bbox, players, _tier = finder.process_frame(frame)
        cx, cy, predicted = smoother.update(frame, bbox)
        started = started or (bbox is not None)
        if started:
            track.append((idx, cx, cy, predicted))
            player_feet_px[idx] = [((x1 + x2) / 2.0, y2) for x1, _, x2, y2 in players]
        idx += 1
    return track, player_feet_px, finder.tier_counts


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the geometry & moments plane on one clip.")
    ap.add_argument("--clip", required=True, help="path to a corner-kick clip")
    ap.add_argument("--frame", type=int, default=0, help="frame index for calibration + corner side")
    ap.add_argument("--no-detect", action="store_true", help="skip ball detection (no torch needed)")
    ap.add_argument("--device", default="cpu", help="YOLO device (cpu / 0 / cuda:0)")
    ap.add_argument("--conf", type=float, default=0.05, help="ball detection confidence floor")
    ap.add_argument("--ball-model", default=None,
                    help="dedicated ball-model weights (default: weights/football-ball-detection.pt "
                         "if present -- fetch with scripts/fetch_ball_weights.sh)")
    ap.add_argument("--max-frames", type=int, default=0, help="cap frames processed (0 = whole clip)")
    ap.add_argument("--overlay", action="store_true", help="also write an annotated overlay.mp4")
    ap.add_argument("--out", default="outputs/demo", help="output directory")
    args = ap.parse_args()

    clip_path = Path(args.clip)
    if not clip_path.exists():
        raise SystemExit(f"clip not found: {clip_path}")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        raise SystemExit(f"could not open clip: {clip_path}")
    fps, n_frames, (w, h) = clip_metadata(cap)
    print(f"clip      : {clip_path.name}  {w}x{h}  {fps:.2f} fps  {n_frames} frames")

    calib_frame = read_frame(cap, args.frame)

    # --- corner side (Task 1) ---
    side = detect_corner_side(calib_frame)
    print(f"corner    : {side.corner_side.value}  (source={side.side_source.value}, "
          f"confidence={side.confidence}) -- confirm/override if low")

    # --- calibration (Task 2): always PnLCalib, per-frame ---
    # PnLCalib is the only calibration path -- the vendored SoccerNet HRNet model, robust
    # to the corner-kick player wall. It builds a per-frame CalibrationTrack (handles
    # camera pan/zoom). Weights are downloaded automatically on first use if missing.
    from src.geometry.pnl_calibration import build_calibration_track
    upper = args.max_frames if args.max_frames else n_frames
    calib_track = build_calibration_track(clip_path, frame_indices=range(upper))
    calib = calib_track.at(args.frame) if calib_track is not None else None
    if calib_track is None:
        print("pnl-calib  : model could not calibrate any frame")
    else:
        extra = (f"  discontinuities={calib_track.discontinuity_frames}"
                 if calib_track.discontinuity_frames else "")
        print(f"pnl-calib  : per-frame track  static={calib_track.static}  "
              f"frames={len(calib_track.per_frame)}  "
              f"mean_reproj={calib_track.mean_reprojection_error_m:.3f} m{extra}")

    if calib is None and calib_track is None:
        print("calibration: none -- metric mapping, key moments & trajectory skipped")
    elif calib is not None:
        print(f"calibration: reprojection_error={calib.reprojection_error_m:.3f} m  "
              f"points_used={calib.points_used}  source={calib.source.value}")

    summary: dict = {
        "clip": clip_path.name,
        "fps": fps,
        "resolution": [w, h],
        "corner_side": side.corner_side.value,
        "corner_side_confidence": side.confidence,
        "calibration": None if calib is None else {
            "reprojection_error_m": calib.reprojection_error_m,
            "points_used": calib.points_used,
            "source": calib.source.value,
        },
    }

    if args.no_detect:
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\nwrote {out_dir/'summary.json'} (no-detect mode)")
        if args.overlay:
            write_overlay(clip_path, out_dir / "overlay.mp4", [], None, fps, (w, h),
                          args.max_frames, calibration=calib, calib_track=calib_track)
            print(f"wrote {out_dir/'overlay.mp4'}")
        cap.release()
        return

    # --- ball detection + smoothing (perception; needs torch) ---
    ball_model = args.ball_model
    if ball_model is None:
        default_weights = Path(__file__).resolve().parents[1] / "weights/football-ball-detection.pt"
        ball_model = str(default_weights) if default_weights.exists() else None
    if ball_model is None:
        print("ball-model : stock yolo11m (COCO sports-ball) -- fetch a soccer fine-tune "
              "with scripts/fetch_ball_weights.sh")
    else:
        print(f"ball-model : {ball_model}")
    try:
        px_track, player_feet_px, tiers = detect_ball_track(
            cap, args.device, args.conf, args.max_frames, ball_model=ball_model
        )
    except ImportError as e:
        cap.release()
        raise SystemExit(
            f"ball detection needs torch+ultralytics ({e}). "
            "Run with the sibling venv's python, or use --no-detect."
        )
    if len(px_track) < 2:
        cap.release()
        raise SystemExit("too few ball detections -- try a lower --conf or a different clip.")
    n_pred = sum(p[3] for p in px_track)
    print(f"ball      : {len(px_track)} frames with a ball position ({n_pred} predicted/extrapolated)")
    print(f"ball tiers: full={tiers['full']}  roi={tiers['roi']}  tiled={tiers['tiled']}  "
          f"miss={tiers['miss']}")
    summary["ball_detection"] = {"model": ball_model or "yolo11m-coco", "tiers": tiers}

    km = None
    metric_track = None
    if calib is not None or calib_track is not None:
        # --- map to metric pitch coords + normalize orientation ---
        # With a per-frame track, each ball sample is mapped through the homography
        # for *its* frame, so a panning/zooming camera no longer corrupts the metric
        # ball path (and hence the trajectory fit + key moments).
        frame_ids = [p[0] for p in px_track]
        px = np.array([[p[1], p[2]] for p in px_track], dtype=np.float64)
        if calib_track is not None:
            mapped = np.vstack([
                apply_homography((calib_track.at(fid) or calib).H, px[i:i + 1])[0]
                for i, fid in enumerate(frame_ids)
            ])
        else:
            mapped = apply_homography(calib.H, px)
        metric = to_canonical(mapped, side.corner_side)
        metric_track = [(frame_ids[i], float(metric[i, 0]), float(metric[i, 1]))
                        for i in range(len(frame_ids))]

        # --- nearest-player distance per frame (taker-foot / contact-gating cross-check) ---
        # Same-frame nearest-player distance only -- not the full I5 gap-bridging
        # pipeline, just enough to drive the key-moment gating on real footage.
        player_feet_metric: list[list[tuple[float, float]]] = []
        for fid in frame_ids:
            feet_px = np.array(player_feet_px.get(fid, []), dtype=np.float64).reshape(-1, 2)
            if len(feet_px) == 0:
                player_feet_metric.append([])
                continue
            H = (calib_track.at(fid) or calib).H if calib_track is not None else calib.H
            mapped_feet = to_canonical(apply_homography(H, feet_px), side.corner_side)
            player_feet_metric.append([tuple(p) for p in mapped_feet])
        player_dist_m = np.array([
            min_player_distance(metric[i], player_feet_metric[i]) for i in range(len(frame_ids))
        ])

        # --- key moments (Task 4) ---
        # Median-prefilter each axis against single-frame glitches; the trajectory
        # fit below still uses the raw metric_track.
        smoothed_xy = np.column_stack([smooth_signal(metric[:, 0]), smooth_signal(metric[:, 1])])
        method = "elastic" if len(smoothed_xy) >= 25 else "threshold"
        print(f"moments   : detector={method} (ELASTIC-style scoring on tracks >= 25 frames)")
        km = detect_key_moments(
            smoothed_xy, fps, frame_offset=frame_ids[0],
            taker_dist_m=player_dist_m, player_dist_m=player_dist_m,
        )
        km_no_gate = detect_key_moments(smoothed_xy, fps, frame_offset=frame_ids[0])
        if km is None:
            print("moments   : no kick detected -- fall back to manual moment tagging")
            if km_no_gate is not None:
                print(f"moments   : (without the taker-foot gate, frame {km_no_gate.t_kick_frame} "
                      "would have matched -- check player detections/H near that frame)")
            summary["key_moments"] = None
        else:
            t_c = km.t_contact_frame
            print(f"moments   : t_kick=frame {km.t_kick_frame} ({km.t_kick_frame/fps:.2f}s)  "
                  f"t_contact={('frame ' + str(t_c) + f' ({t_c/fps:.2f}s)') if t_c is not None else 'None'}")
            summary["key_moments"] = {"t_kick_frame": km.t_kick_frame, "t_contact_frame": t_c}
            if km_no_gate is not None and km_no_gate.t_kick_frame != km.t_kick_frame:
                print(f"moments   : taker-foot gate rejected an earlier candidate at "
                      f"frame {km_no_gate.t_kick_frame}")

        # --- trajectory fit (Task 3) over kick -> contact window if available ---
        if km is not None:
            lo = frame_ids.index(km.t_kick_frame) if km.t_kick_frame in frame_ids else 0
            hi = (frame_ids.index(km.t_contact_frame) + 1
                  if (km.t_contact_frame in frame_ids) else len(metric_track))
            window = metric_track[lo:hi]
        else:
            window = metric_track
        if len(window) >= 2:
            fit = reconstruct_trajectory(window, fps)  # heights_m=None -> horizontal metrics only
            print(f"trajectory: max_speed={fit.max_speed_ms:.1f} m/s  max_height={fit.max_height_m} "
                  "(height None until a height estimator is added)")
            summary["trajectory"] = {
                "max_speed_ms": fit.max_speed_ms,
                "launch_speed_ms": fit.launch_speed_ms,
                "max_height_m": fit.max_height_m,
                "height_at_target_m": fit.height_at_target_m,
                "rmse_m": fit.rmse_m,
                "n_samples": fit.n_samples,
            }

    # --- write outputs ---
    with (out_dir / "ball_track.csv").open("w", newline="", encoding="utf-8") as f:
        wtr = csv.writer(f)
        if metric_track is not None:
            wtr.writerow(["frame_idx", "u_px", "v_px", "predicted", "pitch_x_m", "pitch_y_m"])
            for (fi, u, v, pred), (_, mx, my) in zip(px_track, metric_track):
                wtr.writerow([fi, f"{u:.1f}", f"{v:.1f}", int(pred), f"{mx:.3f}", f"{my:.3f}"])
        else:
            wtr.writerow(["frame_idx", "u_px", "v_px", "predicted"])
            for fi, u, v, pred in px_track:
                wtr.writerow([fi, f"{u:.1f}", f"{v:.1f}", int(pred)])
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nwrote {out_dir/'ball_track.csv'} and {out_dir/'summary.json'}")

    if args.overlay:
        write_overlay(clip_path, out_dir / "overlay.mp4", px_track, km, fps, (w, h),
                      args.max_frames, calibration=calib, calib_track=calib_track)
        print(f"wrote {out_dir/'overlay.mp4'}")
    cap.release()


def _draw_pitch_markings(frame: np.ndarray, H_inv: np.ndarray) -> np.ndarray:
    """Draw FIFA pitch markings projected back onto the frame through H inverse.

    This lets you visually verify whether the calibration lines up with the
    actual pitch in the footage.  Lines are drawn in cyan (goal line, penalty
    area, goal area) with a penalty-spot dot.
    """
    img = frame.copy()

    def _project(metric_xy: tuple[float, float]) -> tuple[int, int]:
        """Map a metric (x, y) point through H inverse to pixel (u, v)."""
        pt = np.array([metric_xy[0], metric_xy[1], 1.0], dtype=np.float64)
        px = H_inv @ pt
        if abs(px[2]) < 1e-12:
            return (-1, -1)
        return (int(px[0] / px[2]), int(px[1] / px[2]))

    # Goal line (x=0, from left edge of penalty area to right edge)
    gl_left = _project((0.0, _CENTRE_Y - PENALTY_AREA_WIDTH_M / 2.0))
    gl_right = _project((0.0, _CENTRE_Y + PENALTY_AREA_WIDTH_M / 2.0))
    cv2.line(img, gl_left, gl_right, (255, 255, 0), 2)  # cyan

    # Penalty area (16.5m line)
    pa_front_left = _project((PENALTY_AREA_DEPTH_M, _CENTRE_Y - PENALTY_AREA_WIDTH_M / 2.0))
    pa_front_right = _project((PENALTY_AREA_DEPTH_M, _CENTRE_Y + PENALTY_AREA_WIDTH_M / 2.0))
    cv2.line(img, pa_front_left, pa_front_right, (255, 255, 0), 2)

    # Penalty area sides (goal line to 16.5m line)
    cv2.line(img, gl_left, pa_front_left, (255, 255, 0), 2)
    cv2.line(img, gl_right, pa_front_right, (255, 255, 0), 2)

    # Goal area (5.5m line)
    ga_front_left = _project((GOAL_AREA_DEPTH_M, _CENTRE_Y - GOAL_AREA_WIDTH_M / 2.0))
    ga_front_right = _project((GOAL_AREA_DEPTH_M, _CENTRE_Y + GOAL_AREA_WIDTH_M / 2.0))
    cv2.line(img, ga_front_left, ga_front_right, (255, 255, 0), 2)

    # Goal area sides
    ga_gl_left = _project((0.0, _CENTRE_Y - GOAL_AREA_WIDTH_M / 2.0))
    ga_gl_right = _project((0.0, _CENTRE_Y + GOAL_AREA_WIDTH_M / 2.0))
    cv2.line(img, ga_gl_left, ga_front_left, (255, 255, 0), 2)
    cv2.line(img, ga_gl_right, ga_front_right, (255, 255, 0), 2)

    # Goal posts
    near_post = _project((0.0, _CENTRE_Y - GOAL_WIDTH_M / 2.0))
    far_post = _project((0.0, _CENTRE_Y + GOAL_WIDTH_M / 2.0))
    cv2.circle(img, near_post, 5, (0, 255, 255), -1)  # yellow dot
    cv2.circle(img, far_post, 5, (0, 255, 255), -1)

    # Penalty spot
    pen_spot = _project((11.0, _CENTRE_Y))
    cv2.circle(img, pen_spot, 5, (0, 255, 255), -1)

    # Labels
    cv2.putText(img, "calibration overlay", (10, img.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    return img


def write_overlay(clip_path, out_path, px_track, km, fps, size, max_frames=0,
                  calibration: Calibration | None = None, calib_track=None) -> None:
    """Annotated overlay: ball trail + t_kick/t_contact markers + optional pitch markings.

    With ``calib_track`` (a per-frame CalibrationTrack) the pitch markings are drawn
    with the homography for *each* frame, so they track a panning/zooming camera.
    """
    by_frame = {fi: (u, v, pred) for fi, u, v, pred in px_track}
    trail: list[tuple[int, int]] = []
    cap = cv2.VideoCapture(str(clip_path))
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    idx = 0
    while True:
        if max_frames and idx >= max_frames:
            break
        ok, frame = cap.read()
        if not ok:
            break
        frame_calib = calib_track.at(idx) if calib_track is not None else calibration
        if frame_calib is not None:
            frame = _draw_pitch_markings(frame, np.linalg.inv(frame_calib.H))
        if idx in by_frame:
            u, v, pred = by_frame[idx]
            trail.append((int(u), int(v)))
            colour = (0, 165, 255) if pred else (0, 255, 0)  # orange=predicted, green=detected
            cv2.circle(frame, (int(u), int(v)), 6, colour, -1)
        for p in trail[-25:]:
            cv2.circle(frame, p, 2, (0, 255, 255), -1)
        if km is not None and idx == km.t_kick_frame:
            cv2.putText(frame, "t_kick", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        if km is not None and km.t_contact_frame is not None and idx == km.t_contact_frame:
            cv2.putText(frame, "t_contact", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
        writer.write(frame)
        idx += 1
    cap.release()
    writer.release()


if __name__ == "__main__":
    main()
