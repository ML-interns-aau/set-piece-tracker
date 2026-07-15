"""Fast, approximate t_kick/t_contact snapshot pairs for a handful of corner clips.

Speed-first version: skips PnLCalib per-frame calibration entirely (not needed to
just *locate* two frames) and runs YOLO ball detection once per clip in pixel
space. Reuses existing project pieces directly:
  - src.engine.detector.FootballDetector (YOLO ball detection)
  - src.geometry.ball_smoother.BallSmoother (pixel smoothing)
  - src.geometry.key_moments.compute_speed / detect_t_kick / detect_t_contact
    (same onset/jerk detectors as the real pipeline, fed pixel speed instead of
    metric speed -- thresholds tuned for px/s instead of m/s)
  - outputs/verification/sample_events.json for sample.mp4 (already computed,
    reused as-is, no recomputation)

Stops as soon as enough corners (default 5) have a usable pair. Writes labeled
PNGs per clip under outputs/snapshots/ and one contact sheet.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.engine.detector import FootballDetector
from src.geometry.ball_smoother import BallSmoother
from src.geometry.key_moments import compute_speed, detect_t_kick, detect_t_contact

REPO_ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_DIR = REPO_ROOT / "outputs" / "snapshots"
TARGET_CORNERS = 5


def pixel_ball_track(clip_path: Path, detector: FootballDetector) -> tuple[np.ndarray, float]:
    cap = cv2.VideoCapture(str(clip_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    smoother = BallSmoother()
    pts = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        balls = detector.detect_ball(frame)
        bbox = None
        if len(balls) > 0:
            best = int(np.argmax(balls.confidence))
            bbox = balls.xyxy[best]
        cx, cy, _ = smoother.update(frame, bbox)
        pts.append((cx, cy))
    cap.release()
    return np.array(pts, dtype=np.float64), fps


def find_moments_pixel_space(track: np.ndarray, fps: float) -> tuple[int, int] | None:
    """Approximate t_kick/t_contact off pixel-space ball speed (px/s)."""
    speed = compute_speed(track, fps)
    kick = detect_t_kick(speed, speed_thresh_ms=120.0, baseline_max_ms=60.0,
                          baseline_frames=5, sustain_frames=2)
    if kick is None:
        return None
    contact = detect_t_contact(speed, kick, jerk_thresh_ms=120.0, min_frames_after=2)
    if contact is None:
        contact = min(kick + 10, len(speed) - 1)
    return kick, contact


def read_frame(clip_path: Path, frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(clip_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"could not read frame {frame_idx} from {clip_path}")
    return frame


def label_frame(frame: np.ndarray, clip_label: str, moment: str, frame_idx: int, fps: float) -> np.ndarray:
    bar_h = 44
    bar = np.zeros((bar_h, frame.shape[1], 3), dtype=np.uint8)
    text = f"{clip_label}  |  {moment}  |  frame {frame_idx} ({frame_idx / fps:.2f}s)"
    colour = (0, 255, 255) if moment == "t_kick" else (0, 165, 255)
    cv2.putText(bar, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2, cv2.LINE_AA)
    return np.vstack([bar, frame])


def save_pair(label: str, clip_path: Path, t_kick: int, t_contact: int, fps: float, index: int) -> tuple[Path, Path]:
    slug = label.replace(" ", "_")
    out_dir = SNAPSHOT_DIR / f"{index:02d}_{slug}"
    out_dir.mkdir(parents=True, exist_ok=True)
    kick_path = out_dir / f"{slug}_t_kick.png"
    contact_path = out_dir / f"{slug}_t_contact.png"
    cv2.imwrite(str(kick_path), label_frame(read_frame(clip_path, t_kick), label, "t_kick", t_kick, fps))
    cv2.imwrite(str(contact_path), label_frame(read_frame(clip_path, t_contact), label, "t_contact", t_contact, fps))
    return kick_path, contact_path


def make_thumbnail(img: np.ndarray, height: int = 300) -> np.ndarray:
    h, w = img.shape[:2]
    scale = height / h
    return cv2.resize(img, (int(w * scale), height))


def build_contact_sheet(pairs: list[tuple[Path, Path]], out_path: Path, thumb_height: int = 300) -> None:
    rows = []
    for kick_path, contact_path in pairs:
        kick_img = make_thumbnail(cv2.imread(str(kick_path)), thumb_height)
        contact_img = make_thumbnail(cv2.imread(str(contact_path)), thumb_height)
        gap = np.full((thumb_height, 6, 3), 255, dtype=np.uint8)
        rows.append(np.hstack([kick_img, gap, contact_img]))
    max_w = max(r.shape[1] for r in rows)
    row_gap = np.full((6, max_w, 3), 255, dtype=np.uint8)
    padded = []
    for i, r in enumerate(rows):
        if r.shape[1] < max_w:
            r = np.hstack([r, np.full((r.shape[0], max_w - r.shape[1], 3), 255, dtype=np.uint8)])
        padded.append(r)
        if i != len(rows) - 1:
            padded.append(row_gap)
    cv2.imwrite(str(out_path), np.vstack(padded))


def main() -> None:
    pairs: list[tuple[Path, Path]] = []
    results: list[str] = []

    events_path = REPO_ROOT / "outputs/verification/sample_events.json"
    data = json.loads(events_path.read_text(encoding="utf-8"))
    by_type = {e["event_type"]: e for e in data["events"]}
    clip_path = REPO_ROOT / "data/samples/sample.mp4"
    kick_f, contact_f, fps = by_type["kick"]["frame"], by_type["contact"]["frame"], data["fps"]
    k, c = save_pair("sample", clip_path, kick_f, contact_f, fps, 1)
    pairs.append((k, c))
    results.append(f"[1] sample: t_kick=frame {kick_f}  t_contact=frame {contact_f} (from existing sample_events.json)")

    candidates = ["clip 2", "clip 3", "clip 4", "clip 5", "clip 6", "clip 7", "clip one"]
    detector = FootballDetector(model_path="yolo11m.pt", conf=0.25, device="cpu")
    idx = 2
    for name in candidates:
        if len(pairs) >= TARGET_CORNERS:
            break
        clip_path = REPO_ROOT / "data/samples" / f"{name}.mp4"
        track, fps = pixel_ball_track(clip_path, detector)
        moments = find_moments_pixel_space(track, fps)
        if moments is None:
            results.append(f"skip {name}: no clear kick onset in pixel-speed signal")
            continue
        kick_f, contact_f = moments
        k, c = save_pair(name, clip_path, kick_f, contact_f, fps, idx)
        pairs.append((k, c))
        results.append(f"[{idx}] {name}: t_kick=frame {kick_f}  t_contact=frame {contact_f}")
        idx += 1

    for line in results:
        print(line)

    if len(pairs) < 4:
        print(f"\nWARNING: only {len(pairs)} usable corners found (wanted 4-5)")

    sheet_path = SNAPSHOT_DIR / "contact_sheet.png"
    build_contact_sheet(pairs, sheet_path)
    print(f"\nwrote {len(pairs)} corner(s), contact sheet: {sheet_path}")


if __name__ == "__main__":
    main()
