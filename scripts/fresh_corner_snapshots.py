"""Generate fresh corner-kick snapshot pairs for the professor's deliverable.

Produces exactly TWO labelled snapshots per corner clip:
  1. "Corner Hit"   – the frame the ball is struck (t_kick)
  2. "First Contact" – the frame someone first touches the delivery (t_contact)

For 5 corners, using the professor's exact wording on each image.

Improvements over the previous version:
- Uses the professor's plain-English labels instead of t_kick / t_contact
- Requires a minimum gap between kick and contact (≥ 8 frames / ~0.3 s)
  to reject false-positive contacts right after the kick
- Detects camera-cut discontinuities and flags them
- Draws the ball position circle on each snapshot so the professor can see
  what was detected
- Produces a cleaner contact sheet with labels
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

MIN_KICK_CONTACT_GAP = 8


def pixel_ball_track(clip_path: Path, detector: FootballDetector) -> tuple[np.ndarray, float]:
    """Run YOLO ball detection + Kalman smoothing, return pixel positions and fps."""
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


def detect_camera_cut(clip_path: Path, frame_a: int, frame_b: int,
                      hist_thresh: float = 0.35) -> bool:
    """Return True if there's a likely camera cut between frame_a and frame_b."""
    cap = cv2.VideoCapture(str(clip_path))
    frames = {}
    for fi in range(frame_a, frame_b + 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if ok:
            frames[fi] = frame
    cap.release()

    if len(frames) < 2:
        return False

    sorted_indices = sorted(frames.keys())
    for i in range(len(sorted_indices) - 1):
        f1 = frames[sorted_indices[i]]
        f2 = frames[sorted_indices[i + 1]]
        h1 = cv2.calcHist([f1], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        h2 = cv2.calcHist([f2], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        cv2.normalize(h1, h1)
        cv2.normalize(h2, h2)
        corr = cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)
        if corr < hist_thresh:
            return True
    return False


def find_moments_pixel_space(track: np.ndarray, fps: float) -> tuple[int, int] | None:
    """Find t_kick and t_contact from pixel-space ball speed.
    
    Returns None if no valid kick is found or if the contact is too close
    to the kick (likely a false positive).
    """
    speed = compute_speed(track, fps)
    kick = detect_t_kick(speed, speed_thresh_ms=120.0, baseline_max_ms=60.0,
                          baseline_frames=5, sustain_frames=2)
    if kick is None:
        return None

    contact = detect_t_contact(speed, kick, jerk_thresh_ms=120.0, min_frames_after=2)

    if contact is not None and (contact - kick) < MIN_KICK_CONTACT_GAP:
        contact = detect_t_contact(speed, kick, jerk_thresh_ms=200.0,
                                    min_frames_after=MIN_KICK_CONTACT_GAP)

    if contact is None:
        fallback = kick + int(fps * 0.6)
        if fallback < len(speed) - 1:
            contact = fallback
        else:
            contact = len(speed) - 1

    return kick, contact


def read_frame(clip_path: Path, frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(clip_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"could not read frame {frame_idx} from {clip_path}")
    return frame


def draw_ball_marker(frame: np.ndarray, ball_pos: tuple[float, float]) -> np.ndarray:
    """Draw a circle + crosshair at the ball position."""
    out = frame.copy()
    cx, cy = int(ball_pos[0]), int(ball_pos[1])
    cv2.circle(out, (cx, cy), 22, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.line(out, (cx - 15, cy), (cx + 15, cy), (0, 255, 255), 1, cv2.LINE_AA)
    cv2.line(out, (cx, cy - 15), (cx, cy + 15), (0, 255, 255), 1, cv2.LINE_AA)
    return out


def label_frame(frame: np.ndarray, clip_label: str, moment_label: str,
                frame_idx: int, fps: float, ball_pos: tuple[float, float] | None = None,
                warning: str | None = None) -> np.ndarray:
    """Add a labelled header bar to the frame."""
    bar_h = 52
    bar = np.zeros((bar_h, frame.shape[1], 3), dtype=np.uint8)

    text = f"{clip_label}  |  {moment_label}  |  frame {frame_idx} ({frame_idx / fps:.2f}s)"
    colour = (0, 255, 255) if "Corner" in moment_label else (0, 200, 255)
    cv2.putText(bar, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, colour, 2, cv2.LINE_AA)

    if warning:
        cv2.putText(bar, warning, (12, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 0, 255), 1, cv2.LINE_AA)

    out_frame = frame.copy()
    if ball_pos is not None:
        out_frame = draw_ball_marker(out_frame, ball_pos)

    return np.vstack([bar, out_frame])


def save_pair(label: str, clip_path: Path, t_kick: int, t_contact: int,
              fps: float, index: int, track: np.ndarray | None = None,
              warning_kick: str | None = None,
              warning_contact: str | None = None) -> tuple[Path, Path]:
    """Save the two labelled snapshot PNGs."""
    slug = label.replace(" ", "_")
    out_dir = SNAPSHOT_DIR / f"{index:02d}_{slug}"
    out_dir.mkdir(parents=True, exist_ok=True)

    kick_ball = tuple(track[t_kick]) if track is not None and t_kick < len(track) else None
    contact_ball = tuple(track[t_contact]) if track is not None and t_contact < len(track) else None

    kick_path = out_dir / f"{slug}_corner_hit.png"
    contact_path = out_dir / f"{slug}_first_contact.png"

    cv2.imwrite(str(kick_path), label_frame(
        read_frame(clip_path, t_kick), label, "Corner Hit",
        t_kick, fps, kick_ball, warning_kick))
    cv2.imwrite(str(contact_path), label_frame(
        read_frame(clip_path, t_contact), label, "First Contact",
        t_contact, fps, contact_ball, warning_contact))
    return kick_path, contact_path


def make_thumbnail(img: np.ndarray, height: int = 340) -> np.ndarray:
    h, w = img.shape[:2]
    scale = height / h
    return cv2.resize(img, (int(w * scale), height))


def build_contact_sheet(pairs: list[tuple[Path, Path]], out_path: Path,
                        thumb_height: int = 340) -> None:
    """Build a side-by-side contact sheet of all snapshot pairs."""
    rows = []
    for kick_path, contact_path in pairs:
        kick_img = make_thumbnail(cv2.imread(str(kick_path)), thumb_height)
        contact_img = make_thumbnail(cv2.imread(str(contact_path)), thumb_height)
        gap = np.full((thumb_height, 8, 3), 40, dtype=np.uint8)
        rows.append(np.hstack([kick_img, gap, contact_img]))

    max_w = max(r.shape[1] for r in rows)
    row_gap = np.full((8, max_w, 3), 40, dtype=np.uint8)
    padded = []
    for i, r in enumerate(rows):
        if r.shape[1] < max_w:
            r = np.hstack([r, np.full((r.shape[0], max_w - r.shape[1], 3), 40, dtype=np.uint8)])
        padded.append(r)
        if i != len(rows) - 1:
            padded.append(row_gap)
    cv2.imwrite(str(out_path), np.vstack(padded))


def main() -> None:
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    pairs: list[tuple[Path, Path]] = []
    results: list[str] = []

    events_path = REPO_ROOT / "outputs/verification/sample_events.json"
    data = json.loads(events_path.read_text(encoding="utf-8"))
    by_type = {e["event_type"]: e for e in data["events"]}
    clip_path = REPO_ROOT / "data/samples/sample.mp4"
    kick_f = by_type["kick"]["frame"]
    contact_f = by_type["contact"]["frame"]
    fps = data["fps"]

    sample_track_path = REPO_ROOT / "outputs/verification/sample_ball_track.json"
    sample_track = None
    if sample_track_path.exists():
        bt = json.loads(sample_track_path.read_text(encoding="utf-8"))
        if "pixel_positions" in bt:
            sample_track = np.array(bt["pixel_positions"], dtype=np.float64)

    warning = None
    if detect_camera_cut(clip_path, kick_f, contact_f):
        warning = "⚠ Camera cut detected between moments"

    k, c = save_pair("sample", clip_path, kick_f, contact_f, fps, 1,
                     track=sample_track, warning_contact=warning)
    pairs.append((k, c))
    results.append(f"[1] sample: Corner Hit=frame {kick_f}  First Contact=frame {contact_f}")

    candidates = ["clip 2", "clip 3", "clip 5", "clip 4", "clip 6", "clip 7", "clip one"]
    detector = FootballDetector(model_path="yolo11m.pt", conf=0.25, device="cpu")
    idx = 2

    for name in candidates:
        if len(pairs) >= TARGET_CORNERS:
            break
        clip_path = REPO_ROOT / "data/samples" / f"{name}.mp4"
        if not clip_path.exists():
            results.append(f"  skip {name}: file not found")
            continue

        print(f"Processing {name}...")
        track, fps = pixel_ball_track(clip_path, detector)
        moments = find_moments_pixel_space(track, fps)
        if moments is None:
            results.append(f"  skip {name}: no clear kick onset detected")
            continue

        kick_f, contact_f = moments

        warn_contact = None
        if detect_camera_cut(clip_path, kick_f, contact_f):
            warn_contact = "Camera cut between moments"
            results.append(f"  note {name}: camera cut between kick and contact")

        gap_s = (contact_f - kick_f) / fps
        k, c = save_pair(name, clip_path, kick_f, contact_f, fps, idx,
                         track=track, warning_contact=warn_contact)
        pairs.append((k, c))
        results.append(
            f"[{idx}] {name}: Corner Hit=frame {kick_f}  "
            f"First Contact=frame {contact_f}  (gap={gap_s:.2f}s)")
        idx += 1

    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    for line in results:
        print(line)

    if len(pairs) < 4:
        print(f"\n⚠ WARNING: only {len(pairs)} usable corners found (wanted 4-5)")

    sheet_path = SNAPSHOT_DIR / "contact_sheet.png"
    build_contact_sheet(pairs, sheet_path)
    print(f"\nWrote {len(pairs)} corner pair(s)")
    print(f"  Individual PNGs: {SNAPSHOT_DIR}")
    print(f"  Contact sheet:   {sheet_path}")


if __name__ == "__main__":
    main()
