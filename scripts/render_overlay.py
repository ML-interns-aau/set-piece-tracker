"""Render a clip's verification overlay video (FR-017).

Usage:
    python scripts/render_overlay.py --clip data/raw/clips/clip_0007.mp4 \
        --events outputs/verification/clip_0007_events.json \
        --out outputs/verification/clip_0007_overlay.mp4
    python scripts/render_overlay.py --clip data/raw/clips/clip_0007.mp4 \
        --events outputs/verification/clip_0007_events.json \
        --out outputs/verification/clip_0007_overlay.mp4 \
        --tracks outputs/verification/clip_0007_tracks.json \
        --calib outputs/verification/clip_0007_calibration.json

Track file format (optional; omit to render zones/event markers with no player
boxes, since no pipeline runner emits this file yet). "is_ball" is optional and
defaults to false; a ball entry is drawn as a ball marker, not a player box:
    {"<frame_idx>": [{"track_id": 7, "bbox": [x1, y1, x2, y2],
                       "team": "attacking", "is_goalkeeper": false},
                      {"track_id": -99, "bbox": [...], "is_ball": true}], ...}

Calibration file format (optional; omit to render without zone overlays):
    {"<frame_idx>": [[h00, h01, h02], [h10, h11, h12], [h20, h21, h22]], ...}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.domain.models import Calibration, CalibrationTrack, Team  # noqa: E402
from src.verification.events import PipelineOutputEvents  # noqa: E402
from src.verification.visualizer import FrameTracks, PipelineVisualizer, TrackedBox  # noqa: E402


def _load_tracks_by_frame(path: Path | None) -> dict[int, FrameTracks]:
    if path is None:
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    tracks_by_frame: dict[int, FrameTracks] = {}
    for frame_str, boxes in raw.items():
        frame_idx = int(frame_str)
        boxes_out = tuple(
            TrackedBox(
                track_id=int(b["track_id"]),
                bbox_xyxy=tuple(float(v) for v in b["bbox"]),
                team=Team(b["team"]) if b.get("team") else None,
                is_goalkeeper=bool(b.get("is_goalkeeper", False)),
                is_ball=bool(b.get("is_ball", False)),
            )
            for b in boxes
        )
        tracks_by_frame[frame_idx] = FrameTracks(frame_idx=frame_idx, boxes=boxes_out)
    return tracks_by_frame


def _load_calib_track(path: Path | None) -> CalibrationTrack | None:
    if path is None:
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    per_frame = {
        int(frame_str): Calibration(
            H=np.array(h, dtype=np.float64), reprojection_error_m=0.0, points_used=4
        )
        for frame_str, h in raw.items()
    }
    return CalibrationTrack(per_frame=per_frame)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Render a clip's verification overlay video (FR-017)."
    )
    ap.add_argument("--clip", required=True, help="path to the clip to overlay")
    ap.add_argument("--events", required=True, help="path to a PipelineOutputEvents JSON")
    ap.add_argument("--out", required=True, help="output overlay mp4 path")
    ap.add_argument("--tracks", default=None,
                     help="optional per-frame tracks JSON (see module docstring)")
    ap.add_argument("--calib", default=None,
                     help="optional per-frame calibration JSON (see module docstring)")
    ap.add_argument("--max-frames", type=int, default=0,
                     help="cap the number of frames rendered (0 = all)")
    args = ap.parse_args()

    clip_path = Path(args.clip)
    if not clip_path.exists():
        raise SystemExit(f"clip not found: {clip_path}")

    events_output = PipelineOutputEvents.from_json(args.events)
    tracks_by_frame = _load_tracks_by_frame(Path(args.tracks) if args.tracks else None)
    calib_track = _load_calib_track(Path(args.calib) if args.calib else None)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    visualizer = PipelineVisualizer()
    result = visualizer.render(
        clip_path=clip_path, out_path=out_path, tracks_by_frame=tracks_by_frame,
        events=events_output.events, calib_track=calib_track, max_frames=args.max_frames,
    )
    print(f"wrote {result}")


if __name__ == "__main__":
    main()
