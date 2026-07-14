from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.domain.models import CalibrationTrack, Team
from src.features.zones import ZONES, ZoneModel
from src.verification.config import VERIFICATION_CONFIG, VerificationConfig
from src.verification.events import EventRecord
from src.verification.overlay import draw_ball, draw_event_marker, draw_player_box, draw_zone

BALL_TRACKER_ID = -99


@dataclass(frozen=True)
class TrackedBox:
    track_id: int
    bbox_xyxy: tuple[float, float, float, float]
    team: Team | None
    is_goalkeeper: bool
    is_ball: bool = False


@dataclass(frozen=True)
class FrameTracks:
    frame_idx: int
    boxes: tuple[TrackedBox, ...]


def from_supervision(
    detections: object,
    frame_idx: int,
    team_by_track: Mapping[int, Team],
    gk_track_ids: Sequence[int] = (),
) -> FrameTracks:
    gk_ids = set(gk_track_ids)
    boxes: list[TrackedBox] = []
    xyxy = getattr(detections, "xyxy")
    tracker_ids = getattr(detections, "tracker_id")
    for bbox, track_id in zip(xyxy, tracker_ids):
        track_id = int(track_id)
        is_ball = track_id == BALL_TRACKER_ID
        boxes.append(TrackedBox(
            track_id=track_id,
            bbox_xyxy=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
            team=None if is_ball else team_by_track.get(track_id),
            is_goalkeeper=track_id in gk_ids,
            is_ball=is_ball,
        ))
    return FrameTracks(frame_idx=frame_idx, boxes=tuple(boxes))


def _active_events(
    frame_idx: int, events: Sequence[EventRecord], duration_frames: int
) -> list[EventRecord]:
    return [e for e in events if e.frame <= frame_idx < e.frame + duration_frames]


def render_frame(
    frame: np.ndarray,
    tracks: FrameTracks,
    events: Sequence[EventRecord],
    H_inv: np.ndarray | None,
    zones: ZoneModel = ZONES,
    config: VerificationConfig = VERIFICATION_CONFIG,
) -> np.ndarray:
    out = frame.copy()

    if H_inv is not None:
        for zone in (
            zones.gk_area, zones.penalty_area, zones.near_post_band,
            zones.far_post_band, zones.near_area, zones.edge_area, zones.short_pass,
        ):
            draw_zone(out, zone, H_inv, config)

    for box in tracks.boxes:
        if box.is_ball:
            x1, y1, x2, y2 = box.bbox_xyxy
            draw_ball(out, ((x1 + x2) / 2.0, (y1 + y2) / 2.0), predicted=False, config=config)
            continue
        draw_player_box(out, box.bbox_xyxy, box.team, box.is_goalkeeper, box.track_id, config)

    for event in _active_events(tracks.frame_idx, events, config.marker_duration_frames):
        draw_event_marker(out, event.event_type, event.confidence, config)

    return out


class PipelineVisualizer:

    def __init__(self, config: VerificationConfig = VERIFICATION_CONFIG) -> None:
        self.config = config

    def render(
        self,
        clip_path: str | Path,
        out_path: str | Path,
        tracks_by_frame: Mapping[int, FrameTracks],
        events: Sequence[EventRecord],
        calib_track: CalibrationTrack | None = None,
        zones: ZoneModel = ZONES,
        max_frames: int = 0,
    ) -> Path:
        cap = cv2.VideoCapture(str(clip_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_path = Path(out_path)
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                  fps, (width, height))

        idx = 0
        try:
            while True:
                if max_frames and idx >= max_frames:
                    break
                ok, frame = cap.read()
                if not ok:
                    break
                tracks = tracks_by_frame.get(idx, FrameTracks(frame_idx=idx, boxes=()))
                calibration = calib_track.at(idx) if calib_track is not None else None
                h_inv = np.linalg.inv(calibration.H) if calibration is not None else None
                rendered = render_frame(frame, tracks, events, h_inv, zones, self.config)
                writer.write(rendered)
                idx += 1
        finally:
            cap.release()
            writer.release()

        return out_path
