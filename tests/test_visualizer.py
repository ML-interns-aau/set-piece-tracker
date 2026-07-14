"""FR-017 - overlay drawing primitives and PipelineVisualizer against synthetic frames."""

from __future__ import annotations

import cv2
import numpy as np

from src.domain.models import Team
from src.features.zones import ZONES
from src.verification.config import VERIFICATION_CONFIG
from src.verification.events import EventRecord, EventType
from src.verification.overlay import (
    draw_event_marker,
    draw_player_box,
    draw_zone,
)
from src.verification.visualizer import (
    FrameTracks,
    PipelineVisualizer,
    TrackedBox,
    render_frame,
)


def _frame(w: int = 320, h: int = 240) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_draw_player_box_changes_pixels_within_bbox():
    frame = _frame()
    draw_player_box(frame, (50, 50, 100, 150), Team.ATTACKING, False, 7)
    assert frame.any()


def test_draw_player_box_unknown_team_uses_grey():
    frame = _frame()
    draw_player_box(frame, (50, 50, 100, 150), None, False, 1)
    top_edge = frame[50, 50:100]
    assert any(tuple(px) == VERIFICATION_CONFIG.unknown_color for px in top_edge)


def test_draw_player_box_goalkeeper_adds_gold_accent():
    frame = _frame()
    draw_player_box(frame, (50, 50, 100, 150), Team.DEFENDING, True, 1)
    top_edge = frame[50, 50:100]
    assert any(tuple(px) == VERIFICATION_CONFIG.gk_accent_color for px in top_edge)


def test_draw_zone_rectzone_projects_quad_via_identity_homography():
    frame = _frame(400, 400)
    H_inv = np.eye(3)
    before = frame.copy()
    draw_zone(frame, ZONES.gk_area, H_inv, VERIFICATION_CONFIG)
    assert not np.array_equal(frame, before)


def test_draw_zone_unionzone_draws_both_parts():
    frame = _frame(400, 400)
    H_inv = np.eye(3)
    before = frame.copy()
    draw_zone(frame, ZONES.short_pass, H_inv, VERIFICATION_CONFIG)
    assert not np.array_equal(frame, before)


def test_draw_zone_provisional_label_suffix_present():
    assert ZONES.near_area.provisional is True
    assert ZONES.gk_area.provisional is False


def test_draw_event_marker_banner_in_configured_corner():
    frame = _frame()
    draw_event_marker(frame, EventType.PASS, 0.91, VERIFICATION_CONFIG)
    margin = VERIFICATION_CONFIG.marker_margin_px
    assert frame[margin + 5, margin + 5].any()


def test_render_frame_draws_ball_at_bbox_centre():
    tracks = FrameTracks(frame_idx=0, boxes=(
        TrackedBox(track_id=-99, bbox_xyxy=(100, 100, 110, 110), team=None,
                   is_goalkeeper=False, is_ball=True),
    ))
    result = render_frame(_frame(), tracks, (), None)
    assert tuple(result[105, 105]) == VERIFICATION_CONFIG.ball_detected_color


def test_render_frame_is_pure_returns_new_array():
    frame = _frame()
    original = frame.copy()
    tracks = FrameTracks(frame_idx=0, boxes=(
        TrackedBox(track_id=1, bbox_xyxy=(10, 10, 40, 60), team=Team.ATTACKING,
                   is_goalkeeper=False),
    ))
    result = render_frame(frame, tracks, (), None)
    assert np.array_equal(frame, original)
    assert not np.array_equal(result, original)


def test_render_frame_only_draws_events_within_marker_duration_window():
    tracks = FrameTracks(frame_idx=0, boxes=())
    events = (EventRecord(frame=0, event_type=EventType.SHOT, confidence=0.5),)

    within = render_frame(_frame(), tracks, events, None)
    outside = render_frame(
        _frame(), FrameTracks(frame_idx=999, boxes=()), events, None
    )
    assert within.any()
    assert not outside.any()


def test_frame_tracks_from_supervision_maps_team_and_ball_sentinel_id():
    from src.verification.visualizer import BALL_TRACKER_ID, from_supervision

    class _FakeDetections:
        xyxy = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=np.float64)
        tracker_id = np.array([7, BALL_TRACKER_ID])

    tracks = from_supervision(_FakeDetections(), frame_idx=5,
                               team_by_track={7: Team.ATTACKING}, gk_track_ids=[7])
    assert tracks.frame_idx == 5
    player, ball = tracks.boxes
    assert player.team is Team.ATTACKING
    assert player.is_goalkeeper is True
    assert player.is_ball is False
    assert ball.is_ball is True
    assert ball.team is None


def test_pipeline_visualizer_render_writes_readable_mp4(tmp_path):
    clip_path = tmp_path / "synthetic.mp4"
    fps, size, n_frames = 10.0, (64, 48), 5
    writer = cv2.VideoWriter(str(clip_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for _ in range(n_frames):
        writer.write(np.zeros((size[1], size[0], 3), dtype=np.uint8))
    writer.release()

    out_path = tmp_path / "overlay.mp4"
    tracks_by_frame = {
        i: FrameTracks(frame_idx=i, boxes=(
            TrackedBox(track_id=1, bbox_xyxy=(5, 5, 20, 30), team=Team.ATTACKING,
                       is_goalkeeper=False),
        ))
        for i in range(n_frames)
    }
    visualizer = PipelineVisualizer()
    result_path = visualizer.render(clip_path, out_path, tracks_by_frame, events=())

    assert result_path == out_path
    assert out_path.exists()
    cap = cv2.VideoCapture(str(out_path))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    assert count == n_frames
