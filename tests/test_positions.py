"""Tests for the pure I10 positions producer (src/geometry/positions.py).

Uses an identity homography so a foot-point pixel (u, v) maps straight to metric
(u, v): this lets us place players at chosen pitch coords by choosing their
bboxes, then assert team/GK/provenance and that the rows flow through
``build_feature_row`` to the expected Appendix-A counts. Pure -- no clip/weights.
"""

from __future__ import annotations

import math

import numpy as np

from src.domain.models import (
    Calibration,
    CornerSide,
    Moment,
    PositionSource,
    Team,
)
from src.features.feature_rows import build_feature_row
from src.geometry.positions import (
    GK_CODE,
    REFEREE_CODE,
    build_player_positions,
    foot_point,
    resolve_attacking_side,
)

IDENTITY = Calibration(H=np.eye(3), reprojection_error_m=0.1, points_used=8)


def _bbox_for_foot(fx: float, fy: float, w: float = 2.0, h: float = 2.0) -> tuple:
    """A bbox whose foot-point (bottom-center) is exactly (fx, fy)."""
    return (fx - w / 2.0, fy - h, fx + w / 2.0, fy)


# track_id: (foot_x, foot_y, team_code)  -- foot coords == metric coords under identity H
_PLAYERS = {
    7: (3.0, 4.0, 0),          # taker (attacking), nearest the ball; short-pass zone
    12: (4.0, 3.0, 0),         # attacking, short-pass option (not taker)
    8: (10.0, 34.0, 0),        # attacking, PENALTY only
    9: (2.0, 28.0, 1),         # defending outfield: near-post band + GK area + NEAR + PEN
    10: (1.0, 34.0, GK_CODE),  # defending GK: GK area + PEN (excluded from excl-GK counts)
    11: (5.0, 5.0, REFEREE_CODE),  # referee -- must be dropped
}
BALL_PX = (3.0, 4.0)


def _tracks():
    return [
        (tid, _bbox_for_foot(fx, fy), 0.9)
        for tid, (fx, fy, _code) in _PLAYERS.items()
    ]


def _team_codes():
    return {tid: code for tid, (_fx, _fy, code) in _PLAYERS.items()}


def test_foot_point_is_bottom_center():
    assert foot_point((10.0, 20.0, 30.0, 60.0)) == (20.0, 60.0)


def test_resolve_attacking_side_picks_taker_cluster():
    code, taker = resolve_attacking_side(_tracks(), BALL_PX, _team_codes())
    assert code == 0          # taker (id 7) is in cluster 0
    assert taker == 7


def test_resolve_attacking_side_fallback_without_ball():
    code, taker = resolve_attacking_side(_tracks(), None, _team_codes())
    assert code == 0          # lowest valid outfield code
    assert taker is None


def test_build_positions_team_gk_and_provenance():
    attacking_code, _taker = resolve_attacking_side(_tracks(), BALL_PX, _team_codes())
    rows = build_player_positions(
        _tracks(), IDENTITY, CornerSide.LEFT,
        clip_id="c1", moment=Moment.T_KICK,
        attacking_code=attacking_code, team_codes=_team_codes(),
    )
    by_id = {r.player_id: r for r in rows}

    assert 11 not in by_id                      # referee dropped
    assert len(rows) == 5

    assert by_id[7].team is Team.ATTACKING
    assert by_id[9].team is Team.DEFENDING and not by_id[9].is_goalkeeper
    assert by_id[10].team is Team.DEFENDING and by_id[10].is_goalkeeper

    # identity homography, LEFT corner -> metric == foot-point pixel
    assert math.isclose(by_id[8].pitch_x, 10.0) and math.isclose(by_id[8].pitch_y, 34.0)

    # all detected this frame -> DETECTED provenance, positive reliability
    assert all(r.position_source is PositionSource.DETECTED for r in rows)
    assert all(0.0 < r.reliability_score <= 1.0 for r in rows)


def test_missing_calibration_yields_nan_and_zero_reliability():
    rows = build_player_positions(
        _tracks(), None, CornerSide.LEFT,
        clip_id="c1", moment=Moment.T_KICK,
        attacking_code=0, team_codes=_team_codes(),
    )
    r = rows[0]
    assert math.isnan(r.pitch_x) and math.isnan(r.pitch_y)
    assert r.reliability_score == 0.0          # missing calibration is the hard gate


def test_gap_marks_extrapolated():
    rows = build_player_positions(
        _tracks(), IDENTITY, CornerSide.LEFT,
        clip_id="c1", moment=Moment.T_KICK,
        attacking_code=0, team_codes=_team_codes(),
        gap_frames={9: 4.0},
    )
    by_id = {r.player_id: r for r in rows}
    assert by_id[9].position_source is PositionSource.EXTRAPOLATED
    assert by_id[7].position_source is PositionSource.DETECTED
    # extrapolated position scores lower than an equivalent detected one
    assert by_id[9].reliability_score < by_id[7].reliability_score


def test_positions_flow_into_feature_row():
    attacking_code, taker = resolve_attacking_side(_tracks(), BALL_PX, _team_codes())
    rows = build_player_positions(
        _tracks(), IDENTITY, CornerSide.LEFT,
        clip_id="c1", moment=Moment.T_KICK,
        attacking_code=attacking_code, team_codes=_team_codes(),
    )
    row = build_feature_row(
        rows, None,
        clip_id="c1", corner_side=CornerSide.LEFT,
        t_kick_frame=30, t_contact_frame=None,
        taker_player_id=taker,
    )

    assert row.num_short_pass_options == 1        # id 12; taker (id 7) excluded
    assert row.num_def_in_near_area == 1          # id 9 (GK id 10 at y=34 is outside NEAR)
    assert row.num_att_players_in_gk_area == 0
    assert row.num_def_players_in_gk_area == 1    # id 9 outfield; GK id 10 excluded
    assert row.num_att_players_in_pen_area == 1   # id 8
    assert row.num_def_players_in_pen_area == 2   # id 9 + GK id 10 (incl. GK)
    assert row.num_def_near_post == 1             # id 9; GK excluded
    assert row.num_def_far_post == 0
    assert row.num_att_player_in_edge_area == 0
    assert row.num_def_player_in_edge_area == 0
    # no trajectory fit supplied -> delivery metrics stay None
    assert row.pass_speed_max_in_ms is None
