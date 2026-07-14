"""FR-014/015 - 13-feature computation off the position-record schema."""

from __future__ import annotations

from src.domain.models import (
    FEATURE_COLUMNS,
    CornerSide,
    Moment,
    PlayerPosition,
    PositionSource,
    ProjectileFit,
    Team,
)
from src.features.feature_rows import build_feature_row


def _pos(
    player_id: int,
    team: Team,
    x: float,
    y: float,
    *,
    is_gk: bool = False,
    moment: Moment = Moment.T_KICK,
) -> PlayerPosition:
    return PlayerPosition(
        clip_id="c1",
        moment=moment,
        player_id=player_id,
        team=team,
        is_goalkeeper=is_gk,
        pitch_x=x,
        pitch_y=y,
        position_source=PositionSource.DETECTED,
        reliability_score=0.9,
    )


def _fit() -> ProjectileFit:
    return ProjectileFit(
        max_speed_ms=24.0,
        launch_speed_ms=20.0,
        launch_angle_deg=38.0,
        max_height_m=7.5,
        height_at_target_m=2.1,
        height_ci_m=(6.0, 9.0),
        rmse_m=0.2,
        n_samples=12,
    )


def test_zone_occupancy_counts():
    positions = [
        # attackers
        _pos(1, Team.ATTACKING, 3.0, 34.0),          # GK area (+PEN)
        _pos(2, Team.ATTACKING, 10.0, 34.0),         # PEN only
        _pos(3, Team.ATTACKING, 19.0, 34.0),         # EDGE
        _pos(4, Team.ATTACKING, 3.0, 4.0),           # short-pass option (+PEN? no: y=4 outside pen)
        # defenders
        _pos(5, Team.DEFENDING, 2.0, 28.0),          # near-post band (+GK +PEN +NEAR)
        _pos(6, Team.DEFENDING, 2.0, 40.0),          # far-post band (+GK +PEN)
        _pos(7, Team.DEFENDING, 19.0, 34.0),         # EDGE
        _pos(8, Team.DEFENDING, 8.0, 20.0),          # NEAR area (+PEN)
        # goalkeeper (defending) in GK area -> excluded from excl-GK counts
        _pos(9, Team.DEFENDING, 1.0, 34.0, is_gk=True),
    ]
    row = build_feature_row(
        positions, _fit(),
        clip_id="c1", corner_side=CornerSide.LEFT,
        t_kick_frame=100, t_contact_frame=140,
    )

    assert row.num_short_pass_options == 1                 # player 4
    assert row.num_att_players_in_gk_area == 1             # player 1
    assert row.num_att_players_in_pen_area == 2            # players 1, 2
    assert row.num_att_player_in_edge_area == 1            # player 3

    # defenders: GK-area count excludes the goalkeeper (players 5, 6; not 9)
    assert row.num_def_players_in_gk_area == 2
    # PENALTY count is NOT excl-GK: players 5, 6, 8 and the GK (9); 7 is EDGE (x=19)
    assert row.num_def_players_in_pen_area == 4
    assert row.num_def_near_post == 1                      # player 5
    assert row.num_def_far_post == 1                       # player 6
    # NEAR area (not excl-GK): player 5 (y=28<30.34) and player 8 (8, 20)
    assert row.num_def_in_near_area == 2
    assert row.num_def_player_in_edge_area == 1            # player 7


def test_goalkeeper_excluded_from_gk_and_post_counts():
    positions = [
        _pos(1, Team.DEFENDING, 1.0, 34.0, is_gk=True),   # GK on the line
        _pos(2, Team.DEFENDING, 2.0, 28.0),               # outfield near post
    ]
    row = build_feature_row(
        positions, None,
        clip_id="c1", corner_side=CornerSide.LEFT,
        t_kick_frame=1, t_contact_frame=None,
    )
    assert row.num_def_players_in_gk_area == 1   # only the outfielder
    assert row.num_def_near_post == 1
    assert row.num_def_far_post == 0


def test_delivery_metrics_from_fit():
    row = build_feature_row(
        [], _fit(),
        clip_id="c1", corner_side=CornerSide.LEFT,
        t_kick_frame=1, t_contact_frame=2,
    )
    assert row.pass_max_height_in_m == 7.5
    assert row.pass_speed_max_in_ms == 24.0
    assert row.pass_hight_in_m_at_target == 2.1


def test_delivery_metrics_none_without_fit():
    row = build_feature_row(
        [], None,
        clip_id="c1", corner_side=CornerSide.LEFT,
        t_kick_frame=1, t_contact_frame=None,
    )
    assert row.pass_max_height_in_m is None
    assert row.pass_speed_max_in_ms is None
    assert row.pass_hight_in_m_at_target is None


def test_only_selected_moment_is_counted():
    positions = [
        _pos(1, Team.ATTACKING, 3.0, 34.0, moment=Moment.T_KICK),
        _pos(2, Team.ATTACKING, 3.0, 34.0, moment=Moment.T_CONTACT),
    ]
    kick = build_feature_row(
        positions, None, clip_id="c1", corner_side=CornerSide.LEFT,
        t_kick_frame=1, t_contact_frame=2, moment=Moment.T_KICK,
    )
    contact = build_feature_row(
        positions, None, clip_id="c1", corner_side=CornerSide.LEFT,
        t_kick_frame=1, t_contact_frame=2, moment=Moment.T_CONTACT,
    )
    assert kick.num_att_players_in_gk_area == 1
    assert contact.num_att_players_in_gk_area == 1
    # each moment sees exactly its own one player
    assert kick.num_att_players_in_pen_area == 1
    assert contact.num_att_players_in_pen_area == 1


def test_taker_excluded_from_short_pass_options():
    # two attackers standing in a short-pass band; one of them is the taker
    positions = [
        _pos(1, Team.ATTACKING, 2.0, 8.0),   # short-pass receiver
        _pos(2, Team.ATTACKING, 8.0, 2.0),   # the corner taker, in a band
    ]
    without = build_feature_row(
        positions, None, clip_id="c1", corner_side=CornerSide.LEFT,
        t_kick_frame=1, t_contact_frame=None,
    )
    with_taker = build_feature_row(
        positions, None, clip_id="c1", corner_side=CornerSide.LEFT,
        t_kick_frame=1, t_contact_frame=None, taker_player_id=2,
    )
    assert without.num_short_pass_options == 2   # taker unknown -> both counted
    assert with_taker.num_short_pass_options == 1  # taker (id 2) excluded


def test_empty_positions_yield_zero_counts():
    row = build_feature_row(
        [], None, clip_id="c1", corner_side=CornerSide.RIGHT,
        t_kick_frame=None, t_contact_frame=None,
    )
    counts = [
        row.num_short_pass_options, row.num_def_in_near_area,
        row.num_att_players_in_gk_area, row.num_def_players_in_gk_area,
        row.num_att_players_in_pen_area, row.num_def_players_in_pen_area,
        row.num_def_near_post, row.num_def_far_post,
        row.num_att_player_in_edge_area, row.num_def_player_in_edge_area,
    ]
    assert counts == [0] * 10


def test_feature_row_as_row_has_all_columns_in_order():
    row = build_feature_row(
        [], _fit(), clip_id="c1", corner_side=CornerSide.LEFT,
        t_kick_frame=10, t_contact_frame=20,
    )
    d = row.as_row()
    assert tuple(d.keys()) == FEATURE_COLUMNS
    assert d["corner_side"] == "left"
    assert d["zone_geometry_version"] == row.zone_geometry_version
