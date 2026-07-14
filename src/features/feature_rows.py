"""Feature computation (FR-014/015, design 5.11; PRD Appendix A).

Turns the per-player position records (FR-012, interface I10) plus the delivery
trajectory fit (FR-009, interface I8) into one :class:`FeatureRow` per corner
(interface I11). Every feature is a pure function of stored positions and the
trajectory fit alone, so a zone-geometry or feature-definition change recomputes
this stage without re-running detection or tracking.

Zone-occupancy counts (Appendix A, 1-10) are evaluated at a chosen moment
(default ``t_kick``, extendable to ``t_contact`` per FR-015) by testing each
position against the versioned zone model. Delivery metrics (11-13) read
straight off the projectile fit. The goalkeeper is excluded from the counts
Appendix A marks "excl. GK" (GK-area and post-band defender counts).

Positions must already be in **canonical orientation** metres (the same frame
the zone model lives in); callers normalise with
``geometry.orientation.to_canonical`` upstream when producing I10 rows.
"""

from __future__ import annotations

from collections.abc import Iterable

from src.domain.models import (
    CornerSide,
    FeatureRow,
    Moment,
    PlayerPosition,
    ProjectileFit,
    Team,
)
from src.features.zones import ZONES, ZoneModel


def _is_att(p: PlayerPosition) -> bool:
    return p.team == Team.ATTACKING


def _is_def(p: PlayerPosition) -> bool:
    return p.team == Team.DEFENDING


def _is_def_outfield(p: PlayerPosition) -> bool:
    """Defending player that is not the goalkeeper (for the excl.-GK counts)."""
    return p.team == Team.DEFENDING and not p.is_goalkeeper


def build_feature_row(
    positions: Iterable[PlayerPosition],
    fit: ProjectileFit | None,
    *,
    clip_id: str,
    corner_side: CornerSide,
    t_kick_frame: int | None,
    t_contact_frame: int | None,
    zones: ZoneModel = ZONES,
    moment: Moment = Moment.T_KICK,
    taker_player_id: int | None = None,
) -> FeatureRow:
    """Compute one corner's feature row (PRD Appendix A, interface I11).

    ``positions`` are the I10 rows for the clip; only those matching ``moment``
    (default ``t_kick``) are counted. ``fit`` supplies the delivery metrics
    (11-13); pass ``None`` when no trajectory was fit, leaving them ``None``.

    ``taker_player_id`` is the corner taker's track ID, excluded from the
    short-pass-options count per Appendix C ("excluding the taker"); the zone
    model itself cannot know which attacker took the corner, so the caller
    supplies it (``None`` = taker unknown, count all attackers in the zone).
    """
    at_moment = [p for p in positions if p.moment == moment]

    def count(predicate, zone) -> int:
        return sum(
            1 for p in at_moment
            if predicate(p) and zone.contains(p.pitch_x, p.pitch_y)
        )

    def _is_short_pass_option(p: PlayerPosition) -> bool:
        return _is_att(p) and p.player_id != taker_player_id

    num_short_pass_options = count(_is_short_pass_option, zones.short_pass)
    num_def_in_near_area = count(_is_def, zones.near_area)
    num_att_players_in_gk_area = count(_is_att, zones.gk_area)
    # GK-area / post-band defender counts exclude the goalkeeper (Appendix A)
    num_def_players_in_gk_area = count(_is_def_outfield, zones.gk_area)
    num_att_players_in_pen_area = count(_is_att, zones.penalty_area)
    num_def_players_in_pen_area = count(_is_def, zones.penalty_area)
    num_def_near_post = count(_is_def_outfield, zones.near_post_band)
    num_def_far_post = count(_is_def_outfield, zones.far_post_band)
    num_att_player_in_edge_area = count(_is_att, zones.edge_area)
    num_def_player_in_edge_area = count(_is_def, zones.edge_area)

    max_height = fit.max_height_m if fit is not None else None
    max_speed = fit.max_speed_ms if fit is not None else None
    height_at_target = fit.height_at_target_m if fit is not None else None

    return FeatureRow(
        clip_id=clip_id,
        corner_side=corner_side,
        t_kick_frame=t_kick_frame,
        t_contact_frame=t_contact_frame,
        zone_geometry_version=zones.version,
        num_short_pass_options=num_short_pass_options,
        num_def_in_near_area=num_def_in_near_area,
        num_att_players_in_gk_area=num_att_players_in_gk_area,
        num_def_players_in_gk_area=num_def_players_in_gk_area,
        num_att_players_in_pen_area=num_att_players_in_pen_area,
        num_def_players_in_pen_area=num_def_players_in_pen_area,
        num_def_near_post=num_def_near_post,
        num_def_far_post=num_def_far_post,
        num_att_player_in_edge_area=num_att_player_in_edge_area,
        num_def_player_in_edge_area=num_def_player_in_edge_area,
        pass_max_height_in_m=max_height,
        pass_speed_max_in_ms=max_speed,
        pass_hight_in_m_at_target=height_at_target,
    )
