"""FR-013 - zone containment against hardcoded canonical-frame coordinates.

Reference geometry (canonical orientation, metres): goal line at x = 0, near
post (0, 30.34), far post (0, 37.66), centre y = 34.
  * GK area:      x 0..5.5,   y 24.84..43.16
  * PENALTY area: x 0..16.5,  y 13.84..54.16
  * near-post band: x 0..5.5, y 24.84..30.34
  * far-post band:  x 0..5.5, y 37.66..43.16
  * NEAR area:    x 0..16.5,  y 13.84..30.34   (provisional)
  * EDGE area:    x 16.5..22, y 13.84..54.16   (provisional, inner edge exclusive)
  * short-pass:   L-shape hugging the goal line and touchline out to ~12 m from
                  the (0, 0) corner, each band ~5 m off its line (provisional)
"""

from __future__ import annotations

from src.features.zones import (
    EDGE_OUTER_X_M,
    SHORT_PASS_RADIUS_M,
    ZONE_GEOMETRY_VERSION,
    ZONES,
    build_zone_model,
)


def test_point_inside_gk_area():
    # near the 6-yard box centre
    assert ZONES.gk_area.contains(3.0, 34.0)
    assert ZONES.penalty_area.contains(3.0, 34.0)  # GK area is inside PENALTY


def test_point_inside_far_post_band():
    # between the far post (37.66) and the far GK edge (43.16), shallow
    assert ZONES.far_post_band.contains(2.0, 40.0)
    # the same point is not on the near-post band
    assert not ZONES.near_post_band.contains(2.0, 40.0)


def test_point_inside_near_post_band():
    assert ZONES.near_post_band.contains(2.0, 28.0)
    assert not ZONES.far_post_band.contains(2.0, 28.0)


def test_point_in_penalty_but_outside_gk():
    # x = 10 is beyond the 5.5 m goal-area depth but within the 16.5 m box
    p = (10.0, 34.0)
    assert ZONES.penalty_area.contains(*p)
    assert not ZONES.gk_area.contains(*p)


def test_point_inside_edge_area():
    # between the 16.5 m line and ~22 m
    assert ZONES.edge_area.contains(19.0, 34.0)
    # just inside the box (x < 16.5) is NOT the edge area
    assert not ZONES.edge_area.contains(15.0, 34.0)
    # beyond the outer edge extent is out
    assert not ZONES.edge_area.contains(EDGE_OUTER_X_M + 1.0, 34.0)


def test_point_inside_short_pass_zone():
    # short-pass is an L-shape: bands hugging the goal line and touchline out
    # to ~12 m from the corner, each ~5 m off its line.
    assert ZONES.short_pass.contains(2.0, 8.0)      # along the goal line band
    assert ZONES.short_pass.contains(8.0, 2.0)      # up the touchline band
    assert ZONES.short_pass.contains(3.0, 4.0)      # in the corner, both bands
    # the diagonal interior of the quadrant is NOT a receiver position
    assert not ZONES.short_pass.contains(8.0, 8.0)
    # beyond ~12 m from the corner along a line is out
    assert not ZONES.short_pass.contains(2.0, 13.0)
    assert not ZONES.short_pass.contains(13.0, 2.0)


def test_point_inside_near_area():
    # near-post side of the box: y below the near post, within box depth
    assert ZONES.near_area.contains(8.0, 20.0)
    # far-post side is not the NEAR area
    assert not ZONES.near_area.contains(8.0, 45.0)


def test_point_outside_all_zones():
    # well up the pitch, past every zone
    p = (40.0, 34.0)
    assert not ZONES.penalty_area.contains(*p)
    assert not ZONES.edge_area.contains(*p)
    assert not ZONES.gk_area.contains(*p)
    assert not ZONES.short_pass.contains(*p)
    assert not ZONES.near_area.contains(*p)


def test_short_pass_excludes_negative_side():
    # the L-shape only covers the on-pitch quadrant (x>=0, y>=0)
    assert not ZONES.short_pass.contains(-1.0, 3.0)
    assert not ZONES.short_pass.contains(3.0, -1.0)


def test_shared_edge_is_not_double_counted():
    # the 16.5 m line is shared by PENALTY and EDGE; it belongs to PENALTY only,
    # so a point exactly on it is never counted in both (FR-013 boundary rule).
    on_line = (16.5, 34.0)
    assert ZONES.penalty_area.contains(*on_line)
    assert not ZONES.edge_area.contains(*on_line)
    # just past the line is EDGE, not PENALTY
    just_past = (16.5001, 34.0)
    assert not ZONES.penalty_area.contains(*just_past)
    assert ZONES.edge_area.contains(*just_past)


def test_penalty_and_goal_lines_belong_to_their_areas():
    # "lines belong to the areas they bound": a player on the line is inside
    assert ZONES.penalty_area.contains(0.0, 13.84)   # on the goal line + side line
    assert ZONES.penalty_area.contains(16.5, 54.16)  # penalty-area far corner
    assert ZONES.gk_area.contains(5.5, 24.84)        # goal-area corner


def test_nested_zones_still_overlap_by_design():
    # GK is inside PENALTY, near-post band is inside GK/NEAR — these SHOULD
    # dual-count; the boundary rule only de-duplicates touching disjoint zones.
    assert ZONES.gk_area.contains(3.0, 34.0) and ZONES.penalty_area.contains(3.0, 34.0)
    assert ZONES.near_post_band.contains(2.0, 28.0) and ZONES.near_area.contains(2.0, 28.0)


def test_provisional_zones_flagged():
    assert ZONES.near_area.provisional
    assert ZONES.edge_area.provisional
    assert ZONES.short_pass.provisional
    # FIFA-fixed zones are not provisional
    assert not ZONES.gk_area.provisional
    assert not ZONES.penalty_area.provisional
    assert ZONES.provisional_zone_names == {"NEAR", "EDGE", "SHORT_PASS"}


def test_zone_model_carries_version():
    assert ZONES.version == ZONE_GEOMETRY_VERSION
    assert build_zone_model().version == ZONE_GEOMETRY_VERSION
    assert SHORT_PASS_RADIUS_M == 12.0


def test_gk_area_boundaries_from_pitch_model():
    # exact FIFA-derived corners: goal area 18.32 x 5.5, centred on y = 34
    assert ZONES.gk_area.contains(0.0, 24.84)   # near-side goal-line corner
    assert ZONES.gk_area.contains(5.5, 43.16)   # far-side front corner
    assert not ZONES.gk_area.contains(0.0, 24.0)   # just outside in y
    assert not ZONES.gk_area.contains(5.6, 34.0)   # just past the 6-yard depth
