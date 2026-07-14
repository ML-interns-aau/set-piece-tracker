"""Pitch-zone model (FR-013, design 5.10; geometry from PRD Appendix C).

Pure coordinate-space region tests against a calibrated, canonically-oriented
position (metres). Nothing here touches cv2/YOLO/file-IO: a zone is a predicate
over ``(x, y)`` and the whole model is unit-testable against hardcoded points.

Coordinate convention (see ``domain.pitch``): x = 0 is the analysed goal line
growing toward halfway; y runs 0..68 across the width; the corner is taken from
the y = 0 side (canonical), so the **near post** is (0, 30.34) and the **far
post** is (0, 37.66). Callers must bring positions into this canonical frame
(``geometry.orientation.to_canonical``) before testing — near/far mean the same
thing across every clip only after that.

Boundary convention (football "lines belong to the areas they bound"): rectangle
bounds are inclusive by default, so a player on the penalty-area line is *inside*
the penalty area. Where two otherwise-disjoint zones abut (the 16.5 m line is
shared by PENALTY and EDGE), the line is given to the official area and the
synthetic neighbour takes its shared edge *exclusive*, so a point on the line is
never counted in both (see EDGE below). Deliberately *nested* zones (GK inside
PENALTY, the post bands inside GK) still overlap by design — a player can be in
both, and the features count them separately.

The geometry is **versioned** (:data:`ZONE_GEOMETRY_VERSION`): a later revision
(PRD §11 confirmation is still pending) re-evaluates this stage against stored
positions, never re-tracking. NEAR, EDGE and short-pass zones are **provisional**
and carry ``provisional=True`` in code and output alike until the FA-project
definitions are confirmed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.domain.pitch import PITCH, PitchModel

# Bump on any geometry change so stored feature rows stay attributable.
ZONE_GEOMETRY_VERSION = "v0.1-provisional"

# Provisional zone extents not fixed by the FIFA pitch model (PRD Appendix C).
EDGE_OUTER_X_M = 22.0          # EDGE area reaches ~22 m from the goal line
SHORT_PASS_RADIUS_M = 12.0     # short-pass options sit within ~12 m of the corner
SHORT_PASS_BAND_DEPTH_M = 5.0  # how far off each line a short-corner receiver stands


# --- zone primitives --------------------------------------------------------
@dataclass(frozen=True)
class RectZone:
    """Axis-aligned rectangle in pitch metres.

    Each boundary is inclusive by default (a line belongs to the area it bounds).
    A shared edge with an adjacent, disjoint zone is set exclusive on the side
    that does *not* own the line, so a point exactly on it lands in one zone
    only.
    """

    name: str
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    provisional: bool = False
    x_min_inclusive: bool = True
    x_max_inclusive: bool = True
    y_min_inclusive: bool = True
    y_max_inclusive: bool = True

    def contains(self, x: float, y: float) -> bool:
        lo_x = self.x_min <= x if self.x_min_inclusive else self.x_min < x
        hi_x = x <= self.x_max if self.x_max_inclusive else x < self.x_max
        lo_y = self.y_min <= y if self.y_min_inclusive else self.y_min < y
        hi_y = y <= self.y_max if self.y_max_inclusive else y < self.y_max
        return lo_x and hi_x and lo_y and hi_y


@dataclass(frozen=True)
class UnionZone:
    """A zone that is the union of its parts (contains = any part contains).

    Used for the short-pass target zones, which Appendix C describes as an
    L-shape "along the goal line and up the touchline" — two bands hugging the
    lines that meet at the corner, rather than the full quadrant between them.
    """

    name: str
    parts: tuple[RectZone, ...]
    provisional: bool = False

    def contains(self, x: float, y: float) -> bool:
        return any(part.contains(x, y) for part in self.parts)


@dataclass(frozen=True)
class ZoneModel:
    """The full set of named zones for one geometry version (FR-013).

    Built from a :class:`PitchModel` so the FIFA-fixed zones (GK, PENALTY,
    post bands) stay in sync with ``domain.pitch``; the provisional extents come
    from the module constants above.
    """

    version: str
    gk_area: RectZone
    penalty_area: RectZone
    near_post_band: RectZone
    far_post_band: RectZone
    near_area: RectZone
    edge_area: RectZone
    short_pass: UnionZone
    provisional_zone_names: frozenset[str] = field(default_factory=frozenset)


def build_zone_model(pitch: PitchModel = PITCH) -> ZoneModel:
    """Construct the Appendix-C zone model in canonical orientation.

    All extents derive from ``pitch`` (FIFA dimensions) plus the provisional
    module constants, so a pitch override or a constant change flows through
    consistently.
    """
    centre_y = pitch.width_m / 2.0
    gk_half_w = pitch.goal_area_width_m / 2.0        # 9.16
    pen_half_w = pitch.penalty_area_width_m / 2.0    # 20.16
    post_half_w = pitch.goal_width_m / 2.0           # 3.66

    gk_depth = pitch.goal_area_depth_m               # 5.5
    pen_depth = pitch.penalty_area_depth_m           # 16.5

    near_post_y = centre_y - post_half_w             # 30.34
    far_post_y = centre_y + post_half_w              # 37.66
    gk_near_y = centre_y - gk_half_w                 # 24.84
    gk_far_y = centre_y + gk_half_w                  # 43.16
    pen_near_y = centre_y - pen_half_w               # 13.84
    pen_far_y = centre_y + pen_half_w                # 54.16

    gk_area = RectZone("GK", 0.0, gk_depth, gk_near_y, gk_far_y)
    penalty_area = RectZone("PENALTY", 0.0, pen_depth, pen_near_y, pen_far_y)
    # post bands: within the GK area, from the post line out to the 6-yard edge
    near_post_band = RectZone("NEAR_POST", 0.0, gk_depth, gk_near_y, near_post_y)
    far_post_band = RectZone("FAR_POST", 0.0, gk_depth, far_post_y, gk_far_y)
    # NEAR area (provisional): near-post side of the box, from the near pen-area
    # edge to the near goalpost extended, full box depth
    near_area = RectZone(
        "NEAR", 0.0, pen_depth, pen_near_y, near_post_y, provisional=True
    )
    # EDGE area (provisional): 16.5 m line out to ~22 m, penalty-area width. The
    # 16.5 m line belongs to PENALTY, so EDGE's inner edge is exclusive — a
    # point exactly on the line counts as PENALTY, never as both.
    edge_area = RectZone(
        "EDGE", pen_depth, EDGE_OUTER_X_M, pen_near_y, pen_far_y,
        provisional=True, x_min_inclusive=False,
    )
    # short-pass target zones (provisional): an L-shape hugging the goal line and
    # the touchline out to ~12 m from the corner, each band SHORT_PASS_BAND_DEPTH
    # m off its line. Excludes the diagonal interior of the quadrant, where a
    # receiver would not stand.
    goal_line_band = RectZone(
        "SHORT_PASS_GOAL_LINE",
        0.0, SHORT_PASS_BAND_DEPTH_M, 0.0, SHORT_PASS_RADIUS_M, provisional=True,
    )
    touchline_band = RectZone(
        "SHORT_PASS_TOUCHLINE",
        0.0, SHORT_PASS_RADIUS_M, 0.0, SHORT_PASS_BAND_DEPTH_M, provisional=True,
    )
    short_pass = UnionZone(
        "SHORT_PASS", (goal_line_band, touchline_band), provisional=True
    )

    provisional = frozenset(
        z.name
        for z in (near_area, edge_area, short_pass)
        if z.provisional
    )
    return ZoneModel(
        version=ZONE_GEOMETRY_VERSION,
        gk_area=gk_area,
        penalty_area=penalty_area,
        near_post_band=near_post_band,
        far_post_band=far_post_band,
        near_area=near_area,
        edge_area=edge_area,
        short_pass=short_pass,
        provisional_zone_names=provisional,
    )


# Module-level default model for callers that don't override the pitch.
ZONES = build_zone_model()
