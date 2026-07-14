"""Pure data types produced by the geometry & moments plane (no cv2/ultralytics).

These are the value objects that cross plane boundaries (see the I1-I13
interface contracts in CLAUDE.md). numpy is allowed here; heavy CV/ML deps
are not.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class CornerSide(str, Enum):
    """Which side the corner is taken from, in the broadcast frame."""

    LEFT = "left"    # canonical: taker near the (0, 0) corner
    RIGHT = "right"  # mirrored (y -> width - y) into canonical


class Source(str, Enum):
    """Provenance of an automatically-produced value."""

    AUTO = "auto"
    MANUAL = "manual"


@dataclass(frozen=True)
class Calibration:
    """Pixel -> pitch-metre mapping for one clip (interface I7).

    ``H`` maps pixel (u, v) to pitch (x, y) in metres. ``reprojection_error_m``
    is the mean re-projection error over the fitted points and feeds the
    reliability score (FR-016).
    """

    H: np.ndarray                    # (3, 3)
    reprojection_error_m: float
    points_used: int
    source: Source = Source.AUTO

    def __post_init__(self) -> None:
        h = np.asarray(self.H, dtype=np.float64)
        if h.shape != (3, 3):
            raise ValueError(f"H must be 3x3, got {h.shape}")
        object.__setattr__(self, "H", h)


@dataclass(frozen=True)
class CalibrationTrack:
    """Per-frame calibration for a clip whose camera pans/zooms (interface I7).

    Holds one :class:`Calibration` per (sampled) frame. ``at(frame_idx)`` returns
    the calibration for a frame, falling back to the nearest computed frame. A
    ``static`` clip collapses to a single shared homography; ``discontinuity_frames``
    marks camera cuts across which calibrations must not be interpolated.
    """

    per_frame: dict[int, Calibration]
    static: bool = False
    discontinuity_frames: tuple[int, ...] = ()

    def at(self, frame_idx: int) -> Calibration | None:
        if not self.per_frame:
            return None
        exact = self.per_frame.get(frame_idx)
        if exact is not None:
            return exact
        nearest = min(self.per_frame, key=lambda k: abs(k - frame_idx))
        return self.per_frame[nearest]

    @property
    def mean_reprojection_error_m(self) -> float:
        vals = [c.reprojection_error_m for c in self.per_frame.values()]
        return float(sum(vals) / len(vals)) if vals else float("nan")


@dataclass(frozen=True)
class CornerSideResult:
    """Result of corner-side detection (interface I6)."""

    corner_side: CornerSide
    side_source: Source
    confidence: float  # 0-1


@dataclass(frozen=True)
class BallSample:
    """One ball observation on the pitch plane (metres). Part of I8's ball_track."""

    frame_idx: int
    x_m: float
    y_m: float


@dataclass(frozen=True)
class ProjectileFit:
    """Delivery-trajectory fit (interface I8, features 11-13).

    Height fields are ``None`` when height could not be estimated (the monocular
    limitation, FR-009); horizontal metrics (max speed) are still populated.
    """

    max_speed_ms: float
    launch_speed_ms: float
    launch_angle_deg: float | None
    max_height_m: float | None
    height_at_target_m: float | None
    height_ci_m: tuple[float, float] | None  # confidence interval on height
    rmse_m: float
    n_samples: int


@dataclass(frozen=True)
class KeyMoments:
    """Detected key-moment frames (interface I9).

    ``t_contact_frame`` is ``None`` when the ball reaches no player (FR-011).
    """

    t_kick_frame: int
    t_contact_frame: int | None
    t_kick_source: Source = Source.AUTO
    t_contact_source: Source = Source.AUTO


# --- Features & reliability plane (FR-012-016) -------------------------------
class Team(str, Enum):
    """Which team a tracked player belongs to (interface I4)."""

    ATTACKING = "attacking"
    DEFENDING = "defending"


class PositionSource(str, Enum):
    """Provenance of a per-player position (FR-012, FR-006, FR-019).

    ``DETECTED`` and ``EXTRAPOLATED`` are never merged (FR-006); a
    ``MANUALLY_CORRECTED`` value is a human judgment layered on top and is
    excluded from automated accuracy self-assessment (FR-019).
    """

    DETECTED = "detected"
    EXTRAPOLATED = "extrapolated"
    MANUALLY_CORRECTED = "manually_corrected"


class Moment(str, Enum):
    """The two key moments a position row can describe (interface I9/I10)."""

    T_KICK = "t_kick"
    T_CONTACT = "t_contact"


# Column order of the positions file (Data Model / design 6.2). Self-describing
# outputs write these names in this order.
POSITIONS_COLUMNS: tuple[str, ...] = (
    "clip_id", "moment", "player_id", "team", "is_goalkeeper",
    "pitch_x", "pitch_y", "velocity_x", "velocity_y",
    "position_source", "reliability_score", "velocity_window_s",
)


@dataclass(frozen=True)
class PlayerPosition:
    """One player at one key moment (interface I10, positions-file row 6.2).

    This is the atomic unit every downstream feature and export consumes
    (design 5.9). Positions are in **canonical orientation** metres (see
    ``domain.pitch`` / ``geometry.orientation``). Velocity fields are populated
    only at ``t_kick`` (from the pre-kick window) and are ``None`` otherwise.
    ``is_goalkeeper`` is only meaningful for the defending team.
    """

    clip_id: str
    moment: Moment
    player_id: int
    team: Team
    is_goalkeeper: bool
    pitch_x: float
    pitch_y: float
    position_source: PositionSource
    reliability_score: float
    velocity_x: float | None = None
    velocity_y: float | None = None
    velocity_window_s: float | None = None

    def as_row(self) -> dict[str, object]:
        """Row dict keyed by :data:`POSITIONS_COLUMNS` for CSV/JSON export."""
        return {
            "clip_id": self.clip_id,
            "moment": self.moment.value,
            "player_id": self.player_id,
            "team": self.team.value,
            "is_goalkeeper": self.is_goalkeeper,
            "pitch_x": self.pitch_x,
            "pitch_y": self.pitch_y,
            "velocity_x": self.velocity_x,
            "velocity_y": self.velocity_y,
            "position_source": self.position_source.value,
            "reliability_score": self.reliability_score,
            "velocity_window_s": self.velocity_window_s,
        }


# Column order of the feature row (Data Model / design 6.3): metadata first,
# then the 13 Appendix-A feature columns in their delivered order/spelling.
FEATURE_COLUMNS: tuple[str, ...] = (
    "clip_id", "corner_side", "t_kick_frame", "t_contact_frame",
    "zone_geometry_version",
    "num_short_pass_options", "num_def_in_near_area",
    "num_att_players_in_gk_area", "num_def_players_in_gk_area",
    "num_att_players_in_pen_area", "num_def_players_in_pen_area",
    "num_def_near_post", "num_def_far_post",
    "num_att_player_in_edge_area", "num_def_player_in_edge_area",
    "pass_max_height_in_m", "pass_speed_max_in_ms", "pass_hight_in_m_at_target",
)


@dataclass(frozen=True)
class FeatureRow:
    """One corner's feature row (interface I11, feature-row 6.3).

    The 13 feature columns follow PRD Appendix A exactly, including the
    delivered spelling ``pass_hight_in_m_at_target``. Delivery metrics
    (11-13) are ``None`` when the trajectory height / target could not be
    estimated (the monocular limitation, FR-009).
    """

    clip_id: str
    corner_side: CornerSide
    t_kick_frame: int | None
    t_contact_frame: int | None
    zone_geometry_version: str
    # zone-occupancy counts (Appendix A, 1-10)
    num_short_pass_options: int
    num_def_in_near_area: int
    num_att_players_in_gk_area: int
    num_def_players_in_gk_area: int
    num_att_players_in_pen_area: int
    num_def_players_in_pen_area: int
    num_def_near_post: int
    num_def_far_post: int
    num_att_player_in_edge_area: int
    num_def_player_in_edge_area: int
    # delivery metrics (Appendix A, 11-13)
    pass_max_height_in_m: float | None
    pass_speed_max_in_ms: float | None
    pass_hight_in_m_at_target: float | None

    def as_row(self) -> dict[str, object]:
        """Row dict keyed by :data:`FEATURE_COLUMNS` for CSV/JSON export."""
        return {
            "clip_id": self.clip_id,
            "corner_side": self.corner_side.value,
            "t_kick_frame": self.t_kick_frame,
            "t_contact_frame": self.t_contact_frame,
            "zone_geometry_version": self.zone_geometry_version,
            "num_short_pass_options": self.num_short_pass_options,
            "num_def_in_near_area": self.num_def_in_near_area,
            "num_att_players_in_gk_area": self.num_att_players_in_gk_area,
            "num_def_players_in_gk_area": self.num_def_players_in_gk_area,
            "num_att_players_in_pen_area": self.num_att_players_in_pen_area,
            "num_def_players_in_pen_area": self.num_def_players_in_pen_area,
            "num_def_near_post": self.num_def_near_post,
            "num_def_far_post": self.num_def_far_post,
            "num_att_player_in_edge_area": self.num_att_player_in_edge_area,
            "num_def_player_in_edge_area": self.num_def_player_in_edge_area,
            "pass_max_height_in_m": self.pass_max_height_in_m,
            "pass_speed_max_in_ms": self.pass_speed_max_in_ms,
            "pass_hight_in_m_at_target": self.pass_hight_in_m_at_target,
        }
