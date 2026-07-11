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
