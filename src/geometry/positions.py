"""Player positions at a key moment (interface I10, design 5.9).

Pure producer of :class:`PlayerPosition` rows: project each tracked player's
**foot-point** (bottom-center of the bbox, the point the homography maps -- I5)
through the frame's calibration to metric pitch coords, normalize to canonical
orientation, and attach team / goalkeeper / provenance / reliability.

No cv2/YOLO here -- perception (detection, tracking, team + GK classification)
runs upstream and hands us plain tuples, so this module unit-tests without a clip
or model weights, like the rest of the pure core.

**Team resolution for a corner.** The team classifier only yields anonymous team
clusters (0 / 1) plus GK / referee sentinels; it does not know which cluster is
*attacking*. For a corner the attacking team is the one taking it, so we identify
the taker as the player nearest the (dead) ball at ``t_kick`` and label that
player's cluster ATTACKING (``resolve_attacking_side``). This also yields the
taker's track id, which the feature stage excludes from the short-pass-options
count. This heuristic is provisional -- a catalog override can set it by hand.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from src.domain.models import (
    Calibration,
    CornerSide,
    Moment,
    PlayerPosition,
    PositionSource,
    Team,
)
from src.features.reliability import ReliabilityInputs, reliability_score
from src.geometry.calibration import apply_homography
from src.geometry.orientation import to_canonical

# Team / role sentinel codes. These mirror ``src.engine.team_classifier`` but are
# redeclared here so this pure module carries no cv2 / sklearn import. Outfield
# teams are non-negative ints (0 / 1); the rest are roles we do not count.
UNKNOWN_CODE = -1
REFEREE_CODE = -2
GK_CODE = -3
_NON_TEAM_CODES = frozenset({UNKNOWN_CODE, REFEREE_CODE})

Bbox = tuple[float, float, float, float]
# One tracked player at a moment: (track_id, bbox (x1,y1,x2,y2) px, detection conf 0..1).
Track = tuple[int, Bbox, float]


def foot_point(bbox_xyxy: Bbox) -> tuple[float, float]:
    """Bottom-center of a bbox -- the pixel the homography maps to the pitch (I5)."""
    x1, _y1, x2, y2 = bbox_xyxy
    return ((float(x1) + float(x2)) / 2.0, float(y2))


def _is_outfield_code(code: int) -> bool:
    """A countable outfield team cluster (not GK, referee, or unknown)."""
    return code not in _NON_TEAM_CODES and code != GK_CODE


def resolve_attacking_side(
    tracks_at_kick: Sequence[Track],
    ball_px: tuple[float, float] | None,
    team_codes: Mapping[int, int],
) -> tuple[int | None, int | None]:
    """Identify ``(attacking_team_code, taker_track_id)`` (provisional heuristic).

    The corner taker is the player whose foot-point is closest to the dead ball
    at ``t_kick``; that player's outfield cluster is the attacking side. Falls
    back to the lowest valid outfield code (and no taker) when the ball position
    or any valid nearest player is missing. Returns ``(None, None)`` if no
    outfield team could be resolved at all.
    """
    valid = [
        (tid, bbox)
        for (tid, bbox, _conf) in tracks_at_kick
        if _is_outfield_code(team_codes.get(tid, UNKNOWN_CODE))
    ]
    if ball_px is not None and valid:
        bx, by = float(ball_px[0]), float(ball_px[1])

        def _dist2(item: tuple[int, Bbox]) -> float:
            fx, fy = foot_point(item[1])
            return (fx - bx) ** 2 + (fy - by) ** 2

        taker_tid, _bbox = min(valid, key=_dist2)
        return team_codes[taker_tid], taker_tid

    codes = sorted({team_codes[tid] for tid, _ in valid}) if valid else []
    return (codes[0] if codes else None), None


def build_player_positions(
    tracks_at_moment: Sequence[Track],
    calibration: Calibration | None,
    corner_side: CornerSide,
    *,
    clip_id: str,
    moment: Moment,
    attacking_code: int | None,
    team_codes: Mapping[int, int],
    gk_track_ids: frozenset[int] = frozenset(),
    gap_frames: Mapping[int, float] | None = None,
) -> list[PlayerPosition]:
    """Project tracked players to canonical metric coords -> ``PlayerPosition`` rows.

    Referee / unknown tracks are dropped (not countable players). A track flagged
    GK (team code ``GK_CODE`` or listed in ``gk_track_ids``) is labelled DEFENDING
    with ``is_goalkeeper=True``; any other outfield track is ATTACKING when its
    code equals ``attacking_code``, else DEFENDING.

    ``gap_frames`` maps a track id to the number of frames since its last real
    detection (0 / absent => detected at the moment); a non-zero gap marks the
    position ``EXTRAPOLATED`` (never merged with detected values, FR-006). When
    ``calibration`` is ``None`` the pitch coords are ``NaN`` and reliability
    collapses to 0 -- the caller routes such a clip to manual review rather than
    trusting a metric-less position.
    """
    reproj = calibration.reprojection_error_m if calibration is not None else None
    rows: list[PlayerPosition] = []

    for track_id, bbox, conf in tracks_at_moment:
        code = team_codes.get(track_id, UNKNOWN_CODE)
        is_gk = code == GK_CODE or track_id in gk_track_ids
        if not is_gk and not _is_outfield_code(code):
            continue  # referee / unknown -- excluded from positions and coverage

        if is_gk:
            team = Team.DEFENDING
        elif attacking_code is not None and code == attacking_code:
            team = Team.ATTACKING
        else:
            team = Team.DEFENDING

        fx, fy = foot_point(bbox)
        if calibration is not None:
            mapped = apply_homography(calibration.H, np.array([[fx, fy]]))
            canon = to_canonical(mapped, corner_side)[0]
            pitch_x, pitch_y = float(canon[0]), float(canon[1])
        else:
            pitch_x = pitch_y = float("nan")

        gap = float(gap_frames.get(track_id, 0.0)) if gap_frames else 0.0
        source = PositionSource.EXTRAPOLATED if gap > 0.0 else PositionSource.DETECTED
        share = 1.0 if source is PositionSource.EXTRAPOLATED else 0.0
        score = reliability_score(
            ReliabilityInputs(
                detection_confidence=float(conf),
                gap_frames=gap,
                extrapolation_share=share,
                reprojection_error_m=reproj,
            )
        )

        rows.append(
            PlayerPosition(
                clip_id=clip_id,
                moment=moment,
                player_id=int(track_id),
                team=team,
                is_goalkeeper=is_gk,
                pitch_x=pitch_x,
                pitch_y=pitch_y,
                position_source=source,
                reliability_score=score,
            )
        )

    return rows
