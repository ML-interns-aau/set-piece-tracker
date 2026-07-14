from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from src.domain.models import Team
from src.features.zones import RectZone, UnionZone
from src.verification.config import VERIFICATION_CONFIG, VerificationConfig
from src.verification.events import EventType


def _project(H_inv: np.ndarray, metric_xy: tuple[float, float]) -> tuple[int, int]:
    pt = np.array([metric_xy[0], metric_xy[1], 1.0], dtype=np.float64)
    px = H_inv @ pt
    if abs(px[2]) < 1e-12:
        return (-1, -1)
    return (int(px[0] / px[2]), int(px[1] / px[2]))


def draw_player_box(
    frame: np.ndarray,
    bbox_xyxy: tuple[float, float, float, float],
    team: Team | None,
    is_goalkeeper: bool,
    track_id: int,
    config: VerificationConfig = VERIFICATION_CONFIG,
) -> None:
    x1, y1, x2, y2 = (int(round(v)) for v in bbox_xyxy)
    color = config.unknown_color if team is None else config.team_colors[team]
    thickness = config.box_thickness + (config.gk_thickness_delta if is_goalkeeper else 0)
    box_color = config.gk_accent_color if is_goalkeeper else color
    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, thickness)

    if is_goalkeeper:
        label = f"{config.gk_label} #{track_id}"
    elif team is None:
        label = f"UNK #{track_id}"
    else:
        label = f"{team.value.upper()} #{track_id}"
    cv2.putText(frame, label, (x1, max(0, y1 - 6)), config.font,
                config.label_font_scale, box_color, config.font_thickness)


def draw_ball(
    frame: np.ndarray,
    centre_px: tuple[float, float],
    predicted: bool,
    config: VerificationConfig = VERIFICATION_CONFIG,
) -> None:
    color = config.ball_predicted_color if predicted else config.ball_detected_color
    u, v = int(round(centre_px[0])), int(round(centre_px[1]))
    cv2.circle(frame, (u, v), config.ball_radius_px, color, -1)


def draw_ball_trail(
    frame: np.ndarray,
    trail_px: Sequence[tuple[int, int]],
    config: VerificationConfig = VERIFICATION_CONFIG,
) -> None:
    for point in trail_px[-config.ball_trail_max_points:]:
        cv2.circle(frame, point, config.ball_trail_radius_px, config.ball_trail_color, -1)


def _zone_color(zone: RectZone | UnionZone, config: VerificationConfig) -> tuple[int, int, int]:
    return config.zone_colors.get(zone.name, config.default_zone_color)


def _draw_rect_zone(
    frame: np.ndarray, zone: RectZone, H_inv: np.ndarray, color: tuple[int, int, int],
    config: VerificationConfig, label: str | None,
) -> None:
    corners_m = (
        (zone.x_min, zone.y_min), (zone.x_max, zone.y_min),
        (zone.x_max, zone.y_max), (zone.x_min, zone.y_max),
    )
    quad = np.array([_project(H_inv, c) for c in corners_m], dtype=np.int32)

    overlay = frame.copy()
    cv2.fillPoly(overlay, [quad], color)
    cv2.addWeighted(overlay, config.zone_fill_alpha, frame, 1 - config.zone_fill_alpha, 0, dst=frame)
    cv2.polylines(frame, [quad], True, color, config.zone_border_thickness)

    if label:
        centroid = quad.mean(axis=0).astype(int)
        cv2.putText(frame, label, tuple(centroid), config.font,
                    config.label_font_scale, color, config.font_thickness)


def draw_zone(
    frame: np.ndarray,
    zone: RectZone | UnionZone,
    H_inv: np.ndarray,
    config: VerificationConfig = VERIFICATION_CONFIG,
) -> None:
    color = _zone_color(zone, config)
    label = zone.name + (" (prov.)" if zone.provisional else "")

    if isinstance(zone, UnionZone):
        for i, part in enumerate(zone.parts):
            _draw_rect_zone(frame, part, H_inv, color, config, label if i == 0 else None)
    else:
        _draw_rect_zone(frame, zone, H_inv, color, config, label)


def draw_event_marker(
    frame: np.ndarray,
    event_type: EventType,
    confidence: float,
    config: VerificationConfig = VERIFICATION_CONFIG,
) -> None:
    color = config.event_colors.get(event_type, config.default_zone_color)
    box_w, box_h = config.marker_box_size_px
    h, w = frame.shape[:2]
    margin = config.marker_margin_px

    if config.marker_corner == "top_right":
        x1, y1 = w - margin - box_w, margin
    elif config.marker_corner == "bottom_left":
        x1, y1 = margin, h - margin - box_h
    elif config.marker_corner == "bottom_right":
        x1, y1 = w - margin - box_w, h - margin - box_h
    else:
        x1, y1 = margin, margin
    x2, y2 = x1 + box_w, y1 + box_h

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)
    cv2.putText(frame, event_type.value.upper(), (x1 + 10, y1 + 28), config.font,
                config.font_scale, (255, 255, 255), config.font_thickness)
    cv2.putText(frame, f"Confidence: {confidence:.2f}", (x1 + 10, y1 + 54), config.font,
                config.label_font_scale, (255, 255, 255), 1)
