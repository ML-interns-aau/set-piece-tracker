from __future__ import annotations

from dataclasses import dataclass, field

import cv2

from src.domain.models import Team
from src.verification.events import EventType

BGR = tuple[int, int, int]


@dataclass(frozen=True)
class VerificationConfig:
    team_colors: dict[Team, BGR] = field(default_factory=lambda: {
        Team.ATTACKING: (180, 105, 255),
        Team.DEFENDING: (128, 0, 128),
    })
    unknown_color: BGR = (128, 128, 128)
    box_thickness: int = 2
    gk_accent_color: BGR = (0, 215, 255)
    gk_thickness_delta: int = 2
    gk_label: str = "GK"
    zone_colors: dict[str, BGR] = field(default_factory=lambda: {
        "GK": (0, 255, 0),
        "PENALTY": (255, 255, 0),
        "NEAR_POST": (0, 200, 255),
        "FAR_POST": (255, 200, 0),
        "NEAR": (200, 0, 200),
        "EDGE": (0, 128, 255),
        "SHORT_PASS": (255, 0, 200),
        "SHORT_PASS_GOAL_LINE": (255, 0, 200),
        "SHORT_PASS_TOUCHLINE": (255, 0, 200),
    })
    default_zone_color: BGR = (255, 255, 0)
    zone_fill_alpha: float = 0.15
    zone_border_thickness: int = 2
    ball_detected_color: BGR = (0, 255, 0)
    ball_predicted_color: BGR = (0, 165, 255)
    ball_trail_color: BGR = (0, 255, 255)
    ball_radius_px: int = 6
    ball_trail_radius_px: int = 2
    ball_trail_max_points: int = 25
    event_colors: dict[EventType, BGR] = field(default_factory=lambda: {
        EventType.KICK: (0, 0, 255),
        EventType.CONTACT: (255, 0, 0),
        EventType.SHOT: (0, 140, 255),
        EventType.PASS: (0, 200, 0),
        EventType.HEADER: (255, 255, 0),
        EventType.CROSS: (255, 0, 255),
        EventType.SAVE: (0, 215, 255),
        EventType.GOAL: (0, 0, 0),
    })
    marker_duration_frames: int = 15
    marker_corner: str = "top_left"
    marker_margin_px: int = 20
    marker_box_size_px: tuple[int, int] = (260, 70)
    font: int = cv2.FONT_HERSHEY_SIMPLEX
    font_scale: float = 0.6
    font_thickness: int = 2
    label_font_scale: float = 0.5
    jsonl_log_path_template: str = "outputs/verification/{clip_id}_reviews.jsonl"


VERIFICATION_CONFIG = VerificationConfig()
