from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from src.domain.models import Team


class EventType(str, Enum):
    KICK = "kick"
    CONTACT = "contact"
    SHOT = "shot"
    PASS = "pass"
    HEADER = "header"
    CROSS = "cross"
    SAVE = "save"
    GOAL = "goal"


EVENT_COLUMNS: tuple[str, ...] = (
    "frame", "event_type", "confidence", "player_ids", "team", "corrected",
    "reliability_score",
)


@dataclass(frozen=True)
class EventRecord:
    frame: int
    event_type: EventType
    confidence: float
    player_ids: tuple[int, ...] = ()
    team: Team | None = None
    corrected: bool = False
    reliability_score: float | None = None

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")

    def as_row(self) -> dict[str, object]:
        return {
            "frame": self.frame,
            "event_type": self.event_type.value,
            "confidence": self.confidence,
            "player_ids": list(self.player_ids),
            "team": self.team.value if self.team is not None else None,
            "corrected": self.corrected,
            "reliability_score": self.reliability_score,
        }

    @staticmethod
    def from_dict(d: dict[str, object]) -> "EventRecord":
        team = d.get("team")
        reliability = d.get("reliability_score")
        player_ids = d.get("player_ids", ())
        return EventRecord(
            frame=int(d["frame"]),
            event_type=EventType(d["event_type"]),
            confidence=float(d["confidence"]),
            player_ids=tuple(int(p) for p in player_ids),
            team=Team(team) if team is not None else None,
            corrected=bool(d.get("corrected", False)),
            reliability_score=float(reliability) if reliability is not None else None,
        )


@dataclass(frozen=True)
class PipelineOutputEvents:
    clip_id: str
    events: tuple[EventRecord, ...]
    known_player_ids: frozenset[int] = field(default_factory=frozenset)
    fps: float | None = None

    def __post_init__(self) -> None:
        frames = [e.frame for e in self.events]
        dupes = sorted({f for f in frames if frames.count(f) > 1})
        if dupes:
            raise ValueError(
                f"duplicate event frames in pipeline output for clip "
                f"{self.clip_id!r}: {dupes}"
            )

    def by_frame(self) -> dict[int, EventRecord]:
        return {e.frame: e for e in self.events}

    def to_dict(self) -> dict[str, object]:
        return {
            "clip_id": self.clip_id,
            "fps": self.fps,
            "known_player_ids": sorted(self.known_player_ids),
            "events": [e.as_row() for e in self.events],
        }

    @staticmethod
    def from_dict(d: dict[str, object]) -> "PipelineOutputEvents":
        events = d.get("events", ())
        known_player_ids = d.get("known_player_ids", ())
        return PipelineOutputEvents(
            clip_id=str(d["clip_id"]),
            events=tuple(EventRecord.from_dict(e) for e in events),
            known_player_ids=frozenset(int(p) for p in known_player_ids),
            fps=float(d["fps"]) if d.get("fps") is not None else None,
        )

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @staticmethod
    def from_json(path: str | Path) -> "PipelineOutputEvents":
        return PipelineOutputEvents.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
