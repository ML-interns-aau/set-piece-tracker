from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ReviewVerdict:
    clip_id: str
    frame: int
    event: str
    verdict: str
    reviewer: str
    timestamp: str

    def to_dict(self) -> dict[str, object]:
        return {
            "clip_id": self.clip_id,
            "frame": self.frame,
            "event": self.event,
            "verdict": self.verdict,
            "reviewer": self.reviewer,
            "timestamp": self.timestamp,
        }

    @staticmethod
    def from_dict(d: dict[str, object]) -> "ReviewVerdict":
        return ReviewVerdict(
            clip_id=str(d["clip_id"]),
            frame=int(d["frame"]),
            event=str(d["event"]),
            verdict=str(d["verdict"]),
            reviewer=str(d["reviewer"]),
            timestamp=str(d["timestamp"]),
        )


class ReviewLogger:
    def __init__(self, log_path: str | Path) -> None:
        self.log_path = Path(log_path)

    def append(self, verdict: ReviewVerdict) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(verdict.to_dict()) + "\n")

    def load_all(self) -> tuple[ReviewVerdict, ...]:
        if not self.log_path.exists():
            return ()
        lines = self.log_path.read_text(encoding="utf-8").splitlines()
        return tuple(ReviewVerdict.from_dict(json.loads(line)) for line in lines if line.strip())

    def reviewed_keys(self) -> frozenset[tuple[str, int]]:
        return frozenset((v.clip_id, v.frame) for v in self.load_all())

    def is_reviewed(self, clip_id: str, frame: int) -> bool:
        return (clip_id, frame) in self.reviewed_keys()
