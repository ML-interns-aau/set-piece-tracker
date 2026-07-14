from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class CorrectionError(Exception):
    pass

class CorrectionSchemaError(CorrectionError):
    pass

class UnknownActionError(CorrectionSchemaError):
    pass

class DuplicateCorrectionError(CorrectionSchemaError):
    pass

class CorrectionApplicationError(CorrectionError):
    pass

class MissingFrameReferenceError(CorrectionApplicationError):
    pass

class InvalidPlayerIdError(CorrectionApplicationError):
    pass

class CorrectionConflictError(CorrectionApplicationError):
    pass

class CorrectionAction(str, Enum):
    CHANGE_EVENT = "change_event"
    CHANGE_PLAYER = "change_player"
    DELETE_EVENT = "delete_event"
    ADD_EVENT = "add_event"


_ALLOWED_FIELDS: dict[CorrectionAction, frozenset[str]] = {
    CorrectionAction.CHANGE_EVENT: frozenset({"old", "new"}),
    CorrectionAction.CHANGE_PLAYER: frozenset({"old_player", "new_player"}),
    CorrectionAction.DELETE_EVENT: frozenset(),
    CorrectionAction.ADD_EVENT: frozenset({"event"}),
}
_OPTIONAL_FIELD_NAMES = ("old", "new", "old_player", "new_player", "event")


@dataclass(frozen=True)
class Correction:
    frame: int
    action: CorrectionAction
    old: str | None = None
    new: str | None = None
    old_player: int | None = None
    new_player: int | None = None
    event: str | None = None

    def __post_init__(self) -> None:
        allowed = _ALLOWED_FIELDS[self.action]
        provided = {
            name for name in _OPTIONAL_FIELD_NAMES if getattr(self, name) is not None
        }
        missing = allowed - provided
        extra = provided - allowed
        if missing:
            raise CorrectionSchemaError(
                f"{self.action.value} at frame {self.frame} is missing required "
                f"field(s): {sorted(missing)}"
            )
        if extra:
            raise CorrectionSchemaError(
                f"{self.action.value} at frame {self.frame} has disallowed "
                f"field(s): {sorted(extra)}"
            )

    def to_dict(self) -> dict[str, object]:
        row: dict[str, object] = {"frame": self.frame, "action": self.action.value}
        for name in _OPTIONAL_FIELD_NAMES:
            value = getattr(self, name)
            if value is not None:
                row[name] = value
        return row

    @staticmethod
    def from_dict(d: dict[str, object]) -> "Correction":
        raw_action = d.get("action")
        try:
            action = CorrectionAction(raw_action)
        except ValueError:
            raise UnknownActionError(f"unknown correction action: {raw_action!r}") from None
        old = d.get("old")
        new = d.get("new")
        event = d.get("event")
        return Correction(
            frame=int(d["frame"]),
            action=action,
            old=old if old is None or isinstance(old, str) else str(old),
            new=new if new is None or isinstance(new, str) else str(new),
            old_player=int(d["old_player"]) if d.get("old_player") is not None else None,
            new_player=int(d["new_player"]) if d.get("new_player") is not None else None,
            event=event if event is None or isinstance(event, str) else str(event),
        )


@dataclass(frozen=True)
class CorrectionFile:
    clip_id: str
    corrections: tuple[Correction, ...]

    def __post_init__(self) -> None:
        seen: set[tuple[int, CorrectionAction]] = set()
        dupes: list[tuple[int, CorrectionAction]] = []
        for c in self.corrections:
            key = (c.frame, c.action)
            if key in seen:
                dupes.append(key)
            seen.add(key)
        if dupes:
            raise DuplicateCorrectionError(
                f"duplicate correction(s) in clip {self.clip_id!r}: "
                f"{[(f, a.value) for f, a in dupes]}"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "clip_id": self.clip_id,
            "corrections": [c.to_dict() for c in self.corrections],
        }

    @staticmethod
    def from_dict(d: dict[str, object]) -> "CorrectionFile":
        corrections = d.get("corrections", ())
        return CorrectionFile(
            clip_id=str(d["clip_id"]),
            corrections=tuple(Correction.from_dict(c) for c in corrections),
        )

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @staticmethod
    def from_json(path: str | Path) -> "CorrectionFile":
        return CorrectionFile.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
