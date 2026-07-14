from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

from src.verification.correction_schema import (
    Correction,
    CorrectionAction,
    CorrectionApplicationError,
    CorrectionConflictError,
    CorrectionFile,
    CorrectionSchemaError,
    InvalidPlayerIdError,
    MissingFrameReferenceError,
)
from src.verification.events import EventRecord, EventType, PipelineOutputEvents


@dataclass(frozen=True)
class CorrectedPipelineOutput:
    clip_id: str
    events: tuple[EventRecord, ...]
    original_events: tuple[EventRecord, ...]
    applied_corrections: tuple[Correction, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "clip_id": self.clip_id,
            "events": [e.as_row() for e in self.events],
            "original_events": [e.as_row() for e in self.original_events],
            "applied_corrections": [c.to_dict() for c in self.applied_corrections],
        }


def _apply_change_event(record: EventRecord, correction: Correction) -> EventRecord:
    if correction.old != record.event_type.value:
        raise CorrectionConflictError(
            f"change_event at frame {correction.frame}: expected old={correction.old!r} "
            f"but current event_type is {record.event_type.value!r}"
        )
    try:
        new_type = EventType(correction.new)
    except ValueError:
        raise CorrectionSchemaError(
            f"change_event at frame {correction.frame}: unknown event type {correction.new!r}"
        ) from None
    return replace(record, event_type=new_type, corrected=True)


def _apply_change_player(
    record: EventRecord, correction: Correction, known_player_ids: frozenset[int]
) -> EventRecord:
    if correction.old_player not in record.player_ids:
        raise CorrectionConflictError(
            f"change_player at frame {correction.frame}: player "
            f"{correction.old_player} is not among current player_ids "
            f"{record.player_ids}"
        )
    if correction.new_player not in known_player_ids:
        raise InvalidPlayerIdError(
            f"change_player at frame {correction.frame}: new_player "
            f"{correction.new_player} is not in the clip's known player roster"
        )
    new_ids = tuple(
        correction.new_player if p == correction.old_player else p
        for p in record.player_ids
    )
    return replace(record, player_ids=new_ids, corrected=True)


def apply_corrections(
    pipeline_output: PipelineOutputEvents, correction_file: CorrectionFile
) -> CorrectedPipelineOutput:
    if correction_file.clip_id != pipeline_output.clip_id:
        raise CorrectionApplicationError(
            f"clip_id mismatch: pipeline output is {pipeline_output.clip_id!r}, "
            f"correction file is {correction_file.clip_id!r}"
        )

    by_frame = dict(pipeline_output.by_frame())

    for correction in correction_file.corrections:
        if correction.action is CorrectionAction.CHANGE_EVENT:
            record = by_frame.get(correction.frame)
            if record is None:
                raise MissingFrameReferenceError(
                    f"change_event references frame {correction.frame}, which has "
                    f"no event in clip {pipeline_output.clip_id!r}"
                )
            by_frame[correction.frame] = _apply_change_event(record, correction)

        elif correction.action is CorrectionAction.CHANGE_PLAYER:
            record = by_frame.get(correction.frame)
            if record is None:
                raise MissingFrameReferenceError(
                    f"change_player references frame {correction.frame}, which has "
                    f"no event in clip {pipeline_output.clip_id!r}"
                )
            by_frame[correction.frame] = _apply_change_player(
                record, correction, pipeline_output.known_player_ids
            )

        elif correction.action is CorrectionAction.DELETE_EVENT:
            if correction.frame not in by_frame:
                raise MissingFrameReferenceError(
                    f"delete_event references frame {correction.frame}, which has "
                    f"no event in clip {pipeline_output.clip_id!r}"
                )
            del by_frame[correction.frame]

        elif correction.action is CorrectionAction.ADD_EVENT:
            if correction.frame in by_frame:
                raise CorrectionApplicationError(
                    f"add_event references frame {correction.frame}, which already "
                    f"has an event in clip {pipeline_output.clip_id!r} — use "
                    f"change_event to modify it instead"
                )
            try:
                new_type = EventType(correction.event)
            except ValueError:
                raise CorrectionSchemaError(
                    f"add_event at frame {correction.frame}: unknown event type "
                    f"{correction.event!r}"
                ) from None
            by_frame[correction.frame] = EventRecord(
                frame=correction.frame, event_type=new_type, confidence=1.0,
                player_ids=(), team=None, corrected=True,
            )

        else:
            raise CorrectionSchemaError(f"unhandled correction action: {correction.action}")

    corrected_events = tuple(sorted(by_frame.values(), key=lambda e: e.frame))
    return CorrectedPipelineOutput(
        clip_id=pipeline_output.clip_id,
        events=corrected_events,
        original_events=pipeline_output.events,
        applied_corrections=correction_file.corrections,
    )


def write_corrected_output(corrected: CorrectedPipelineOutput, out_path: str | Path) -> None:
    Path(out_path).write_text(json.dumps(corrected.to_dict(), indent=2), encoding="utf-8")
