from __future__ import annotations

from collections.abc import Sequence

from src.domain.models import KeyMoments, PlayerPosition
from src.verification.events import EventRecord, EventType, PipelineOutputEvents


def events_from_key_moments(
    clip_id: str,
    key_moments: KeyMoments,
    positions: Sequence[PlayerPosition],
    known_player_ids: frozenset[int],
    fps: float | None = None,
) -> PipelineOutputEvents:
    events: list[EventRecord] = []

    def _mean_reliability(rows: Sequence[PlayerPosition]) -> float | None:
        if not rows:
            return None
        return sum(p.reliability_score for p in rows) / len(rows)

    kick_positions = tuple(p for p in positions if p.moment.value == "t_kick")
    events.append(
        EventRecord(
            frame=key_moments.t_kick_frame,
            event_type=EventType.KICK,
            confidence=1.0,
            player_ids=tuple(p.player_id for p in kick_positions),
            reliability_score=_mean_reliability(kick_positions),
        )
    )

    if key_moments.t_contact_frame is not None:
        contact_positions = tuple(p for p in positions if p.moment.value == "t_contact")
        events.append(
            EventRecord(
                frame=key_moments.t_contact_frame,
                event_type=EventType.CONTACT,
                confidence=1.0,
                player_ids=tuple(p.player_id for p in contact_positions),
                reliability_score=_mean_reliability(contact_positions),
            )
        )

    return PipelineOutputEvents(
        clip_id=clip_id, events=tuple(events),
        known_player_ids=known_player_ids, fps=fps,
    )


def apply_player_correction_to_positions(
    corrected_events: Sequence[EventRecord],
    positions: Sequence[PlayerPosition],
) -> tuple[PlayerPosition, ...]:
    raise NotImplementedError(
        "position-level correction bridge is not implemented — see this "
        "function's docstring for the open design question"
    )
