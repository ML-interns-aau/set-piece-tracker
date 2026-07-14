"""FR-019 - correction-file schema, apply_corrections, and the KeyMoments bridge."""

from __future__ import annotations

import json

import pytest

from src.domain.models import KeyMoments, Moment, PlayerPosition, PositionSource, Team
from src.verification.bridge import (
    apply_player_correction_to_positions,
    events_from_key_moments,
)
from src.verification.correction import apply_corrections, write_corrected_output
from src.verification.correction_schema import (
    Correction,
    CorrectionAction,
    CorrectionApplicationError,
    CorrectionConflictError,
    CorrectionFile,
    CorrectionSchemaError,
    DuplicateCorrectionError,
    InvalidPlayerIdError,
    MissingFrameReferenceError,
    UnknownActionError,
)
from src.verification.events import EventRecord, EventType, PipelineOutputEvents


def _events(*records: EventRecord, clip_id: str = "clip_0007",
            known_player_ids=frozenset({1, 7, 11})) -> PipelineOutputEvents:
    return PipelineOutputEvents(clip_id=clip_id, events=records, known_player_ids=known_player_ids)


def _rec(frame: int, event_type: EventType = EventType.SHOT, confidence: float = 0.9,
         player_ids: tuple[int, ...] = (11,), team: Team | None = Team.ATTACKING) -> EventRecord:
    return EventRecord(frame=frame, event_type=event_type, confidence=confidence,
                        player_ids=player_ids, team=team)


# --- Correction schema validation --------------------------------------------

def test_change_event_requires_old_and_new():
    with pytest.raises(CorrectionSchemaError):
        Correction(frame=520, action=CorrectionAction.CHANGE_EVENT, old="shot")


def test_change_event_rejects_player_fields():
    with pytest.raises(CorrectionSchemaError):
        Correction(frame=520, action=CorrectionAction.CHANGE_EVENT, old="shot", new="pass",
                    old_player=11)


def test_add_event_requires_event_field():
    with pytest.raises(CorrectionSchemaError):
        Correction(frame=540, action=CorrectionAction.ADD_EVENT)


def test_delete_event_rejects_any_extra_field():
    with pytest.raises(CorrectionSchemaError):
        Correction(frame=530, action=CorrectionAction.DELETE_EVENT, old="shot")


def test_change_player_requires_old_and_new_player():
    with pytest.raises(CorrectionSchemaError):
        Correction(frame=521, action=CorrectionAction.CHANGE_PLAYER, old_player=11)


def test_unknown_action_string_raises():
    with pytest.raises(UnknownActionError):
        Correction.from_dict({"frame": 520, "action": "teleport_event"})


def test_correction_file_round_trips_literal_example_json():
    payload = {
        "clip_id": "match001",
        "corrections": [
            {"frame": 520, "action": "change_event", "old": "shot", "new": "pass"},
            {"frame": 521, "action": "change_player", "old_player": 11, "new_player": 7},
            {"frame": 530, "action": "delete_event"},
            {"frame": 540, "action": "add_event", "event": "header"},
        ],
    }
    corr_file = CorrectionFile.from_dict(payload)
    assert corr_file.clip_id == "match001"
    assert len(corr_file.corrections) == 4
    assert corr_file.to_dict() == payload


def test_duplicate_frame_and_action_raises():
    with pytest.raises(DuplicateCorrectionError):
        CorrectionFile(
            clip_id="match001",
            corrections=(
                Correction(frame=530, action=CorrectionAction.DELETE_EVENT),
                Correction(frame=530, action=CorrectionAction.DELETE_EVENT),
            ),
        )


def test_same_frame_different_actions_is_allowed():
    corr_file = CorrectionFile(
        clip_id="match001",
        corrections=(
            Correction(frame=521, action=CorrectionAction.CHANGE_EVENT, old="shot", new="pass"),
            Correction(frame=521, action=CorrectionAction.CHANGE_PLAYER,
                       old_player=11, new_player=7),
        ),
    )
    assert len(corr_file.corrections) == 2


# --- apply_corrections --------------------------------------------------------

def test_apply_change_event_updates_event_type_and_flags_corrected():
    output = _events(_rec(520, EventType.SHOT))
    corr_file = CorrectionFile(
        clip_id="clip_0007",
        corrections=(Correction(520, CorrectionAction.CHANGE_EVENT, old="shot", new="pass"),),
    )
    result = apply_corrections(output, corr_file)
    assert result.events[0].event_type is EventType.PASS
    assert result.events[0].corrected is True


def test_apply_change_event_conflicting_old_value_raises():
    output = _events(_rec(520, EventType.SHOT))
    corr_file = CorrectionFile(
        clip_id="clip_0007",
        corrections=(Correction(520, CorrectionAction.CHANGE_EVENT, old="cross", new="pass"),),
    )
    with pytest.raises(CorrectionConflictError):
        apply_corrections(output, corr_file)


def test_apply_change_player_validates_against_known_roster():
    output = _events(_rec(521, player_ids=(11,)))
    corr_file = CorrectionFile(
        clip_id="clip_0007",
        corrections=(Correction(521, CorrectionAction.CHANGE_PLAYER,
                                 old_player=11, new_player=7),),
    )
    result = apply_corrections(output, corr_file)
    assert result.events[0].player_ids == (7,)
    assert result.events[0].corrected is True


def test_apply_change_player_unknown_new_player_id_raises():
    output = _events(_rec(521, player_ids=(11,)))
    corr_file = CorrectionFile(
        clip_id="clip_0007",
        corrections=(Correction(521, CorrectionAction.CHANGE_PLAYER,
                                 old_player=11, new_player=99),),
    )
    with pytest.raises(InvalidPlayerIdError):
        apply_corrections(output, corr_file)


def test_apply_change_player_old_player_not_present_raises():
    output = _events(_rec(521, player_ids=(11,)))
    corr_file = CorrectionFile(
        clip_id="clip_0007",
        corrections=(Correction(521, CorrectionAction.CHANGE_PLAYER,
                                 old_player=1, new_player=7),),
    )
    with pytest.raises(CorrectionConflictError):
        apply_corrections(output, corr_file)


def test_apply_delete_event_removes_record():
    output = _events(_rec(530, EventType.SAVE))
    corr_file = CorrectionFile(
        clip_id="clip_0007",
        corrections=(Correction(530, CorrectionAction.DELETE_EVENT),),
    )
    result = apply_corrections(output, corr_file)
    assert result.events == ()


def test_apply_add_event_on_new_frame_inserts_record():
    output = _events(_rec(520, EventType.SHOT))
    corr_file = CorrectionFile(
        clip_id="clip_0007",
        corrections=(Correction(540, CorrectionAction.ADD_EVENT, event="header"),),
    )
    result = apply_corrections(output, corr_file)
    assert [e.frame for e in result.events] == [520, 540]
    added = result.events[1]
    assert added.event_type is EventType.HEADER
    assert added.corrected is True


def test_apply_add_event_on_existing_frame_raises():
    output = _events(_rec(520, EventType.SHOT))
    corr_file = CorrectionFile(
        clip_id="clip_0007",
        corrections=(Correction(520, CorrectionAction.ADD_EVENT, event="header"),),
    )
    with pytest.raises(CorrectionApplicationError):
        apply_corrections(output, corr_file)


@pytest.mark.parametrize("action, kwargs", [
    (CorrectionAction.CHANGE_EVENT, {"old": "shot", "new": "pass"}),
    (CorrectionAction.CHANGE_PLAYER, {"old_player": 11, "new_player": 7}),
    (CorrectionAction.DELETE_EVENT, {}),
])
def test_apply_missing_frame_reference_raises(action, kwargs):
    output = _events(_rec(520, EventType.SHOT))
    corr_file = CorrectionFile(clip_id="clip_0007",
                                corrections=(Correction(999, action, **kwargs),))
    with pytest.raises(MissingFrameReferenceError):
        apply_corrections(output, corr_file)


def test_apply_corrections_never_mutates_original_pipeline_output():
    output = _events(_rec(520, EventType.SHOT))
    corr_file = CorrectionFile(
        clip_id="clip_0007",
        corrections=(Correction(520, CorrectionAction.CHANGE_EVENT, old="shot", new="pass"),),
    )
    apply_corrections(output, corr_file)
    assert output.events[0].event_type is EventType.SHOT
    assert output.events[0].corrected is False


def test_apply_corrections_multiple_edits_across_all_actions():
    output = _events(
        _rec(520, EventType.SHOT, player_ids=(11,)),
        _rec(521, EventType.PASS, player_ids=(11,)),
        _rec(530, EventType.SAVE, player_ids=(1,)),
    )
    corr_file = CorrectionFile(
        clip_id="clip_0007",
        corrections=(
            Correction(520, CorrectionAction.CHANGE_EVENT, old="shot", new="pass"),
            Correction(521, CorrectionAction.CHANGE_PLAYER, old_player=11, new_player=7),
            Correction(530, CorrectionAction.DELETE_EVENT),
            Correction(540, CorrectionAction.ADD_EVENT, event="header"),
        ),
    )
    result = apply_corrections(output, corr_file)
    by_frame = result.events
    assert [e.frame for e in by_frame] == [520, 521, 540]
    assert by_frame[0].event_type is EventType.PASS
    assert by_frame[1].player_ids == (7,)


def test_write_corrected_output_writes_new_file(tmp_path):
    output = _events(_rec(520, EventType.SHOT))
    corr_file = CorrectionFile(
        clip_id="clip_0007",
        corrections=(Correction(520, CorrectionAction.CHANGE_EVENT, old="shot", new="pass"),),
    )
    result = apply_corrections(output, corr_file)
    out_path = tmp_path / "clip_0007_corrected.json"
    write_corrected_output(result, out_path)
    assert out_path.exists()
    reloaded = json.loads(out_path.read_text(encoding="utf-8"))
    assert reloaded["clip_id"] == "clip_0007"
    assert reloaded["events"][0]["event_type"] == "pass"
    assert reloaded["original_events"][0]["event_type"] == "shot"


# --- KeyMoments bridge --------------------------------------------------------

def test_events_from_key_moments_bridges_kick_and_contact():
    km = KeyMoments(t_kick_frame=100, t_contact_frame=110)
    positions = (
        PlayerPosition(clip_id="clip_0007", moment=Moment.T_KICK, player_id=7,
                        team=Team.ATTACKING, is_goalkeeper=False, pitch_x=5.0, pitch_y=30.0,
                        position_source=PositionSource.DETECTED, reliability_score=0.9),
        PlayerPosition(clip_id="clip_0007", moment=Moment.T_CONTACT, player_id=1,
                        team=Team.DEFENDING, is_goalkeeper=True, pitch_x=1.0, pitch_y=34.0,
                        position_source=PositionSource.DETECTED, reliability_score=0.8),
    )
    result = events_from_key_moments("clip_0007", km, positions, frozenset({1, 7}))
    assert [e.event_type for e in result.events] == [EventType.KICK, EventType.CONTACT]
    assert result.events[0].frame == 100
    assert result.events[0].player_ids == (7,)
    assert result.events[1].frame == 110
    assert result.events[1].player_ids == (1,)


def test_events_from_key_moments_omits_contact_when_none():
    km = KeyMoments(t_kick_frame=100, t_contact_frame=None)
    result = events_from_key_moments("clip_0007", km, (), frozenset())
    assert [e.event_type for e in result.events] == [EventType.KICK]


def test_position_correction_bridge_stub_raises_not_implemented():
    with pytest.raises(NotImplementedError):
        apply_player_correction_to_positions((), ())
