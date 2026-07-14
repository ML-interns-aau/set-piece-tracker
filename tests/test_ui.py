"""FR-018 - manual verification UI: HTTP route and JSON-logic tests only.

Client-side behavior (keyboard shortcuts, video lockstep, event navigation in
``static/app.js``) is not covered here — there is no Selenium/browser
automation in this suite. What's tested is the server-side contract the
frontend depends on: `build_events_payload`, `parse_verdict_body`, and two
smoke requests against a live `VerificationServer` on a background thread.
"""

from __future__ import annotations

import json
import threading
import urllib.request
from urllib.error import HTTPError

import pytest

from src.verification.events import EventRecord, EventType, PipelineOutputEvents
from src.verification.logger import ReviewLogger, ReviewVerdict
from src.verification.ui import (
    ClipPaths,
    VerificationRegistry,
    build_events_payload,
    parse_verdict_body,
    run_server,
)


def _registry(tmp_path, clip_id="clip_0007") -> VerificationRegistry:
    events = PipelineOutputEvents(
        clip_id=clip_id,
        events=(
            EventRecord(frame=100, event_type=EventType.KICK, confidence=1.0,
                        player_ids=(7,), reliability_score=0.9),
            EventRecord(frame=140, event_type=EventType.SHOT, confidence=0.8,
                        player_ids=(11,)),
        ),
        known_player_ids=frozenset({1, 7, 11}),
        fps=25.0,
    )
    events_path = tmp_path / f"{clip_id}_events.json"
    events.to_json(events_path)
    return VerificationRegistry(clips={
        clip_id: ClipPaths(original=tmp_path / "original.mp4",
                            overlay=tmp_path / "overlay.mp4", events_json=events_path),
    })


def test_build_events_payload_marks_reviewed_flag(tmp_path):
    registry = _registry(tmp_path)
    logger = ReviewLogger(tmp_path / "reviews.jsonl")
    logger.append(ReviewVerdict(clip_id="clip_0007", frame=100, event="kick",
                                 verdict="confirm", reviewer="manual",
                                 timestamp="2026-01-01T00:00:00"))

    rows = build_events_payload(registry, logger, "clip_0007")
    by_frame = {r["frame"]: r for r in rows}
    assert by_frame[100]["reviewed"] is True
    assert by_frame[140]["reviewed"] is False
    assert by_frame[100]["reliability_score"] == 0.9
    assert by_frame[100]["timestamp_s"] == pytest.approx(4.0)


def test_parse_verdict_body_accepts_literal_example_shape():
    body = json.dumps({
        "clip_id": "match001", "frame": 524, "event": "pass",
        "verdict": "verified", "reviewer": "manual",
    }).encode("utf-8")
    verdict = parse_verdict_body(body)
    assert verdict.clip_id == "match001"
    assert verdict.frame == 524
    assert verdict.event == "pass"
    assert verdict.verdict == "verified"
    assert verdict.reviewer == "manual"
    assert verdict.timestamp  # server-stamped, non-empty


def test_parse_verdict_body_rejects_missing_fields():
    body = json.dumps({"clip_id": "match001", "frame": 524}).encode("utf-8")
    with pytest.raises(ValueError):
        parse_verdict_body(body)


def _start_server(tmp_path):
    registry = _registry(tmp_path)
    logger = ReviewLogger(tmp_path / "reviews.jsonl")
    server = run_server(registry, logger, host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, logger


def test_server_smoke_get_index_returns_200(tmp_path):
    server, _ = _start_server(tmp_path)
    try:
        port = server.server_address[1]
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/") as resp:
            assert resp.status == 200
            assert b"Manual Verification" in resp.read()
    finally:
        server.shutdown()
        server.server_close()


def test_server_smoke_post_verdict_appends_to_log_and_get_events_reflects_it(tmp_path):
    server, logger = _start_server(tmp_path)
    try:
        port = server.server_address[1]
        payload = json.dumps({
            "clip_id": "clip_0007", "frame": 100, "event": "kick",
            "verdict": "confirm", "reviewer": "tester",
        }).encode("utf-8")
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/verdict", data=payload, method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 201

        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/api/events?clip_id=clip_0007"
        ) as resp:
            rows = json.loads(resp.read())
        by_frame = {r["frame"]: r for r in rows}
        assert by_frame[100]["reviewed"] is True
        assert logger.is_reviewed("clip_0007", 100)
    finally:
        server.shutdown()
        server.server_close()


def test_server_unknown_clip_verdict_returns_404(tmp_path):
    server, _ = _start_server(tmp_path)
    try:
        port = server.server_address[1]
        payload = json.dumps({
            "clip_id": "does_not_exist", "frame": 1, "event": "kick",
            "verdict": "confirm", "reviewer": "tester",
        }).encode("utf-8")
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/verdict", data=payload, method="POST",
            headers={"Content-Type": "application/json"},
        )
        with pytest.raises(HTTPError) as exc_info:
            urllib.request.urlopen(req)
        assert exc_info.value.code == 404
    finally:
        server.shutdown()
        server.server_close()
