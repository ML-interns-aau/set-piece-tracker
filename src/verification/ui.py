from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from src.verification.events import PipelineOutputEvents
from src.verification.logger import ReviewLogger, ReviewVerdict

STATIC_DIR = Path(__file__).resolve().parent / "static"
INDEX_HTML_PATH = STATIC_DIR / "index.html"

_ALLOWED_STATIC: dict[str, tuple[str, Path]] = {
    "app.js": ("application/javascript", STATIC_DIR / "app.js"),
    "style.css": ("text/css", STATIC_DIR / "style.css"),
}


@dataclass(frozen=True)
class ClipPaths:
    original: Path
    overlay: Path
    events_json: Path


@dataclass(frozen=True)
class VerificationRegistry:
    clips: dict[str, ClipPaths]

    def load_events(self, clip_id: str) -> PipelineOutputEvents:
        return PipelineOutputEvents.from_json(self.clips[clip_id].events_json)


def build_events_payload(
    registry: VerificationRegistry, logger: ReviewLogger, clip_id: str
) -> list[dict[str, object]]:
    output = registry.load_events(clip_id)
    fps = output.fps or 25.0
    rows: list[dict[str, object]] = []
    for event in output.events:
        rows.append({
            "clip_id": clip_id,
            "frame": event.frame,
            "event_type": event.event_type.value,
            "confidence": event.confidence,
            "reliability_score": event.reliability_score,
            "team": event.team.value if event.team is not None else None,
            "player_ids": list(event.player_ids),
            "timestamp_s": event.frame / fps,
            "reviewed": logger.is_reviewed(clip_id, event.frame),
        })
    return rows


def parse_verdict_body(raw_bytes: bytes) -> ReviewVerdict:
    payload = json.loads(raw_bytes.decode("utf-8"))
    required = ("clip_id", "frame", "event", "verdict", "reviewer")
    missing = [k for k in required if k not in payload]
    if missing:
        raise ValueError(f"verdict payload missing field(s): {missing}")
    return ReviewVerdict(
        clip_id=str(payload["clip_id"]),
        frame=int(payload["frame"]),
        event=str(payload["event"]),
        verdict=str(payload["verdict"]),
        reviewer=str(payload["reviewer"]),
        timestamp=datetime.now().isoformat(),
    )


def _parse_range(range_header: str, size: int) -> tuple[int, int] | None:
    try:
        _, _, range_spec = range_header.partition("=")
        start_s, _, end_s = range_spec.partition("-")
        if start_s == "":
            length = int(end_s)
            start, end = max(0, size - length), size - 1
        else:
            start = int(start_s)
            end = int(end_s) if end_s else size - 1
        end = min(end, size - 1)
    except ValueError:
        return None
    if start > end or start < 0:
        return None
    return start, end


class VerificationServer(ThreadingHTTPServer):
    def __init__(
        self, address: tuple[str, int], registry: VerificationRegistry, logger: ReviewLogger
    ) -> None:
        super().__init__(address, VerificationRequestHandler)
        self.registry = registry
        self.logger = logger


class VerificationRequestHandler(BaseHTTPRequestHandler):
    server: VerificationServer

    def log_message(self, format: str, *args: object) -> None:
        pass

    def _send_bytes(self, data: bytes, content_type: str, status: int = HTTPStatus.OK) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, payload: object, status: int = HTTPStatus.OK) -> None:
        self._send_bytes(json.dumps(payload).encode("utf-8"), "application/json", status)

    def _send_file(self, path: Path, content_type: str) -> None:
        if not path.exists():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        data = path.read_bytes()
        range_header = self.headers.get("Range")
        if range_header and content_type.startswith("video/"):
            parsed_range = _parse_range(range_header, len(data))
            if parsed_range is None:
                self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                return
            start, end = parsed_range
            chunk = data[start:end + 1]
            self.send_response(HTTPStatus.PARTIAL_CONTENT)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Range", f"bytes {start}-{end}/{len(data)}")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(len(chunk)))
            self.end_headers()
            self.wfile.write(chunk)
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/":
            self._send_file(INDEX_HTML_PATH, "text/html")
            return

        if path.startswith("/static/"):
            entry = _ALLOWED_STATIC.get(path[len("/static/"):])
            if entry is None:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            content_type, file_path = entry
            self._send_file(file_path, content_type)
            return

        if path.startswith("/video/original/") or path.startswith("/video/overlay/"):
            is_original = path.startswith("/video/original/")
            prefix = "/video/original/" if is_original else "/video/overlay/"
            clip_id = path[len(prefix):]
            clip = self.server.registry.clips.get(clip_id)
            if clip is None:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self._send_file(clip.original if is_original else clip.overlay, "video/mp4")
            return

        if path == "/api/clips":
            self._send_json(sorted(self.server.registry.clips))
            return

        if path == "/api/events":
            clip_ids = parse_qs(parsed.query).get("clip_id")
            if not clip_ids or clip_ids[0] not in self.server.registry.clips:
                self.send_error(HTTPStatus.BAD_REQUEST)
                return
            payload = build_events_payload(self.server.registry, self.server.logger, clip_ids[0])
            self._send_json(payload)
            return

        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        if self.path != "/api/verdict":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        try:
            verdict = parse_verdict_body(raw)
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return
        if verdict.clip_id not in self.server.registry.clips:
            self._send_json({"error": f"unknown clip_id {verdict.clip_id!r}"},
                             status=HTTPStatus.NOT_FOUND)
            return
        self.server.logger.append(verdict)
        self._send_json(verdict.to_dict(), status=HTTPStatus.CREATED)


def run_server(
    registry: VerificationRegistry, logger: ReviewLogger,
    host: str = "127.0.0.1", port: int = 8765,
) -> VerificationServer:
    return VerificationServer((host, port), registry, logger)


def main() -> None:
    parser = argparse.ArgumentParser(description="Manual verification review server (FR-018)")
    parser.add_argument("--clip", required=True, help="clip_id to review")
    parser.add_argument("--original", required=True, type=Path, help="path to the original clip")
    parser.add_argument("--overlay", required=True, type=Path, help="path to the overlay video")
    parser.add_argument("--events", required=True, type=Path,
                         help="path to the PipelineOutputEvents JSON")
    parser.add_argument("--log", required=True, type=Path, help="path to the review JSONL log")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    registry = VerificationRegistry(clips={
        args.clip: ClipPaths(original=args.original, overlay=args.overlay,
                              events_json=args.events),
    })
    logger = ReviewLogger(args.log)
    server = run_server(registry, logger, args.host, args.port)
    print(f"Serving verification UI at http://{args.host}:{args.port}/ (clip={args.clip})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
