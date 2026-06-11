from __future__ import annotations

import argparse
import json
import logging
import mimetypes
import sys
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from backend.services import (  # noqa: E402
    AUDIO_DIR,
    STATIC_DIR,
    config_payload,
    ensure_runtime_dirs,
    generate_scenario,
    generate_vocabulary,
    load_vocabulary_topics,
    save_session,
    synthesize_speech,
    topic_vocabulary,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("learnlanguage")


class TutorRequestHandler(BaseHTTPRequestHandler):
    server_version = "LearnLanguage/1.0"

    def log_message(self, fmt: str, *args: object) -> None:
        logger.info("%s - %s", self.address_string(), fmt % args)

    def read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8"))

    def send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, path: Path) -> None:
        if not path.exists() or not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")
            return
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        try:
            is_static_asset = str(path.resolve()).startswith(str(STATIC_DIR.resolve()))
        except Exception:
            is_static_asset = False
        cache_header = "no-cache" if is_static_asset else "public, max-age=3600"
        self.send_header("Cache-Control", cache_header)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        if path == "/":
            self.send_file(STATIC_DIR / "index.html")
            return
        if path == "/api/config":
            self.send_json(config_payload())
            return
        if path == "/api/topics":
            self.send_json({"topics": load_vocabulary_topics()})
            return
        if path.startswith("/static/"):
            relative = Path(path.removeprefix("/static/"))
            target = (STATIC_DIR / relative).resolve()
            if not str(target).startswith(str(STATIC_DIR.resolve())):
                self.send_error(HTTPStatus.FORBIDDEN, "Forbidden")
                return
            self.send_file(target)
            return
        if path.startswith("/audio/"):
            relative = Path(path.removeprefix("/audio/")).name
            self.send_file(AUDIO_DIR / relative)
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Route not found")

    def do_POST(self) -> None:
        try:
            payload = self.read_json()
            parsed = urlparse(self.path)
            path = parsed.path
            if path == "/api/vocabulary/generate":
                self.send_json(
                    generate_vocabulary(
                        concept=str(payload.get("concept") or "daily conversation"),
                        count=int(payload.get("count") or 12),
                        language_code=str(payload.get("language") or "es"),
                        difficulty=str(payload.get("difficulty") or "intermediate"),
                    )
                )
                return
            if path == "/api/vocabulary/topic":
                self.send_json(
                    topic_vocabulary(
                        topic_id=str(payload.get("topicId") or "0-adjectives"),
                        language_code=str(payload.get("language") or "es"),
                        count=int(payload.get("count") or 20),
                        difficulty=str(payload.get("difficulty") or "intermediate"),
                    )
                )
                return
            if path == "/api/scenario/generate":
                self.send_json(
                    generate_scenario(
                        topic=str(payload.get("topic") or "a delayed train and a decision"),
                        language_code=str(payload.get("language") or "es"),
                        difficulty=str(payload.get("difficulty") or "intermediate"),
                    )
                )
                return
            if path == "/api/tts":
                self.send_json(
                    synthesize_speech(
                        text=str(payload.get("text") or ""),
                        language_code=str(payload.get("language") or "es"),
                        voice=payload.get("voice"),
                        rate=str(payload.get("rate") or "-10%"),
                    )
                )
                return
            if path == "/api/session/save":
                self.send_json(save_session(payload))
                return
            self.send_error(HTTPStatus.NOT_FOUND, "Route not found")
        except Exception as exc:
            logger.exception("Request failed: %s", exc)
            self.send_json({"error": "Request failed", "detail": str(exc)}, status=500)


def run(host: str, port: int, open_browser: bool) -> None:
    ensure_runtime_dirs()
    server = ThreadingHTTPServer((host, port), TutorRequestHandler)
    url = f"http://{host}:{port}"
    logger.info("LearnLanguage is running at %s", url)
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down LearnLanguage.")
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the LearnLanguage local web app.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open", action="store_true", help="Open the app in your default browser.")
    args = parser.parse_args()
    run(args.host, args.port, args.open)


if __name__ == "__main__":
    main()
