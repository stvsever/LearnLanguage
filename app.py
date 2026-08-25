"""LearnLanguage local web server.

Zero-framework HTTP server (Python stdlib) exposing a small JSON API over the
backend services, plus static file serving for the frontend. All learner state
lives in the browser (localStorage); the server only generates content and audio.
"""
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
from urllib.parse import parse_qs, unquote, urlparse

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from backend import config, content, tts  # noqa: E402
from backend.grammar import grammar_profile  # noqa: E402
from backend.languages import public_language_payload  # noqa: E402
from backend.llm import LLMUnavailable  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("learnlanguage")


def config_payload() -> dict:
    return {
        "appName": config.APP_NAME,
        "version": config.APP_VERSION,
        "provider": config.active_provider(),
        "model": config.active_model(),
        "modelChoices": config.MODEL_CHOICES,
        "defaultLanguage": config.DEFAULT_LANGUAGE,
        "languages": public_language_payload(),
        "levels": list(content.CEFR_LEVELS),
        "seedLanguages": sorted(
            p.name.split("_")[0] for p in config.SEED_DIR.glob("*_core.json")
        ),
    }


class TutorRequestHandler(BaseHTTPRequestHandler):
    server_version = f"LearnLanguage/{config.APP_VERSION}"

    def log_message(self, fmt: str, *args: object) -> None:
        logger.info("%s - %s", self.address_string(), fmt % args)

    # -- helpers -----------------------------------------------------------
    def read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, path: Path, cache: str) -> None:
        if not path.exists() or not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")
            return
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache)
        self.end_headers()
        self.wfile.write(body)

    # -- routes ------------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802 - stdlib naming
        path = unquote(urlparse(self.path).path)
        if path == "/" or path == "/index.html":
            self.send_file(config.STATIC_DIR / "index.html", "no-cache")
            return
        if path == "/api/config":
            self.send_json(config_payload())
            return
        if path == "/api/grammar":
            query = parse_qs(urlparse(self.path).query)
            self.send_json(grammar_profile(str((query.get("language") or [config.DEFAULT_LANGUAGE])[0])))
            return
        if path.startswith("/static/"):
            target = (config.STATIC_DIR / path.removeprefix("/static/")).resolve()
            if not str(target).startswith(str(config.STATIC_DIR.resolve())):
                self.send_error(HTTPStatus.FORBIDDEN, "Forbidden")
                return
            self.send_file(target, "no-cache")
            return
        if path.startswith("/audio/"):
            name = Path(path.removeprefix("/audio/")).name
            self.send_file(config.AUDIO_DIR / name, "public, max-age=604800, immutable")
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Route not found")

    def do_POST(self) -> None:  # noqa: N802 - stdlib naming
        path = urlparse(self.path).path
        try:
            payload = self.read_json()
            if path == "/api/lesson":
                if payload.get("seed"):
                    lang = str(payload.get("language") or config.DEFAULT_LANGUAGE)
                    seeded = content.seed_lesson(lang, 999)
                    if seeded is None:
                        self.send_json({"error": "no_seed", "detail": "No starter deck for this language."}, status=404)
                    else:
                        self.send_json(seeded)
                    return
                self.send_json(content.generate_lesson(
                    topic=str(payload.get("topic") or ""),
                    language_code=str(payload.get("language") or config.DEFAULT_LANGUAGE),
                    level=str(payload.get("level") or "A2"),
                    count=int(payload.get("count") or 12),
                    known_words=payload.get("knownWords") or None,
                    model=payload.get("model"),
                ))
                return
            if path == "/api/compose":
                self.send_json(content.generate_composition(
                    prompt=str(payload.get("prompt") or ""),
                    language_code=str(payload.get("language") or config.DEFAULT_LANGUAGE),
                    level=str(payload.get("level") or "A2"),
                    length=str(payload.get("length") or "medium"),
                    model=payload.get("model"),
                ))
                return
            if path == "/api/gloss":
                self.send_json(content.generate_gloss(
                    text=str(payload.get("text") or ""),
                    context=str(payload.get("context") or ""),
                    language_code=str(payload.get("language") or config.DEFAULT_LANGUAGE),
                    model=payload.get("model"),
                ))
                return
            if path == "/api/tts":
                self.send_json(tts.synthesize(
                    text=str(payload.get("text") or ""),
                    language_code=str(payload.get("language") or config.DEFAULT_LANGUAGE),
                    voice=payload.get("voice"),
                    rate=payload.get("rate"),
                ))
                return
            self.send_error(HTTPStatus.NOT_FOUND, "Route not found")
        except LLMUnavailable as exc:
            self.send_json({"error": "llm_unavailable", "detail": str(exc)}, status=503)
        except ValueError as exc:
            self.send_json({"error": "bad_request", "detail": str(exc)}, status=400)
        except Exception as exc:  # noqa: BLE001 - top-level request guard
            logger.exception("Request failed: %s", exc)
            self.send_json({"error": "server_error", "detail": str(exc)}, status=500)


def run(host: str, port: int, open_browser: bool) -> None:
    config.ensure_runtime_dirs()
    server = ThreadingHTTPServer((host, port), TutorRequestHandler)
    url = f"http://{host}:{port}"
    logger.info("LearnLanguage %s | provider=%s model=%s | %s",
                config.APP_VERSION, config.active_provider(), config.active_model(), url)
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
