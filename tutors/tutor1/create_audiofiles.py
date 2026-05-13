#!/usr/bin/env python3
import asyncio
import json
import logging
import os
import shutil
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import edge_tts

# ── Settings ──────────────────────────────────────────────────────────────────
APP_DIR        = Path(__file__).resolve().parent
CONFIG_JSON    = APP_DIR / "data" / "vocabulary_es.json"
BASE_DIR       = APP_DIR / "audio" / "spanish"
SPANISH_VOICE  = "es-ES-AlvaroNeural"
MAX_WORKERS    = 20
MAX_RETRIES    = 1
BACKOFF_S      = 0.5     # seconds (doubles each retry)
PER_CALL_SLEEP = 0.05    # 50 ms after each success
FINAL_BACKOFF  = 2.0     # delay before last serial pass

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Globals ───────────────────────────────────────────────────────────────────
counter_lock = threading.Lock()
error_lock   = threading.Lock()
processed = saved = skipped = failed = 0
failed_global   = []          # [(dir, phrase), ...]
unique_errors   = set()       # {str(error), ...}

# ── Helpers ───────────────────────────────────────────────────────────────────
def ensure_dir(p): os.makedirs(p, exist_ok=True)
def load_topics(p): return json.load(open(p, encoding="utf-8")).get("topics", [])

def update_progress(status, path):
    global processed, saved, skipped, failed
    with counter_lock:
        processed += 1
        if   status.startswith("Saved"):  saved   += 1
        elif status == "Skipped":         skipped += 1
        elif status.startswith("Failed"): failed  += 1
        log.info(f"[{processed:>4}] {status:<15} {path}")

def clean_file(path):
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass

def record_error(exc: Exception):
    with error_lock:
        unique_errors.add(str(exc))

# ── Worker ────────────────────────────────────────────────────────────────────
def handle_phrase(topic_dir: str, phrase: str, record_fail: bool = True):
    out_path = os.path.join(topic_dir, f"{phrase}.mp3")

    # Skip if a valid file already exists
    if os.path.isfile(out_path) and os.path.getsize(out_path) > 0:
        update_progress("Skipped", out_path)
        return True

    delay = BACKOFF_S
    for attempt in range(1, MAX_RETRIES + 1):
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".tmp", dir=topic_dir)
        os.close(tmp_fd)

        try:
            async def _synth():
                communicate = edge_tts.Communicate(text=phrase, voice=SPANISH_VOICE, rate="-10%")
                await communicate.save(tmp_path)

            asyncio.run(_synth())
            shutil.move(tmp_path, out_path)  # atomic replace
            time.sleep(PER_CALL_SLEEP)
            tag = "Saved" if attempt == 1 else f"Saved (retry×{attempt-1})"
            update_progress(tag, out_path)
            return True
        except Exception as e:
            record_error(e)
            clean_file(tmp_path)
            clean_file(out_path)
            if attempt == MAX_RETRIES:
                update_progress(f"Failed ×{MAX_RETRIES}", out_path)
                if record_fail:
                    failed_global.append((topic_dir, phrase))
            else:
                time.sleep(delay)
                delay *= 2
    return False

# ── Pipeline ──────────────────────────────────────────────────────────────────
def main():
    ensure_dir(BASE_DIR)
    topics = load_topics(CONFIG_JSON)
    if not topics:
        log.error("No topics found.")
        return

    tasks = []
    for t in topics:
        dir_es = os.path.join(BASE_DIR, t["name"][1])
        ensure_dir(dir_es)
        for _, word in t["entries"]:
            tasks.append((dir_es, word))

    log.info(f"Total phrases to consider: {len(tasks)}")

    with ThreadPoolExecutor(MAX_WORKERS) as pool:
        futures = [pool.submit(handle_phrase, d, w) for d, w in tasks]
        for _ in as_completed(futures):
            pass  # progress logged inside worker

    # Final serial retry
    if failed_global:
        log.warning(f"Second pass for {len(failed_global)} failures …")
        time.sleep(FINAL_BACKOFF)
        retry_items = failed_global.copy()
        failed_global.clear()
        for d, w in retry_items:
            handle_phrase(d, w, record_fail=True)

    # Summary
    log.info("──────────── Summary ────────────")
    log.info(f"Processed : {processed}")
    log.info(f"Saved     : {saved}")
    log.info(f"Skipped   : {skipped}")
    log.info(f"Failed    : {failed}")

    if unique_errors:
        log.info("\nUnique error messages:")
        for err in unique_errors:
            log.info(f"  • {err}")

    log.info("All done!")

if __name__ == "__main__":
    main()
