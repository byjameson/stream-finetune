"""Scheduled worker for the reviewed research queue.

    python research/research_worker.py --once
    python research/research_worker.py --loop --interval 3600 --limit 3

Only approved topics are fetched. The worker never creates topics and never
touches pending/rejected items. A lock prevents overlapping runs, interrupted
``running`` items are returned to ``approved`` after a stale timeout, and every
pass appends one JSON line to the run log. Keep the teacher provider optional;
Wikipedia research works without it.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time
from pathlib import Path

from atomzero.doc_memory import DocMemory
from atomzero.research_queue import ResearchQueue, ResearchQueueError
from atomzero.teacher_provider import TeacherProvider, TeacherProviderError

ROOT = Path(__file__).resolve().parents[1]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def acquire_lock(path: str | Path,
                 stale_seconds: int = 21600) -> str | None:
    """Create an exclusive lock; return its token, or None if already held."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            age = time.time() - path.stat().st_mtime
        except FileNotFoundError:
            age = 0.0
        try:
            info = json.loads(path.read_text(encoding="utf-8"))
            created = float(info.get("created", 0))
            if created:
                age = time.time() - created
        except (OSError, ValueError, TypeError):
            pass
        if age < max(60, int(stale_seconds)):
            return None
        try:
            path.unlink()
        except OSError:
            return None
    token = f"{os.getpid()}-{time.time_ns()}"
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        return None
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump({"pid": os.getpid(), "created": time.time(),
                   "host": os.environ.get("COMPUTERNAME", ""),
                   "token": token}, handle)
    return token


def release_lock(path: str | Path, token: str) -> None:
    path = Path(path)
    try:
        info = json.loads(path.read_text(encoding="utf-8"))
        if info.get("token") == token:
            path.unlink()
    except (OSError, ValueError, TypeError):
        pass


def append_log(path: str | Path, record: dict) -> None:
    if not path:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def item_summary(item: dict) -> dict:
    return {
        "id": item.get("id", ""),
        "topic": item.get("topic", ""),
        "status": item.get("status", ""),
        "sources": len(item.get("sources") or []),
        "error": str(item.get("error", ""))[:300],
    }


def run_once(args) -> dict:
    started = time.perf_counter()
    queue = ResearchQueue(args.queue)
    recovered = queue.recover_stale_running(args.stale_item_seconds)
    teacher = None
    if args.teacher_url or args.teacher_model:
        teacher = TeacherProvider(
            args.teacher_url, args.teacher_model,
            os.environ.get("TEACHER_API_KEY", ""),
            args.teacher_timeout, args.teacher_label)
    docs = DocMemory(args.db)
    try:
        ran = queue.run_approved(docs, args.limit, teacher=teacher)
        stats = queue.stats()
    finally:
        docs.close()
    return {
        "event": "research_run",
        "created": now(),
        "duration_s": round(time.perf_counter() - started, 2),
        "queue": str(args.queue),
        "db": str(args.db),
        "limit": args.limit,
        "teacher": bool(teacher),
        "recovered": recovered,
        "ran": [item_summary(item) for item in ran],
        "stats": stats,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", default="data/research_queue.json")
    ap.add_argument("--db", default="data/docs.db")
    ap.add_argument("--limit", type=int, default=1)
    ap.add_argument("--lock", default="data/research_worker.lock")
    ap.add_argument("--log", default="data/research_worker.jsonl")
    ap.add_argument("--stale-lock-seconds", type=int, default=21600)
    ap.add_argument("--stale-item-seconds", type=int, default=21600)
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--once", action="store_true",
                    help="run one pass (the default; kept for schedulers)")
    ap.add_argument("--interval", type=int, default=3600)
    ap.add_argument("--max-runs", type=int, default=0)
    ap.add_argument("--teacher-url", default="")
    ap.add_argument("--teacher-model", default="")
    ap.add_argument("--teacher-label", default="openai-compatible")
    ap.add_argument("--teacher-timeout", type=int, default=60)
    args = ap.parse_args(argv)

    runs = 0
    while True:
        token = acquire_lock(args.lock, args.stale_lock_seconds)
        if token is None:
            record = {"event": "research_locked", "created": now(),
                      "lock": str(args.lock)}
            append_log(args.log, record)
            print(json.dumps(record, ensure_ascii=False))
            return 0
        try:
            record = run_once(args)
            code = 0
        except (ResearchQueueError, TeacherProviderError, OSError,
                ValueError, sqlite3.Error) as exc:
            record = {"event": "research_error", "created": now(),
                      "error": str(exc)[:500]}
            code = 1
        finally:
            release_lock(args.lock, token)
        append_log(args.log, record)
        print(json.dumps(record, ensure_ascii=False))
        runs += 1
        if code != 0 or not args.loop:
            return code
        if args.max_runs and runs >= args.max_runs:
            return 0
        time.sleep(max(1, args.interval))


if __name__ == "__main__":
    raise SystemExit(main())
