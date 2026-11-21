"""Store manually supplied teacher answers as unverified RAG data.

This module is shared by the HTTP server and the reviewed research queue so both
paths use the same provenance, limits and `verified=false` marking.
"""
from __future__ import annotations

import hashlib
import re
import urllib.parse

MAX_TEACHER_BYTES = 200_000
MAX_TEACHER_TOPIC = 300
MAX_TEACHER_LABEL = 100
MAX_SOURCE_URL = 2048


class TeacherNoteTooLarge(ValueError):
    pass


def add_teacher_note(docs, teacher, model, topic, answer,
                     source_url: str = "") -> dict:
    """Validate and store a teacher answer with derived/unverified metadata."""
    if docs is None:
        raise ValueError("no document memory configured")
    teacher = str(teacher or "manual").strip() or "manual"
    model = str(model or "").strip()
    topic = str(topic or "").strip()
    answer = str(answer or "").strip()
    source_url = str(source_url or "").strip()
    for label, value, limit in (
            ("teacher", teacher, MAX_TEACHER_LABEL),
            ("model", model, MAX_TEACHER_LABEL),
            ("topic", topic, MAX_TEACHER_TOPIC)):
        if len(value) > limit:
            raise ValueError(f"{label} is too long (max {limit} chars)")
        if any(ord(ch) < 32 for ch in value):
            raise ValueError(f"{label} contains control characters")
    if not answer:
        raise ValueError("answer is required")
    if source_url:
        if len(source_url) > MAX_SOURCE_URL:
            raise ValueError(
                f"source_url is too long (max {MAX_SOURCE_URL} chars)")
        parsed = urllib.parse.urlparse(source_url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise ValueError("source_url must be an http(s) URL")
    text = f"Topic: {topic}\n\n{answer}" if topic else answer
    raw = text.encode("utf-8")
    if len(raw) > MAX_TEACHER_BYTES:
        raise TeacherNoteTooLarge(
            f"teacher note is too large (max {MAX_TEACHER_BYTES} bytes)")
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", teacher).strip("-")[:48]
    digest = hashlib.sha256(raw).hexdigest()
    source = f"teacher://{safe or 'manual'}/{digest[:16]}"
    title = (f"Teacher note: {topic}" if topic
             else f"Teacher note ({teacher})")
    chunks = docs.add_text(source, text, title=title, meta={
        "kind": "teacher",
        "source_url": source_url,
        "teacher": teacher,
        "model": model,
        "verified": False,
    })
    return {
        "source": source,
        "title": title,
        "chunks": chunks,
        "bytes": len(raw),
        "sha256": digest,
        "kind": "teacher",
        "verified": False,
        "source_url": source_url,
        "teacher": teacher,
        "model": model,
        "note": "teacher-derived and unverified",
    }
