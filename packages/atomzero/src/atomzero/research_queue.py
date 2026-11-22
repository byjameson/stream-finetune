"""Reviewed research queue: pending -> approved -> Wikipedia -> DocMemory.

Only approved topics are fetched. A local scheduler or the HTTP
``/v1/research/queue/run`` endpoint can process them without further user
interaction; rejected or pending items never touch the network.
"""
from __future__ import annotations

import copy
import json
import re
import threading
import time
from pathlib import Path

from atomzero.teacher_notes import add_teacher_note


class ResearchQueueError(ValueError):
    pass


class ItemNotFound(ResearchQueueError):
    pass


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _clean_topic(topic: str) -> str:
    topic = str(topic or "").strip()
    if not topic:
        raise ResearchQueueError("topic is required")
    if len(topic) > 200:
        raise ResearchQueueError("topic is too long (max 200 chars)")
    if any(ord(ch) < 32 for ch in topic):
        raise ResearchQueueError("topic contains control characters")
    return topic


def _clean_lang(lang: str) -> str:
    lang = str(lang or "en").strip().lower()
    if not re.fullmatch(r"[a-z]{2,10}(-[a-z]{2,10})?", lang):
        raise ResearchQueueError("invalid language code")
    return lang


def _clean_pages(pages) -> int:
    try:
        pages = int(pages)
    except (TypeError, ValueError) as exc:
        raise ResearchQueueError("pages must be an integer") from exc
    if not 1 <= pages <= 5:
        raise ResearchQueueError("pages must be between 1 and 5")
    return pages


def _page_row(page) -> dict:
    page = page if isinstance(page, dict) else {}
    try:
        chunks = int(page.get("chunks", 0))
    except (TypeError, ValueError):
        chunks = 0
    return {"source": str(page.get("source", "")),
            "title": str(page.get("title", "")),
            "chunks": chunks,
            "preview": str(page.get("preview", ""))[:200]}


def _teacher_prompt(topic: str, pages: list) -> str:
    blocks = []
    for index, page in enumerate(pages[:5], 1):
        page = page if isinstance(page, dict) else {}
        context = str(page.get("context") or page.get("preview") or "").strip()
        if not context:
            continue
        blocks.append(
            f"[{index}] {page.get('title', '')}\n"
            f"URL: {page.get('source', '')}\n{context[:4000]}")
    return (
        f"Write a concise factual synthesis about: {topic}\n"
        "Use only the sources below. Do not invent facts. If the sources "
        "disagree or are incomplete, say so.\n\n" + "\n\n".join(blocks))


class ResearchQueue:
    def __init__(self, path: str | Path = "data/research_queue.json"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.items = self._load()

    def _load(self) -> list[dict]:
        if not self.path.exists():
            return []
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ResearchQueueError(
                f"cannot read research queue: {exc}") from exc
        if not isinstance(data, list):
            raise ResearchQueueError("research queue must be a JSON list")
        return [item for item in data if isinstance(item, dict)]

    def _reload(self) -> None:
        self.items = self._load()

    def _save(self) -> None:
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(self.items, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8")
        tmp.replace(self.path)

    def _find(self, item_id: str) -> dict:
        item_id = str(item_id or "").strip()
        for item in self.items:
            if item.get("id") == item_id:
                return item
        raise ItemNotFound(f"research item not found: {item_id}")

    def _next_id(self) -> str:
        nums = []
        for item in self.items:
            match = re.fullmatch(r"r(\d+)", str(item.get("id", "")))
            if match:
                nums.append(int(match.group(1)))
        return f"r{max(nums, default=0) + 1:04d}"

    def add(self, topic: str, lang: str = "en", pages: int = 2,
            kind: str = "research", origin_id: str = "") -> dict:
        topic = _clean_topic(topic)
        lang = _clean_lang(lang)
        pages = _clean_pages(pages)
        kind = "curiosity" if str(kind).lower() == "curiosity" else "research"
        origin_id = str(origin_id or "")[:64]
        with self._lock:
            self._reload()
            for item in self.items:
                if (item.get("status") in ("pending", "approved", "running")
                        and str(item.get("topic", "")).lower() == topic.lower()):
                    raise ResearchQueueError("topic is already queued")
            item = {
                "id": self._next_id(),
                "topic": topic,
                "status": "pending",
                "lang": lang,
                "max_pages": pages,
                "kind": kind,
                "origin_id": origin_id,
                "created": _now(),
                "approved_at": "",
                "started_at": "",
                "finished_at": "",
                "sources": [],
                "error": "",
            }
            self.items.append(item)
            self._save()
            return copy.deepcopy(item)

    def approve(self, item_id: str) -> dict:
        with self._lock:
            self._reload()
            item = self._find(item_id)
            if item.get("status") not in ("pending", "approved"):
                raise ResearchQueueError(
                    f"cannot approve a {item.get('status')} item")
            item["status"] = "approved"
            item["approved_at"] = _now()
            item["error"] = ""
            self._save()
            return copy.deepcopy(item)

    def reject(self, item_id: str) -> dict:
        with self._lock:
            self._reload()
            item = self._find(item_id)
            if item.get("status") not in ("pending", "approved"):
                raise ResearchQueueError(
                    f"cannot reject a {item.get('status')} item")
            item["status"] = "rejected"
            item["finished_at"] = _now()
            self._save()
            return copy.deepcopy(item)

    def list(self) -> list[dict]:
        with self._lock:
            self._reload()
            return copy.deepcopy(self.items)

    def stats(self) -> dict:
        items = self.list()
        counts: dict[str, int] = {}
        for item in items:
            status = str(item.get("status", "unknown"))
            counts[status] = counts.get(status, 0) + 1
        return {"total": len(items), **counts}

    def recover_stale_running(self, stale_seconds: int = 21600) -> list[str]:
        """Return interrupted ``running`` items to ``approved`` for a retry."""
        cutoff = time.time() - max(60, int(stale_seconds))
        recovered: list[str] = []
        with self._lock:
            self._reload()
            for item in self.items:
                if item.get("status") != "running":
                    continue
                try:
                    started = time.mktime(time.strptime(
                        str(item.get("started_at", "")),
                        "%Y-%m-%dT%H:%M:%S"))
                except (TypeError, ValueError, OverflowError):
                    started = 0.0
                if started <= cutoff:
                    item["status"] = "approved"
                    item["started_at"] = ""
                    item["error"] = "recovered stale running item"
                    recovered.append(str(item.get("id", "")))
            if recovered:
                self._save()
        return recovered

    def run_approved(self, docs, limit: int = 1, fetcher=None,
                     teacher=None) -> list[dict]:
        """Fetch approved topics into DocMemory; pending items never run.

        If ``teacher`` is supplied, its synthesis of the fetched Wikipedia
        context is stored as an unverified teacher note.
        """
        if docs is None:
            raise ResearchQueueError("no document memory configured")
        if fetcher is None:
            from atomzero.web_research import wikipedia_research
            fetcher = wikipedia_research
        try:
            limit = max(1, min(int(limit), 5))
        except (TypeError, ValueError) as exc:
            raise ResearchQueueError("limit must be an integer") from exc

        selected: list[str] = []
        with self._lock:
            self._reload()
            for item in self.items:
                if (item.get("status") == "approved"
                        and len(selected) < limit):
                    item["status"] = "running"
                    item["started_at"] = _now()
                    item["error"] = ""
                    selected.append(str(item["id"]))
            self._save()

        results: list[dict] = []
        for item_id in selected:
            with self._lock:
                self._reload()
                item = self._find(item_id)
                topic = str(item.get("topic", ""))
                lang = str(item.get("lang", "en"))
                pages = int(item.get("max_pages", 2))
            try:
                found = fetcher(docs, topic, lang, pages)
            except Exception as exc:
                with self._lock:
                    self._reload()
                    item = self._find(item_id)
                    item["status"] = "error"
                    item["error"] = str(exc)[:500]
                    item["finished_at"] = _now()
                    item["sources"] = []
                    self._save()
                    results.append(copy.deepcopy(item))
            else:
                teacher_result = None
                if teacher is not None:
                    label = str(getattr(teacher, "label", "provider"))
                    model = str(getattr(teacher, "model", ""))
                    if not found:
                        teacher_result = {
                            "label": label, "model": model, "source": "",
                            "title": "", "verified": False,
                            "answer_preview": "",
                            "error": "no source pages to synthesize",
                        }
                    else:
                        try:
                            prompt = _teacher_prompt(topic, found)
                            answer = teacher.ask(
                                prompt,
                                system=("You are a concise research assistant. "
                                        "Use only the supplied sources. "
                                        "State uncertainty."),
                                max_tokens=512)
                            source_url = next(
                                (str(page.get("source", ""))
                                 for page in found
                                 if str(page.get("source", "")).startswith(
                                     ("http://", "https://"))), "")
                            note = add_teacher_note(
                                docs, label, model, topic, answer,
                                source_url)
                            teacher_result = {
                                "label": label, "model": model,
                                "source": note["source"],
                                "title": note["title"],
                                "verified": False,
                                "answer_preview": answer[:500],
                                "error": "",
                            }
                        except Exception as exc:
                            teacher_result = {
                                "label": label, "model": model, "source": "",
                                "title": "", "verified": False,
                                "answer_preview": "",
                                "error": str(exc)[:500],
                            }
                with self._lock:
                    self._reload()
                    item = self._find(item_id)
                    item["status"] = "done"
                    item["finished_at"] = _now()
                    item["error"] = ""
                    item["sources"] = [
                        _page_row(page) for page in (found or [])]
                    if teacher is not None:
                        item["teacher"] = teacher_result or {}
                    self._save()
                    results.append(copy.deepcopy(item))
        return results
