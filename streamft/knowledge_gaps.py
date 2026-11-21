"""Knowledge-gap store for reviewed research proposals.

When the assistant has no local source for a question, the question is recorded
here. A human can turn a gap into a pending research-queue topic; the scheduled
worker fetches it only after approval. Gaps never train weights directly.
"""
from __future__ import annotations

import copy
import json
import re
import threading
import time
from pathlib import Path


class KnowledgeGapError(ValueError):
    pass


class GapNotFound(KnowledgeGapError):
    pass


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _clean_query(query: str) -> str:
    query = re.sub(r"\s+", " ", str(query or "")).strip()
    if not query:
        raise KnowledgeGapError("query is required")
    if len(query) > 500:
        raise KnowledgeGapError("query is too long (max 500 chars)")
    if "\x00" in query:
        raise KnowledgeGapError("query contains a NUL byte")
    return query


class KnowledgeGapStore:
    def __init__(self, path: str | Path = "data/knowledge_gaps.json"):
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
            raise KnowledgeGapError(
                f"cannot read knowledge gaps: {exc}") from exc
        if not isinstance(data, list):
            raise KnowledgeGapError("knowledge gaps must be a JSON list")
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
        raise GapNotFound(f"knowledge gap not found: {item_id}")

    def _next_id(self) -> str:
        nums = []
        for item in self.items:
            match = re.fullmatch(r"g(\d+)", str(item.get("id", "")))
            if match:
                nums.append(int(match.group(1)))
        return f"g{max(nums, default=0) + 1:04d}"

    def add(self, query: str, session_id: str = "",
            reason: str = "no_local_source") -> dict:
        query = _clean_query(query)
        session_id = str(session_id or "").strip()[:64]
        with self._lock:
            self._reload()
            normalized = query.lower()
            for item in self.items:
                if (item.get("status") in ("pending", "proposed")
                        and str(item.get("query", "")).lower() == normalized):
                    item["count"] = int(item.get("count", 1)) + 1
                    item["last_seen"] = _now()
                    if session_id:
                        item["session_id"] = session_id
                    self._save()
                    return copy.deepcopy(item)
            item = {
                "id": self._next_id(),
                "query": query,
                "status": "pending",
                "count": 1,
                "reason": str(reason or "no_local_source")[:100],
                "session_id": session_id,
                "research_id": "",
                "first_seen": _now(),
                "last_seen": _now(),
                "proposed_at": "",
                "dismissed_at": "",
            }
            self.items.append(item)
            self._save()
            return copy.deepcopy(item)

    def get(self, item_id: str) -> dict:
        with self._lock:
            self._reload()
            return copy.deepcopy(self._find(item_id))

    def list(self) -> list[dict]:
        with self._lock:
            self._reload()
            return copy.deepcopy(self.items)

    def stats(self) -> dict:
        counts: dict[str, int] = {}
        for item in self.list():
            status = str(item.get("status", "unknown"))
            counts[status] = counts.get(status, 0) + 1
        return {"total": sum(counts.values()), **counts}

    def mark_proposed(self, item_id: str, research_id: str) -> dict:
        with self._lock:
            self._reload()
            item = self._find(item_id)
            if item.get("status") != "pending":
                raise KnowledgeGapError(
                    f"cannot propose a {item.get('status')} gap")
            item["status"] = "proposed"
            item["research_id"] = str(research_id or "")
            item["proposed_at"] = _now()
            self._save()
            return copy.deepcopy(item)

    def dismiss(self, item_id: str) -> dict:
        with self._lock:
            self._reload()
            item = self._find(item_id)
            if item.get("status") not in ("pending", "proposed"):
                raise KnowledgeGapError(
                    f"cannot dismiss a {item.get('status')} gap")
            item["status"] = "dismissed"
            item["dismissed_at"] = _now()
            self._save()
            return copy.deepcopy(item)
