"""Reviewed feedback store for the personal self-improvement loop.

Flow:
    user feedback -> pending review -> approved -> export training JSONL
    -> gated LoRA update -> accepted adapter (or rejected candidate)
"""
from __future__ import annotations

import copy
import json
import threading
import time
from pathlib import Path


class FeedbackError(ValueError):
    pass


class FeedbackNotFound(FeedbackError):
    pass


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _clean(text, label: str, limit: int) -> str:
    text = str(text or "").strip()
    if len(text) > limit:
        raise FeedbackError(f"{label} is too long (max {limit} chars)")
    if "\x00" in text:
        raise FeedbackError(f"{label} contains a NUL byte")
    return text


class FeedbackStore:
    def __init__(self, path: str | Path = "data/feedback_queue.json"):
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
            raise FeedbackError(f"cannot read feedback store: {exc}") from exc
        if not isinstance(data, list):
            raise FeedbackError("feedback store must be a JSON list")
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
        raise FeedbackNotFound(f"feedback item not found: {item_id}")

    def _next_id(self) -> str:
        nums = []
        for item in self.items:
            value = str(item.get("id", ""))
            if value.startswith("f") and value[1:].isdigit():
                nums.append(int(value[1:]))
        return f"f{max(nums, default=0) + 1:04d}"

    def add(self, user: str, assistant: str, rating: int = 0,
            corrected: str = "", session_id: str = "") -> dict:
        user = _clean(user, "user", 2000)
        assistant = _clean(assistant, "assistant", 8000)
        corrected = _clean(corrected, "corrected", 8000)
        session_id = _clean(session_id, "session_id", 64)
        if not user:
            raise FeedbackError("user text is required")
        if not assistant and not corrected:
            raise FeedbackError("assistant or corrected text is required")
        try:
            rating = int(rating)
        except (TypeError, ValueError) as exc:
            raise FeedbackError("rating must be an integer") from exc
        if rating not in (-1, 0, 1):
            raise FeedbackError("rating must be -1, 0 or 1")
        with self._lock:
            self._reload()
            for item in self.items:
                if (item.get("status") in ("pending", "approved")
                        and item.get("user") == user
                        and item.get("assistant") == assistant
                        and item.get("corrected", "") == corrected
                        and item.get("rating") == rating):
                    raise FeedbackError("feedback is already queued")
            item = {
                "id": self._next_id(),
                "user": user,
                "assistant": assistant,
                "corrected": corrected,
                "rating": rating,
                "session_id": session_id,
                "status": "pending",
                "created": _now(),
                "reviewed_at": "",
            }
            self.items.append(item)
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

    def approve(self, item_id: str) -> dict:
        with self._lock:
            self._reload()
            item = self._find(item_id)
            if item.get("status") != "pending":
                raise FeedbackError(
                    f"cannot approve a {item.get('status')} item")
            if (item.get("rating") == -1
                    and not str(item.get("corrected", "")).strip()):
                raise FeedbackError(
                    "negative feedback needs a corrected answer")
            item["status"] = "approved"
            item["reviewed_at"] = _now()
            self._save()
            return copy.deepcopy(item)

    def reject(self, item_id: str) -> dict:
        with self._lock:
            self._reload()
            item = self._find(item_id)
            if item.get("status") not in ("pending", "approved"):
                raise FeedbackError(
                    f"cannot reject a {item.get('status')} item")
            item["status"] = "rejected"
            item["reviewed_at"] = _now()
            self._save()
            return copy.deepcopy(item)

    def export_training(self, path: str | Path) -> dict:
        """Write approved feedback as user/assistant JSONL for the gate."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        seen = set()
        with self._lock:
            self._reload()
            for item in self.items:
                if item.get("status") != "approved":
                    continue
                assistant = str(item.get("corrected") or
                                item.get("assistant") or "").strip()
                user = str(item.get("user") or "").strip()
                if not user or not assistant:
                    continue
                key = (user, assistant)
                if key in seen:
                    continue
                seen.add(key)
                rows.append({"user": user, "assistant": assistant})
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n"
                    for row in rows),
            encoding="utf-8")
        tmp.replace(path)
        return {"path": str(path), "examples": len(rows)}
