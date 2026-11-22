"""Curiosity proposals: turn unanswered questions into reviewed topics.

The assistant does not browse or ask a teacher on its own. It records a
knowledge gap when no local source matches. This module proposes those gaps as
**pending** research topics; a human must approve them before the worker fetches
sources or calls the teacher. Nothing here changes model weights.
"""
from __future__ import annotations

from atomzero.knowledge_gaps import GapNotFound, KnowledgeGapError
from atomzero.research_queue import ResearchQueueError


class CuriosityError(ValueError):
    pass


def propose_from_gaps(gaps, queue, limit: int = 3, lang: str = "en",
                      pages: int = 1, min_count: int = 1) -> dict:
    """Create pending research items from pending knowledge gaps.

    The operation is deliberately local: no network, teacher, or weight update.
    """
    if gaps is None or queue is None:
        raise CuriosityError("knowledge gaps and research queue are required")
    try:
        limit = max(1, min(int(limit), 10))
        min_count = max(1, int(min_count))
    except (TypeError, ValueError) as exc:
        raise CuriosityError("limit/min_count must be integers") from exc
    pending = [gap for gap in gaps.list()
               if gap.get("status") == "pending"
               and int(gap.get("count", 1)) >= min_count]
    pending.sort(key=lambda gap: (-int(gap.get("count", 1)),
                                  str(gap.get("first_seen", ""))))
    proposed = []
    skipped = []
    for gap in pending:
        if len(proposed) >= limit:
            break
        try:
            item = queue.add(gap["query"], lang, pages,
                             kind="curiosity", origin_id=gap["id"])
            gap = gaps.mark_proposed(gap["id"], item["id"])
        except (KnowledgeGapError, GapNotFound, ResearchQueueError,
                ValueError) as exc:
            skipped.append({"gap": gap.get("id", ""),
                            "query": gap.get("query", ""),
                            "error": str(exc)[:200]})
            continue
        proposed.append({"gap": gap, "research": item})
    return {
        "proposed": proposed,
        "skipped": skipped,
        "next": ("approve the pending research items, then run the worker; "
                 "approval is required before any fetch or teacher call"),
    }


def curiosity_items(queue, limit: int = 20) -> list[dict]:
    """Return research items created by the curiosity path."""
    if queue is None:
        return []
    items = [item for item in queue.list()
             if item.get("kind") == "curiosity"]
    return items[:max(1, int(limit))]
