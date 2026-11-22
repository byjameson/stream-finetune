"""Review knowledge gaps and turn them into pending research topics.

    python research/gap_loop.py list
    python research/gap_loop.py propose g0001
    python research/gap_loop.py dismiss g0002

A proposal only creates a *pending* research-queue item. It is not fetched
until a human approves it, and it never updates model weights.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from atomzero.knowledge_gaps import KnowledgeGapError, KnowledgeGapStore
from atomzero.research_queue import ResearchQueue, ResearchQueueError

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="data/knowledge_gaps.json")
    ap.add_argument("--queue", default="data/research_queue.json")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    propose = sub.add_parser("propose")
    propose.add_argument("id")
    propose.add_argument("--lang", default="en")
    propose.add_argument("--pages", type=int, default=1)
    dismiss = sub.add_parser("dismiss")
    dismiss.add_argument("id")
    args = ap.parse_args(argv)

    try:
        gaps = KnowledgeGapStore(ROOT / args.store)
        if args.cmd == "list":
            print(json.dumps({"data": gaps.list(), "stats": gaps.stats()},
                             indent=2, ensure_ascii=False))
        elif args.cmd == "dismiss":
            print(json.dumps(gaps.dismiss(args.id), ensure_ascii=False))
        else:
            gap = gaps.get(args.id)
            if gap.get("status") != "pending":
                raise KnowledgeGapError(
                    f"gap is already {gap.get('status')}")
            queue = ResearchQueue(ROOT / args.queue)
            item = queue.add(gap["query"], args.lang, args.pages)
            gap = gaps.mark_proposed(args.id, item["id"])
            print(json.dumps({
                "gap": gap,
                "research": item,
                "next": (f"approve {item['id']} in the research queue, then "
                         "run the scheduled worker"),
            }, indent=2, ensure_ascii=False))
    except (KnowledgeGapError, ResearchQueueError, OSError,
            json.JSONDecodeError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
