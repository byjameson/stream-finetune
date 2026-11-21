"""CLI runner for the reviewed research queue.

    python research/research_queue.py add "Ankara history" --pages 2
    python research/research_queue.py list
    python research/research_queue.py approve r0001
    python research/research_queue.py run --db data/docs.db --limit 1

Only approved topics are fetched. Schedule the ``run`` command with Task
Scheduler/cron for reviewed autonomous research.
"""
from __future__ import annotations

import argparse
import json

from streamft.doc_memory import DocMemory
from streamft.research_queue import ResearchQueue, ResearchQueueError


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", default="data/research_queue.json")
    sub = ap.add_subparsers(dest="cmd", required=True)
    add = sub.add_parser("add")
    add.add_argument("topic")
    add.add_argument("--lang", default="en")
    add.add_argument("--pages", type=int, default=2)
    sub.add_parser("list")
    approve = sub.add_parser("approve")
    approve.add_argument("id")
    reject = sub.add_parser("reject")
    reject.add_argument("id")
    run = sub.add_parser("run")
    run.add_argument("--db", default="data/docs.db")
    run.add_argument("--limit", type=int, default=1)
    args = ap.parse_args(argv)

    try:
        queue = ResearchQueue(args.queue)
        if args.cmd == "add":
            print(json.dumps(queue.add(args.topic, args.lang, args.pages),
                             ensure_ascii=False))
        elif args.cmd == "list":
            print(json.dumps({"data": queue.list(), "stats": queue.stats()},
                             indent=2, ensure_ascii=False))
        elif args.cmd == "approve":
            print(json.dumps(queue.approve(args.id), ensure_ascii=False))
        elif args.cmd == "reject":
            print(json.dumps(queue.reject(args.id), ensure_ascii=False))
        else:
            docs = DocMemory(args.db)
            try:
                ran = queue.run_approved(docs, args.limit)
            finally:
                docs.close()
            print(json.dumps({"ran": ran, "stats": queue.stats()},
                             indent=2, ensure_ascii=False))
    except (ResearchQueueError, OSError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
