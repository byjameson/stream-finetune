"""CLI runner for the reviewed research queue.

    python research/research_queue.py add "Ankara history" --pages 2
    python research/research_queue.py list
    python research/research_queue.py approve r0001
    python research/research_queue.py run --db data/docs.db --limit 1
    python research/research_queue.py run --db data/docs.db --limit 1 \
      --teacher-url http://127.0.0.1:11434/v1 --teacher-model llama3.2:3b

Only approved topics are fetched. Add ``--teacher-*`` to synthesize the
fetched Wikipedia context into an unverified teacher note. Schedule the ``run``
command with Task Scheduler/cron for reviewed autonomous research.
"""
from __future__ import annotations

import argparse
import json
import os

from atomzero.doc_memory import DocMemory
from atomzero.research_queue import ResearchQueue, ResearchQueueError
from atomzero.teacher_provider import TeacherProvider, TeacherProviderError


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
    run.add_argument("--teacher-url", default="")
    run.add_argument("--teacher-model", default="")
    run.add_argument("--teacher-label", default="openai-compatible")
    run.add_argument("--teacher-timeout", type=int, default=60)
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
            teacher = None
            if args.teacher_url or args.teacher_model:
                teacher = TeacherProvider(
                    args.teacher_url, args.teacher_model,
                    os.environ.get("TEACHER_API_KEY", ""),
                    args.teacher_timeout, args.teacher_label)
            docs = DocMemory(args.db)
            try:
                ran = queue.run_approved(
                    docs, args.limit, teacher=teacher)
            finally:
                docs.close()
            print(json.dumps({"ran": ran, "stats": queue.stats()},
                             indent=2, ensure_ascii=False))
    except (ResearchQueueError, TeacherProviderError, OSError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
