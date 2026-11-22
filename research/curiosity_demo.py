"""Live curiosity loop: gap -> proposal -> approval -> teacher -> RAG.

    python research/curiosity_demo.py

Requires the local Ollama endpoint and network access for Wikipedia. Nothing
runs before approval and model weights are never changed.
"""
from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

from atomzero.curiosity import propose_from_gaps
from atomzero.doc_memory import DocMemory
from atomzero.knowledge_gaps import KnowledgeGapStore
from atomzero.research_queue import ResearchQueue
from atomzero.teacher_provider import TeacherProvider

try:
    from research.chat_human import ChatHuman
except ImportError:
    from chat_human import ChatHuman

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:11434/v1")
    ap.add_argument("--model", default="llama3.2:3b")
    ap.add_argument("--question",
                    default="What is the capital of New Zealand?")
    ap.add_argument("--out",
                    default="research/results/curiosity_live.json")
    args = ap.parse_args(argv)
    teacher = TeacherProvider(args.url, args.model, label="ollama")
    with tempfile.TemporaryDirectory(prefix="atomzero-curiosity-") as folder:
        root = Path(folder)
        docs = DocMemory(root / "docs.db")
        gaps = KnowledgeGapStore(root / "gaps.json")
        queue = ResearchQueue(root / "queue.json")
        try:
            gaps.add(args.question, reason="live_curiosity_demo")
            human = ChatHuman(root / "memory.json", docs=docs)
            before = human.respond(args.question)
            proposal = propose_from_gaps(gaps, queue, limit=1, pages=1)
            item = proposal["proposed"][0]["research"]
            unapproved_run = queue.run_approved(
                docs, limit=1, teacher=teacher)
            queue.approve(item["id"])
            ran = queue.run_approved(docs, limit=1, teacher=teacher)
            after = human.respond(args.question)
        finally:
            docs.close()
    payload = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "question": args.question,
        "before": before,
        "proposal": proposal,
        "unapproved_run": unapproved_run,
        "approved_run": ran,
        "after": after,
        "weights_changed": False,
        "note": ("Only an approved topic was fetched and sent to the teacher. "
                 "The result is an unverified RAG note, not a weight update."),
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    print("before:", before.replace("\n", " | ")[:160])
    print("unapproved_run:", len(unapproved_run), "item(s)")
    print("approved_run:", ran[0]["status"] if ran else "none")
    print("after:", after.replace("\n", " | ")[:220])
    print("weights_changed:", False)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
