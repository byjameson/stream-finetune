"""Demonstrate teacher -> RAG learning without weight updates.

    python research/teacher_learning_demo.py

Requires the local Ollama endpoint used by the teacher (llama3.2:3b by default).
This is not autonomous curiosity: the question is explicit, the answer is
stored as an unverified teacher note, and later answers retrieve that note.
"""
from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

from atomzero.doc_memory import DocMemory
from atomzero.teacher_notes import add_teacher_note
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
                    default="research/results/teacher_rag_learning.json")
    args = ap.parse_args(argv)
    teacher = TeacherProvider(args.url, args.model, label="ollama")
    with tempfile.TemporaryDirectory(prefix="atomzero-teacher-learn-") as folder:
        root = Path(folder)
        docs = DocMemory(root / "docs.db")
        try:
            human = ChatHuman(root / "memory.json", docs=docs)
            before = human.respond(args.question)
            answer = teacher.ask(
                f"Answer this question concisely and factually: {args.question}",
                system="You are a factual teacher. Give a short direct answer.",
                max_tokens=128)
            note = add_teacher_note(
                docs, "ollama", args.model, args.question, answer)
            after = human.respond(args.question)
        finally:
            docs.close()
    payload = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "question": args.question,
        "before": before,
        "teacher": {"model": args.model, "answer": answer},
        "note": note,
        "after": after,
        "weights_changed": False,
        "note_text": ("The answer was stored in RAG as an unverified teacher "
                      "note; no model weights were updated."),
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    print("before:", before.replace("\n", " | ")[:180])
    print("teacher:", answer.replace("\n", " | ")[:180])
    print("after:", after.replace("\n", " | ")[:220])
    print("weights_changed:", False)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
