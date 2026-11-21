"""Narrow end-to-end benchmark for the Atom Zero assistant.

    python research/benchmark_assistant.py \
        --model checkpoints/transformer_behavior_50m_long \
        --adapter checkpoints/adapters_50m/v0001 \
        --out research/results/assistant_benchmark_adapter.json

This is a product-behavior benchmark, not MMLU or a general-reasoning score.
It checks tools, memory, tasks, source-cited retrieval, teacher-note review,
missing-source detection and a short generation.
"""
from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import torch

from streamft.agent_runtime import AgentSession
from streamft.doc_memory import DocMemory
from streamft.knowledge_gaps import KnowledgeGapStore

try:
    from research.chat_human import load_model
except ImportError:  # running as `python research/benchmark_assistant.py`
    from chat_human import load_model

ROOT = Path(__file__).resolve().parents[1]


def _alpha(text: str) -> float:
    return sum(ch.isalpha() for ch in text) / max(len(text), 1)


def _case(name: str, fn) -> dict:
    try:
        passed, detail = fn()
    except Exception as exc:  # a benchmark must report, not crash
        passed, detail = False, f"{type(exc).__name__}: {exc}"
    return {"name": name, "passed": bool(passed), "detail": str(detail)}


def run(model_path: str, tokenizer: str, adapter: str = "") -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tok = load_model(model_path, device, "transformer", tokenizer,
                            adapter)
    torch.manual_seed(0)
    with tempfile.TemporaryDirectory(prefix="atomzero-bench-") as folder:
        root = Path(folder)
        docs = DocMemory(root / "docs.db")
        gaps = KnowledgeGapStore(root / "gaps.json")
        docs.add_text(
            "https://example.test/policy",
            "The launch code is ORANGE-42. The backup room is Room B-12.",
            title="Launch policy")
        docs.add_text("warranty.txt", "The warranty code is W-7781.",
                      title="Warranty")
        docs.add_text("support.txt",
                      "The support email is help@example.test.",
                      title="Support")
        docs.add_text("office.txt", "The office room is Room 5C.",
                      title="Office")
        docs.add_text("security.txt", "The visitor badge is GREEN-7.",
                      title="Security")
        docs.add_text(
            "teacher://bench/teacher",
            "Topic: Teacher code\n\nThe teacher code is T-9000.",
            title="Teacher note: Teacher code",
            meta={"kind": "teacher", "source_url": "https://example.com/chat",
                  "teacher": "chatgpt-web", "model": "gpt-5",
                  "verified": False})
        agent = AgentSession("bench", root / "memory.json", model=model,
                             tokenizer=tok, device=device, docs=docs,
                             gap_store=gaps)
        tasks = []

        def greeting():
            reply = agent.chat("hello")["reply"]
            return "Hello" in reply, reply

        def help_text():
            reply = agent.chat("help")["reply"]
            return ("not a general reasoning" in reply
                    and "tasks" in reply), reply

        def math_task():
            result = agent.chat("what is 12 * 7?")
            return (result["reply"] == "12 * 7 = 84"
                    and result["tool_calls"][0]["name"] == "calculator"), (
                        result["reply"])

        def time_task():
            reply = agent.chat("what time is it")["reply"]
            return reply.startswith("It is "), reply

        def remember_name():
            reply = agent.chat("my name is Alex")["reply"]
            return "remember" in reply.lower(), reply

        def recall_name():
            reply = agent.chat("what is my name?")["reply"]
            return "Alex" in reply, reply

        def add_task():
            reply = agent.chat("add task buy milk")["reply"]
            return "Added task 1" in reply and "buy milk" in reply, reply

        def list_task():
            reply = agent.chat("list tasks")["reply"]
            return "buy milk" in reply, reply

        def complete_task():
            reply = agent.chat("complete task 1")["reply"]
            return "Completed task 1" in reply, reply

        def tasks_after_complete():
            reply = agent.chat("list tasks")["reply"]
            return "no open tasks" in reply.lower(), reply

        def remember_likes():
            reply = agent.chat("remember that I like green tea")["reply"]
            return "remember" in reply.lower(), reply

        def recall_likes():
            reply = agent.chat("what do I like?")["reply"]
            return "green tea" in reply.lower(), reply

        def grounded(query: str, keyword: str, source: str):
            result = agent.chat(query)
            reply = result["reply"]
            sources = result.get("sources") or []
            return (keyword.lower() in reply.lower()
                    and any(item.get("source") == source
                            for item in sources)), {
                        "reply": reply[:300],
                        "sources": [item.get("source") for item in sources],
                    }

        def retrieval():
            return grounded("what is the launch code?", "ORANGE-42",
                            "https://example.test/policy")

        def warranty_retrieval():
            return grounded("what is the warranty code?", "W-7781",
                            "warranty.txt")

        def support_retrieval():
            return grounded("what is the support email?",
                            "help@example.test", "support.txt")

        def office_retrieval():
            return grounded("what is the office room?", "5C", "office.txt")

        def badge_retrieval():
            return grounded("what is the visitor badge?", "GREEN-7",
                            "security.txt")

        def unverified_teacher():
            reply = agent.chat("what is the teacher code?")["reply"]
            return ("T-9000" in reply and "unverified" in reply
                    and "https://example.com/chat" in reply), reply[:300]

        def verified_teacher():
            docs.verify_source("teacher://bench/teacher", "bench-reviewer")
            reply = agent.chat("what is the teacher code?")["reply"]
            return ("verified by bench-reviewer" in reply
                    and "unverified teacher-derived note" not in reply), (
                        reply[:300])

        def missing_source_gap():
            result = agent.chat("what is the population of Paris?")
            reply = result["reply"]
            items = gaps.list()
            return ("do not have a source" in reply
                    and any(item["query"] == "what is the population of Paris?"
                            for item in items)), {
                        "reply": reply[:200],
                        "gaps": [item["query"] for item in items],
                    }

        def short_generation():
            reply = agent.chat("tell me a short story")["reply"]
            return (len(reply) >= 20 and _alpha(reply) >= 0.5
                    and "<|" not in reply), reply[:300]

        for name, fn in [
            ("greeting", greeting),
            ("help", help_text),
            ("math", math_task),
            ("time", time_task),
            ("remember_name", remember_name),
            ("recall_name", recall_name),
            ("add_task", add_task),
            ("list_task", list_task),
            ("complete_task", complete_task),
            ("tasks_after_complete", tasks_after_complete),
            ("remember_likes", remember_likes),
            ("recall_likes", recall_likes),
            ("retrieval_citation", retrieval),
            ("warranty_retrieval", warranty_retrieval),
            ("support_retrieval", support_retrieval),
            ("office_retrieval", office_retrieval),
            ("badge_retrieval", badge_retrieval),
            ("teacher_unverified", unverified_teacher),
            ("teacher_verified", verified_teacher),
            ("missing_source_gap", missing_source_gap),
            ("short_generation", short_generation),
        ]:
            tasks.append(_case(name, fn))
        docs.close()
    passed = sum(task["passed"] for task in tasks)
    return {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": model_path,
        "adapter": adapter,
        "score": passed / max(len(tasks), 1),
        "passed": passed,
        "total": len(tasks),
        "tasks": tasks,
        "note": ("Narrow product-behavior benchmark (tools, memory, tasks, RAG, "
                 "gap detection, one generation). Not MMLU or a general "
                 "reasoning/coding score."),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model", default="checkpoints/transformer_behavior_50m_long")
    ap.add_argument("--tokenizer", default="data/chat_en_bpe8k.json")
    ap.add_argument("--adapter", default="")
    ap.add_argument(
        "--out", default="research/results/assistant_benchmark.json")
    args = ap.parse_args(argv)
    payload = run(args.model, args.tokenizer, args.adapter)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    for task in payload["tasks"]:
        print(("PASS " if task["passed"] else "FAIL ") + task["name"])
    print(f"score: {payload['passed']}/{payload['total']} "
          f"({payload['score'] * 100:.1f}%)")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
