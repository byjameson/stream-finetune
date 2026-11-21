"""End-to-end virtual-human evaluation: memory, tasks, open-ended chat."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from streamft.transformer import TransformerLM

try:
    from research.chat_human import ChatHuman, load_model
except ImportError:
    from chat_human import ChatHuman, load_model

ROOT = Path(__file__).resolve().parents[1]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TURNS = [
    "hello",
    "my name is Alex",
    "remember that I like green tea",
    "what is my name?",
    "what do I like?",
    "add task buy milk",
    "add task call mom",
    "list tasks",
    "complete task 1",
    "list tasks",
    "what do you remember?",
    "how are you?",
    "tell me a short story",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="checkpoints/transformer_chat_skill_21m")
    ap.add_argument("--arch", default="transformer")
    ap.add_argument("--adapter", default="")
    ap.add_argument("--out", default="research/results/chat_human_eval.json")
    args = ap.parse_args()
    memory = ROOT / "data/chat_human_eval_memory.json"
    if memory.exists():
        memory.unlink()
    model, tok = load_model(args.model, DEVICE, args.arch,
                            "data/chat_en_bpe8k.json", args.adapter)
    human = ChatHuman(memory, model, tok, DEVICE, args.arch)
    rows = []
    for text in TURNS:
        reply = human.respond(text)
        rows.append({"user": text, "assistant": reply})
    by_prompt = {row["user"]: row["assistant"] for row in rows}
    checks = {
        "greeting": by_prompt["hello"].lower().startswith(("hello", "hi")),
        "small_talk": ("well" in by_prompt["how are you?"].lower()
                       or "good" in by_prompt["how are you?"].lower()),
        "story": len(rows[12]["assistant"]) > 40,
        "name": by_prompt["what is my name?"] == "Your name is Alex.",
        "likes": "green tea" in by_prompt["what do I like?"],
        "task_add_1": "buy milk" in by_prompt["add task buy milk"],
        "task_add_2": "call mom" in by_prompt["add task call mom"],
        "task_list_both": ("buy milk" in rows[7]["assistant"]
                           and "call mom" in rows[7]["assistant"]),
        "task_complete": "Completed task 1" in by_prompt[
            "complete task 1"],
        "task_list_after": ("call mom" in rows[9]["assistant"]
                            and "buy milk" not in rows[9]["assistant"]),
        "remember": ("Alex" in by_prompt["what do you remember?"]
                     and "green tea" in by_prompt["what do you remember?"]),
    }
    open_rows = [rows[0], rows[11], rows[12]]
    open_ok = all(row["assistant"]
                  and sum(c.isalpha() for c in row["assistant"])
                  / max(len(row["assistant"]), 1) >= 0.5
                  and "<|" not in row["assistant"]
                  for row in open_rows)
    payload = {
        "model": args.model,
        "arch": args.arch,
        "checks": checks,
        "open_ended_ok": open_ok,
        "verdict": "PASS_VIRTUAL_HUMAN_SMOKE"
        if all(checks.values()) and open_ok else "BLOCKED",
        "turns": rows,
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
