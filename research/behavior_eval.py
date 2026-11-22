"""Behavioral gate for a transformer chat checkpoint.

Checks the virtual-human memory/task commands plus held-out format skills.
Greedy decoding keeps the result deterministic.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from tokenizers import Tokenizer

from streamft.lora import apply_adapter
from atomzero.transformer import TransformerLM, generate

try:
    from research.chat_human import ChatHuman
except ImportError:
    from chat_human import ChatHuman

ROOT = Path(__file__).resolve().parents[1]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(path: str, adapter: str = ""):
    ck = torch.load(ROOT / path / "model.pt", map_location="cpu",
                    weights_only=False)
    cfg = ck["config"]
    model = TransformerLM(cfg["vocab"], cfg["dim"], cfg["layers"],
                          cfg["heads"], cfg["ff"]).to(DEVICE)
    model.load_state_dict(ck["state"])
    if adapter:
        apply_adapter(model, ROOT / adapter)
    model.eval()
    tok = Tokenizer.from_file(str(ROOT / "data/chat_en_bpe8k.json"))
    return model, tok


def ask(model, tok, prompt: str, max_new: int = 60) -> str:
    uid = tok.token_to_id("<|user|>")
    aid = tok.token_to_id("<|assistant|>")
    torch.manual_seed(7)
    text, _ = generate(
        model, tok, f"<|user|> {prompt} <|assistant|>",
        max_new=max_new, temperature=0.0, top_k=0,
        repetition_penalty=1.1, banned_ids=(uid, aid), device=DEVICE)
    return text.strip()


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def repeat3(text: str) -> float:
    words = re.findall(r"\w+", text.lower())
    tri = list(zip(words, words[1:], words[2:]))
    return 1 - len(set(tri)) / len(tri) if tri else 0.0


def behavior(model, tok, memory_path: Path) -> dict:
    if memory_path.exists():
        memory_path.unlink()
    human = ChatHuman(memory_path, model, tok, DEVICE, "transformer")
    turns = [
        "hello", "my name is Alex", "remember that I like green tea",
        "what is my name?", "what do I like?", "add task buy milk",
        "add task call mom", "list tasks", "complete task 1", "list tasks",
        "what do you remember?", "how are you?", "tell me a short story",
    ]
    rows = [{"user": text, "assistant": human.respond(text)} for text in turns]
    replies = {row["user"]: row["assistant"] for row in rows}
    story = rows[12]["assistant"]
    story_repeat = repeat3(story)
    format_prompts = {
        "count": "count from one to five",
        "brief": "be very brief: hello",
        "polite": "make it polite: open the window",
        "summarize": "summarize in one sentence: The cat sat on the mat.",
        "echo": "repeat after me: The book is on the desk.",
        "colors": "give me two colors",
    }
    generated = {name: human.respond(prompt)
                 for name, prompt in format_prompts.items()}
    raw_generated = {name: ask(model, tok, prompt)
                     for name, prompt in format_prompts.items()}
    checks = {
        "greeting": replies["hello"].lower().startswith(("hello", "hi")),
        "small_talk": "well" in replies["how are you?"].lower()
        or "good" in replies["how are you?"].lower(),
        "name": replies["what is my name?"] == "Your name is Alex.",
        "likes": "green tea" in replies["what do I like?"],
        "task_add": "buy milk" in replies["add task buy milk"],
        "task_list": ("buy milk" in rows[7]["assistant"]
                      and "call mom" in rows[7]["assistant"]),
        "task_complete": "Completed task 1" in replies["complete task 1"],
        "task_list_after": ("call mom" in rows[9]["assistant"]
                            and "buy milk" not in rows[9]["assistant"]),
        "remember": ("Alex" in replies["what do you remember?"]
                     and "green tea" in replies["what do you remember?"]),
        "story": len(story) > 60 and story.count(".") >= 2
        and story_repeat < 0.20,
        "count": norm(generated["count"]).strip(" .") == (
            "one, two, three, four, five"),
        "brief": len(generated["brief"]) <= 12
        and any(w in norm(generated["brief"]) for w in ("hi", "hello")),
        "polite": ("could" in norm(generated["polite"])
                   or "would" in norm(generated["polite"]))
        and "please" in norm(generated["polite"]),
        "summarize": ("cat" in norm(generated["summarize"])
                      and "mat" in norm(generated["summarize"])
                      and len(generated["summarize"]) < 80),
        "echo": "book is on the desk" in norm(generated["echo"]),
        "colors": sum(c in norm(generated["colors"]) for c in
                      ("red", "green", "blue", "yellow", "black",
                       "white", "orange")) >= 2,
    }
    return {"checks": checks, "passed": all(checks.values()),
            "replies": replies, "generated": generated,
            "raw_generated": raw_generated,
            "story_repeat3": story_repeat}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    model, tok = load_model(args.model, args.adapter)
    result = behavior(model, tok, ROOT / "data/behavior_eval_memory.json")
    payload = {"model": args.model, **result}
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
