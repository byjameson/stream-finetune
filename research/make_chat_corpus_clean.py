"""Build a cleaner dialogue-heavy English corpus for a 21M control run.

Same tokenizer and model size as the broad English run; the only change is the
data mix. Math/code-heavy messages are filtered out and a small set of
synthetic memory/task dialogues is added.

    python research/make_chat_corpus_clean.py
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_chat_corpus import SPECIALS, chat_doc, clean, text_doc  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TOKENIZER = ROOT / "data/chat_en_bpe8k.json"
PREFIX = "chat_en_clean"
SMOL_CHARS = 80_000_000
EVERYDAY_CHARS = 10_000_000
ULTRA_CHARS = 20_000_000
OASST_CHARS = 5_000_000
DOLLY_CHARS = 3_000_000
STORY_CHARS = 8_000_000
SYNTH_CHARS = 2_000_000

BAD = ("```", "$", "\\", "{", "}", "[", "]", "def ", "import ",
       "select ", "function(", "equation", "integral", "matrix",
       "theorem", "sqrt(", "python", "javascript", "sql", "http://",
       "https://")


def clean_turns(messages) -> str:
    turns = []
    for message in messages or []:
        role = message.get("role")
        if role not in ("user", "assistant"):
            continue
        text = clean(message.get("content", ""))
        if len(text) < 2:
            continue
        low = text.lower()
        if any(marker in low for marker in BAD):
            continue
        alpha = sum(c.isalpha() for c in text) / len(text)
        digits = sum(c.isdigit() for c in text) / len(text)
        if alpha < 0.55 or digits > 0.15:
            continue
        turns.append((role, text))
    if not any(role == "user" for role, _ in turns):
        return ""
    if not any(role == "assistant" for role, _ in turns):
        return ""
    return chat_doc(turns)


def source_smol_small():
    ds = load_dataset("HuggingFaceTB/smol-smoltalk", split="train",
                      streaming=True)
    for row in ds:
        doc = clean_turns(row.get("messages"))
        if doc:
            yield doc, "smol_small"


def source_everyday():
    ds = load_dataset("HuggingFaceTB/smoltalk", "everyday-conversations",
                      split="train", streaming=True)
    for row in ds:
        doc = clean_turns(row.get("messages"))
        if doc:
            yield doc, "everyday"


def source_ultrachat():
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft",
                      streaming=True)
    for row in ds:
        doc = clean_turns(row.get("messages"))
        if doc:
            yield doc, "ultrachat"


def source_oasst_en():
    ds = load_dataset("OpenAssistant/oasst2", split="train", streaming=True)
    rows = list(ds)
    by_id = {r["message_id"]: r for r in rows}
    for r in rows:
        if r.get("lang") != "en" or r.get("deleted"):
            continue
        if r.get("role") != "assistant":
            continue
        parent = by_id.get(r.get("parent_id"))
        if not parent or parent.get("deleted"):
            continue
        if parent.get("role") != "prompter":
            continue
        doc = clean_turns([
            {"role": "user", "content": parent.get("text", "")},
            {"role": "assistant", "content": r.get("text", "")},
        ])
        if doc:
            yield doc, "oasst"


def source_dolly():
    ds = load_dataset("databricks/databricks-dolly-15k", split="train",
                      streaming=True)
    for row in ds:
        user = row.get("instruction") or ""
        context = row.get("context") or ""
        if context:
            user = f"{user}\n\n{context}"
        doc = clean_turns([
            {"role": "user", "content": user},
            {"role": "assistant", "content": row.get("response", "")},
        ])
        if doc:
            yield doc, "dolly"


def source_stories():
    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    chars = 0
    for row in ds:
        doc = text_doc(row.get("text", ""))
        if doc:
            yield doc, "stories"
            chars += len(doc)
            if chars >= STORY_CHARS:
                break


def synthetic_dialogues():
    pairs = [
        ("hello", "Hello! How can I help you today?"),
        ("hi", "Hi there! What would you like to do?"),
        ("how are you?", "I am doing well, thank you for asking. How are you?"),
        ("what can you do?", "I can chat, remember facts, and keep a task list."),
        ("my name is Alex", "Nice to meet you, Alex. I will remember your name."),
        ("remember that I like green tea", "I will remember that."),
        ("add task buy milk", "Added task: buy milk."),
        ("list tasks", "You have one open task: buy milk."),
        ("complete task 1", "Completed task 1."),
        ("good morning", "Good morning! I hope you have a nice day."),
        ("good night", "Good night! Sleep well."),
        ("thank you", "You are welcome."),
        ("what is my name?", "Your name is Alex."),
        ("do you remember what I like?", "You like green tea."),
    ]
    people = ["Alex", "Sam", "Mia", "Leo", "Nora"]
    likes = ["green tea", "coffee", "books", "music", "walking"]
    tasks = ["buy milk", "call mom", "read a book", "go for a walk"]
    for name in people:
        for like in likes:
            for task in tasks:
                yield chat_doc([
                    ("user", f"my name is {name}"),
                    ("assistant", f"Nice to meet you, {name}."),
                    ("user", f"remember that I like {like}"),
                    ("assistant", "I will remember that."),
                    ("user", f"add task {task}"),
                    ("assistant", f"Added task: {task}."),
                ]), "synthetic"
    for prompt, reply in pairs:
        yield chat_doc([("user", prompt), ("assistant", reply)]), "synthetic"


def collect(name, gen, docs, seen, cap):
    chars = 0
    added = 0
    for doc, source in gen:
        if chars >= cap:
            break
        key = hash(doc)
        if key in seen:
            continue
        seen.add(key)
        docs.append((doc, source))
        chars += len(doc)
        added += 1
    print(f"{name}: {added} docs, {chars/1e6:.1f}M chars", flush=True)


def main():
    docs: list[tuple[str, str]] = []
    seen: set[int] = set()
    collect("smol_small", source_smol_small(), docs, seen, SMOL_CHARS)
    collect("everyday", source_everyday(), docs, seen, EVERYDAY_CHARS)
    collect("ultrachat", source_ultrachat(), docs, seen, ULTRA_CHARS)
    collect("oasst", source_oasst_en(), docs, seen, OASST_CHARS)
    collect("dolly", source_dolly(), docs, seen, DOLLY_CHARS)
    collect("synthetic", synthetic_dialogues(), docs, seen, SYNTH_CHARS)
    collect("stories", source_stories(), docs, seen, STORY_CHARS)

    train, valid_chat, valid_story = [], [], []
    for i, (doc, source) in enumerate(docs):
        is_valid = i % 50 == 0
        if source == "stories":
            (valid_story if is_valid else train).append(doc)
        else:
            (valid_chat if is_valid else train).append(doc)
    random.Random(0).shuffle(train)
    print(f"train {len(train)} docs; valid chat {len(valid_chat)}, "
          f"stories {len(valid_story)}", flush=True)

    tok = Tokenizer.from_file(str(TOKENIZER))
    end_id = tok.token_to_id("<|end|>")

    def pack(parts):
        ids = []
        for doc in parts:
            ids.extend(tok.encode(doc).ids)
        return torch.tensor(ids, dtype=torch.int64)

    outputs = {
        f"data/{PREFIX}_train.pt": pack(train),
        f"data/{PREFIX}_valid.pt": pack(valid_chat),
        f"data/{PREFIX}_valid_mixed.pt": pack(valid_chat + valid_story),
        f"data/{PREFIX}_stories_valid.pt": pack(valid_story),
    }
    for rel, tensor in outputs.items():
        torch.save(tensor, ROOT / rel)
        print(f"{rel}: {tensor.numel()} tokens", flush=True)

    meta = {
        "language": "en",
        "tokenizer": str(TOKENIZER),
        "special_tokens": SPECIALS,
        "end_id": end_id,
        "sources": ["HuggingFaceTB/smol-smoltalk",
                    "HuggingFaceTB/smoltalk/everyday-conversations",
                    "HuggingFaceH4/ultrachat_200k",
                    "OpenAssistant/oasst2", "databricks/dolly-15k",
                    "synthetic memory/task dialogues",
                    "roneneldan/TinyStories"],
        "docs": {"train": len(train), "valid_chat": len(valid_chat),
                 "valid_story": len(valid_story)},
        "tokens": {rel: int(t.numel()) for rel, t in outputs.items()},
        "filter": "no code/math markers; alpha>=0.55; digits<=0.15",
    }
    path = ROOT / f"data/{PREFIX}_meta.json"
    path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    (ROOT / f"data/{PREFIX}_sample.txt").write_text(
        "\n\n".join(train[:20]) + "\n", encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
