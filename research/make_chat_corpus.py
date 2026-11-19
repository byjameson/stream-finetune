"""Build a Turkish-first chat corpus for a from-scratch spiking LM.

Sources (all public, permissive licences):
  - OpenAssistant/oasst2 Turkish prompter -> assistant pairs (Apache-2.0)
  - merve/turkish_instructions (Apache-2.0)
  - TFLai/Turkish-Alpaca (Apache-2.0)
  - kilicai/turkish-sft-multi-turn-dialogue-10k (check card)
  - 3nesdeniz/turkish-daily-dialogues-5k (CC BY 4.0)
  - wikimedia/wikipedia 20231101.tr (CC BY-SA)
  - roneneldan/TinyStories (CDLA-Sharing) for English grammar support

The output is a packed int64 tensor plus a freshly trained 8k BPE tokenizer.
Run:
    python research/make_chat_corpus.py
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import torch
from datasets import load_dataset
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

ROOT = Path(__file__).resolve().parents[1]
SPECIALS = ["<|user|>", "<|assistant|>", "<|end|>"]
WIKI_CHARS = 30_000_000
STORY_CHARS = 8_000_000
CHAT_CHARS = 30_000_000
TOKENIZER_SAMPLE_CHARS = 20_000_000


def clean(text: str, max_len: int = 1500) -> str:
    if not isinstance(text, str):
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "".join(ch for ch in text if ch in "\n\t" or ord(ch) >= 32)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove quiz/hex artifacts seen in some SFT sets.
    text = re.sub(r"\b[0-9a-f]{8,}\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b\d{5,}\b", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_len]


def good(text: str, min_len: int = 2) -> str:
    text = clean(text)
    if len(text) < min_len:
        return ""
    letters = sum(c.isalpha() for c in text)
    if letters < max(2, int(len(text) * 0.3)):
        return ""
    return text


def chat_doc(turns) -> str:
    parts = []
    roles = []
    for role, text in turns:
        if role not in ("user", "assistant"):
            continue
        text = good(text)
        if not text:
            continue
        tag = "<|user|>" if role == "user" else "<|assistant|>"
        parts.append(f"{tag} {text}")
        roles.append(role)
    if "user" not in roles or "assistant" not in roles:
        return ""
    return " ".join(parts) + " <|end|>"


def text_doc(text: str) -> str:
    text = clean(text)
    return f"{text} <|end|>" if len(text) >= 80 else ""


def source_oasst():
    ds = load_dataset("OpenAssistant/oasst2", split="train", streaming=True)
    rows = list(ds)
    by_id = {r["message_id"]: r for r in rows}
    for r in rows:
        if r.get("lang") != "tr" or r.get("deleted"):
            continue
        if r.get("role") != "assistant":
            continue
        parent = by_id.get(r.get("parent_id"))
        if not parent or parent.get("deleted"):
            continue
        if parent.get("role") != "prompter":
            continue
        doc = chat_doc([("user", parent.get("text", "")),
                        ("assistant", r.get("text", ""))])
        if doc:
            yield doc, "oasst_tr"


def source_merve():
    ds = load_dataset("merve/turkish_instructions", split="train",
                      streaming=True)
    for row in ds:
        r = {str(k).strip(): v for k, v in row.items()}
        user = r.get("talimat") or ""
        extra = r.get("giriş") or r.get("giris") or ""
        out = r.get("çıktı") or r.get("cikti") or ""
        if extra:
            user = f"{user}\n{extra}"
        doc = chat_doc([("user", user), ("assistant", out)])
        if doc:
            yield doc, "merve_tr"


def source_alpaca():
    ds = load_dataset("TFLai/Turkish-Alpaca", split="train", streaming=True)
    for row in ds:
        user = row.get("instruction") or ""
        extra = row.get("input") or ""
        out = row.get("output") or ""
        if extra:
            user = f"{user}\n{extra}"
        doc = chat_doc([("user", user), ("assistant", out)])
        if doc:
            yield doc, "alpaca_tr"


def source_multi(name, source):
    try:
        ds = load_dataset(name, split="train", streaming=True)
    except Exception as exc:
        print(f"skip {name}: {type(exc).__name__}: {str(exc)[:120]}")
        return
    for row in ds:
        msgs = row.get("messages") or []
        turns = [(m.get("role"), m.get("content", "")) for m in msgs]
        doc = chat_doc(turns)
        if doc:
            yield doc, source


def source_wiki():
    ds = load_dataset("wikimedia/wikipedia", "20231101.tr", split="train",
                      streaming=True)
    chars = 0
    for row in ds:
        doc = text_doc(row.get("text", ""))
        if doc:
            yield doc, "wiki_tr"
            chars += len(doc)
            if chars >= WIKI_CHARS:
                break


def source_stories():
    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    chars = 0
    for row in ds:
        doc = text_doc(row.get("text", ""))
        if doc:
            yield doc, "stories_en"
            chars += len(doc)
            if chars >= STORY_CHARS:
                break


def collect(name, gen, docs, seen, char_cap):
    t0 = time.time()
    added = chars = 0
    for doc, source in gen:
        if chars >= char_cap:
            break
        key = hash(doc)
        if key in seen:
            continue
        seen.add(key)
        docs.append((doc, source))
        added += 1
        chars += len(doc)
    print(f"{name}: {added} docs, {chars/1e6:.1f}M chars, "
          f"{time.time()-t0:.0f}s", flush=True)


def main():
    docs: list[tuple[str, str]] = []
    seen: set[int] = set()
    collect("oasst_tr", source_oasst(), docs, seen, CHAT_CHARS)
    collect("merve_tr", source_merve(), docs, seen, CHAT_CHARS)
    collect("alpaca_tr", source_alpaca(), docs, seen, CHAT_CHARS)
    collect("kilicai_tr",
            source_multi("kilicai/turkish-sft-multi-turn-dialogue-10k",
                         "kilicai_tr"),
            docs, seen, CHAT_CHARS)
    collect("daily_tr",
            source_multi("3nesdeniz/turkish-daily-dialogues-5k", "daily_tr"),
            docs, seen, CHAT_CHARS)
    collect("wiki_tr", source_wiki(), docs, seen, WIKI_CHARS)
    collect("stories_en", source_stories(), docs, seen, STORY_CHARS)

    train, valid_chat, valid_wiki, valid_story = [], [], [], []
    for i, (doc, source) in enumerate(docs):
        is_valid = i % 50 == 0
        if source.startswith("wiki"):
            (valid_wiki if is_valid else train).append(doc)
        elif source.startswith("stories"):
            (valid_story if is_valid else train).append(doc)
        else:
            (valid_chat if is_valid else train).append(doc)

    import random
    random.Random(0).shuffle(train)
    print(f"train {len(train)} docs; valid chat {len(valid_chat)}, "
          f"wiki {len(valid_wiki)}, stories {len(valid_story)}", flush=True)

    # Train a fresh 8k BPE tokenizer on a balanced train sample.
    sample = []
    chars = 0
    for doc in train:
        sample.append(doc)
        chars += len(doc)
        if chars >= TOKENIZER_SAMPLE_CHARS:
            break
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=8192, special_tokens=SPECIALS, min_frequency=2,
        show_progress=True)
    tok.train_from_iterator(sample, trainer=trainer)
    tok_path = ROOT / "data/chat_tr_bpe8k.json"
    tok.save(str(tok_path))
    print(f"tokenizer vocab {tok.get_vocab_size()} -> {tok_path}", flush=True)

    end_id = tok.token_to_id("<|end|>")

    def pack(parts):
        ids = []
        for doc in parts:
            ids.extend(tok.encode(doc).ids)
        return torch.tensor(ids, dtype=torch.int64)

    outputs = {
        "data/chat_tr_train.pt": pack(train),
        "data/chat_tr_valid.pt": pack(valid_chat),
        "data/chat_tr_valid_mixed.pt": pack(valid_chat + valid_wiki
                                            + valid_story),
        "data/wiki_tr_valid.pt": pack(valid_wiki),
        "data/stories_en_valid.pt": pack(valid_story),
    }
    for rel, tensor in outputs.items():
        path = ROOT / rel
        torch.save(tensor, path)
        print(f"{rel}: {tensor.numel()} tokens", flush=True)

    meta = {
        "tokenizer": str(tok_path),
        "special_tokens": SPECIALS,
        "end_id": end_id,
        "sources": {
            "chat_tr": ["OpenAssistant/oasst2", "merve/turkish_instructions",
                        "TFLai/Turkish-Alpaca",
                        "kilicai/turkish-sft-multi-turn-dialogue-10k",
                        "3nesdeniz/turkish-daily-dialogues-5k"],
            "wiki_tr": "wikimedia/wikipedia 20231101.tr",
            "stories_en": "roneneldan/TinyStories",
        },
        "docs": {"train": len(train), "valid_chat": len(valid_chat),
                 "valid_wiki": len(valid_wiki),
                 "valid_story": len(valid_story)},
        "tokens": {rel: int(t.numel()) for rel, t in outputs.items()},
        "licences": {
            "oasst2": "Apache-2.0",
            "merve/turkish_instructions": "Apache-2.0",
            "TFLai/Turkish-Alpaca": "Apache-2.0",
            "3nesdeniz/turkish-daily-dialogues-5k": "CC BY 4.0",
            "wikipedia": "CC BY-SA",
            "TinyStories": "CDLA-Sharing",
        },
    }
    meta_path = ROOT / "data/chat_tr_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    (ROOT / "data/chat_tr_sample.txt").write_text(
        "\n\n".join(train[:20]) + "\n", encoding="utf-8")
    print(f"wrote {meta_path}", flush=True)


if __name__ == "__main__":
    main()
