"""Build an English-first chat corpus for the from-scratch spiking LM.

Sources:
  - HuggingFaceTB/smoltalk (Apache-2.0 for the new subsets)
  - HuggingFaceH4/ultrachat_200k (MIT)
  - OpenAssistant/oasst2 English pairs (Apache-2.0)
  - databricks/databricks-dolly-15k (CC BY-SA 3.0)
  - wikimedia/wikipedia 20231101.en (CC BY-SA)
  - roneneldan/TinyStories (CDLA-Sharing)

Run:
    python research/make_chat_corpus_en.py
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import torch
from datasets import load_dataset
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_chat_corpus import SPECIALS, chat_doc, text_doc  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "chat_en"
WIKI_CHARS = 10_000_000
STORY_CHARS = 8_000_000
SMOL_CHARS = 45_000_000
ULTRA_CHARS = 25_000_000
TOKENIZER_SAMPLE_CHARS = 20_000_000


def source_smoltalk():
    ds = load_dataset("HuggingFaceTB/smoltalk", "all", split="train",
                      streaming=True)
    for row in ds:
        msgs = row.get("messages") or []
        turns = [(m.get("role"), m.get("content", "")) for m in msgs]
        doc = chat_doc(turns)
        if doc:
            yield doc, "smoltalk_en"


def source_ultrachat():
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft",
                      streaming=True)
    for row in ds:
        msgs = row.get("messages") or []
        turns = [(m.get("role"), m.get("content", "")) for m in msgs]
        doc = chat_doc(turns)
        if doc:
            yield doc, "ultrachat_en"


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
        doc = chat_doc([("user", parent.get("text", "")),
                        ("assistant", r.get("text", ""))])
        if doc:
            yield doc, "oasst_en"


def source_dolly():
    ds = load_dataset("databricks/databricks-dolly-15k", split="train",
                      streaming=True)
    for row in ds:
        user = row.get("instruction") or ""
        context = row.get("context") or ""
        if context:
            user = f"{user}\n\n{context}"
        doc = chat_doc([("user", user), ("assistant", row.get("response", ""))])
        if doc:
            yield doc, "dolly_en"


def source_wiki_en():
    ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train",
                      streaming=True)
    chars = 0
    for row in ds:
        doc = text_doc(row.get("text", ""))
        if doc:
            yield doc, "wiki_en"
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
    collect("smoltalk_en", source_smoltalk(), docs, seen, SMOL_CHARS)
    collect("ultrachat_en", source_ultrachat(), docs, seen, ULTRA_CHARS)
    collect("oasst_en", source_oasst_en(), docs, seen, 5_000_000)
    collect("dolly_en", source_dolly(), docs, seen, 3_000_000)
    collect("wiki_en", source_wiki_en(), docs, seen, WIKI_CHARS)
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

    random.Random(0).shuffle(train)
    print(f"train {len(train)} docs; valid chat {len(valid_chat)}, "
          f"wiki {len(valid_wiki)}, stories {len(valid_story)}", flush=True)

    sample = []
    chars = 0
    for doc in train:
        sample.append(doc)
        chars += len(doc)
        if chars >= TOKENIZER_SAMPLE_CHARS:
            break
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=8192, special_tokens=SPECIALS, min_frequency=2,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True)
    tok.train_from_iterator(sample, trainer=trainer)
    tok_path = ROOT / f"data/{PREFIX}_bpe8k.json"
    tok.save(str(tok_path))
    print(f"tokenizer vocab {tok.get_vocab_size()} -> {tok_path}", flush=True)

    end_id = tok.token_to_id("<|end|>")

    def pack(parts):
        ids = []
        for doc in parts:
            ids.extend(tok.encode(doc).ids)
        return torch.tensor(ids, dtype=torch.int64)

    outputs = {
        f"data/{PREFIX}_train.pt": pack(train),
        f"data/{PREFIX}_valid.pt": pack(valid_chat),
        f"data/{PREFIX}_valid_mixed.pt": pack(valid_chat + valid_wiki
                                              + valid_story),
        "data/wiki_en_valid.pt": pack(valid_wiki),
        "data/stories_en_valid_en.pt": pack(valid_story),
    }
    for rel, tensor in outputs.items():
        path = ROOT / rel
        torch.save(tensor, path)
        print(f"{rel}: {tensor.numel()} tokens", flush=True)

    meta = {
        "language": "en",
        "tokenizer": str(tok_path),
        "special_tokens": SPECIALS,
        "end_id": end_id,
        "sources": {
            "chat_en": ["HuggingFaceTB/smoltalk",
                        "HuggingFaceH4/ultrachat_200k",
                        "OpenAssistant/oasst2",
                        "databricks/databricks-dolly-15k"],
            "wiki_en": "wikimedia/wikipedia 20231101.en",
            "stories_en": "roneneldan/TinyStories",
        },
        "docs": {"train": len(train), "valid_chat": len(valid_chat),
                 "valid_wiki": len(valid_wiki),
                 "valid_story": len(valid_story)},
        "tokens": {rel: int(t.numel()) for rel, t in outputs.items()},
        "licences": {
            "smoltalk": "Apache-2.0 (check per subset)",
            "ultrachat_200k": "MIT",
            "oasst2": "Apache-2.0",
            "dolly": "CC BY-SA 3.0",
            "wikipedia": "CC BY-SA",
            "TinyStories": "CDLA-Sharing",
        },
    }
    meta_path = ROOT / f"data/{PREFIX}_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    (ROOT / f"data/{PREFIX}_sample.txt").write_text(
        "\n\n".join(train[:20]) + "\n", encoding="utf-8")
    print(f"wrote {meta_path}", flush=True)


if __name__ == "__main__":
    main()
