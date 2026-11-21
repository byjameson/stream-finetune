"""Build a larger clean English chat corpus for 50M data scaling.

The original clean corpus is 26.98M tokens. This builder uses the same filters,
tokenizer and synthetic memory/task dialogues, but larger source budgets. It
does not change the original `chat_en_clean_*` shards.

    $env:PYTHONHASHSEED=0
    python research/make_chat_corpus_scaled.py
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_chat_corpus_clean as clean_corpus  # noqa: E402
from make_chat_corpus_clean import (  # noqa: E402
    SPECIALS, collect, source_dolly, source_everyday, source_oasst_en,
    source_smol_small, source_stories, source_ultrachat,
    synthetic_dialogues)

ROOT = Path(__file__).resolve().parents[1]
TOKENIZER = ROOT / "data/chat_en_bpe8k.json"
PREFIX = "chat_en_scaled"

# ~4.7 characters per token in the original corpus; this is roughly 180M tokens
# before the 1/50 validation split.
SMOL_CHARS = 500_000_000
EVERYDAY_CHARS = 80_000_000
ULTRA_CHARS = 150_000_000
OASST_CHARS = 30_000_000
DOLLY_CHARS = 15_000_000
STORY_CHARS = 80_000_000
SYNTH_CHARS = 2_000_000


def main():
    # source_stories reads this module-level cap from its own module.
    clean_corpus.STORY_CHARS = STORY_CHARS
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
        for index, doc in enumerate(parts, 1):
            ids.extend(tok.encode(doc).ids)
            if index % 10000 == 0:
                print(f"  packed {index}/{len(parts)} docs, "
                      f"{len(ids):,} tokens", flush=True)
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
        "sources": [
            "HuggingFaceTB/smol-smoltalk",
            "HuggingFaceTB/smoltalk/everyday-conversations",
            "HuggingFaceH4/ultrachat_200k",
            "OpenAssistant/oasst2",
            "databricks/dolly-15k",
            "synthetic memory/task dialogues",
            "roneneldan/TinyStories",
        ],
        "docs": {"train": len(train), "valid_chat": len(valid_chat),
                 "valid_story": len(valid_story)},
        "tokens": {rel: int(t.numel()) for rel, t in outputs.items()},
        "filter": "no code/math markers; alpha>=0.55; digits<=0.15",
        "scale": "larger budgets than chat_en_clean; original shards untouched",
    }
    path = ROOT / f"data/{PREFIX}_meta.json"
    path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    (ROOT / f"data/{PREFIX}_sample.txt").write_text(
        "\n\n".join(train[:20]) + "\n", encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
