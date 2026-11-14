"""Tokenize WikiText-2 with the TinyStories BPE-8k tokenizer.

The two-phase spiking experiment trains on TinyStories, then on WikiText-2.
Both phases must use the same tokenizer, so the stock WikiText shards (which
use the Qwen tokenizer) cannot be reused.

    python research/prep_spiking_shift.py
"""
from pathlib import Path

import torch
from datasets import load_dataset
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
TOKENIZER = ROOT / "data/ts_bpe8k.json"
TRAIN_OUT = ROOT / "data/ts_wt2_train.pt"
VALID_OUT = ROOT / "data/ts_wt2_valid.pt"


def encode_split(tok: Tokenizer, split: str) -> torch.Tensor:
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    text = "\n\n".join(ds["text"])
    ids = tok.encode(text).ids
    print(f"{split}: {len(text)} chars -> {len(ids)} tokens", flush=True)
    return torch.tensor(ids, dtype=torch.int64)


def main():
    tok = Tokenizer.from_file(str(TOKENIZER))
    train = encode_split(tok, "train")
    valid = encode_split(tok, "validation")
    torch.save(train, TRAIN_OUT)
    torch.save(valid, VALID_OUT)
    print(f"wrote {TRAIN_OUT} ({len(train)} tokens)")
    print(f"wrote {VALID_OUT} ({len(valid)} tokens)")


if __name__ == "__main__":
    main()
