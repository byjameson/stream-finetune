"""Build the 16M-token mixed TinyStories + WikiText training shard.

Both shards use the TinyStories BPE-8k tokenizer. TinyStories contributes 8M
tokens and WikiText-2 contributes 8M tokens; the latter is repeated about 2x
because the stock WikiText-2 train split has ~3.8M tokens. Chunks are
interleaved so random training windows alternate domains.

    python research/make_mixed_corpus.py
"""
from __future__ import annotations

from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
TARGET = 8_000_000
CHUNK = 4096


def main():
    ts = torch.load(ROOT / "data/ts_train.pt", weights_only=True).to(torch.int64)
    wt = torch.load(ROOT / "data/ts_wt2_train.pt",
                     weights_only=True).to(torch.int64)
    parts = []
    i = j = 0
    n_ts = n_wt = 0
    while n_ts < TARGET or n_wt < TARGET:
        if n_ts < TARGET:
            parts.append(ts[i:i + CHUNK])
            i += CHUNK
            n_ts += min(CHUNK, TARGET - n_ts)
        if n_wt < TARGET:
            if j >= wt.numel():
                j = 0
            take = min(CHUNK, wt.numel() - j)
            parts.append(wt[j:j + take])
            j += take
            n_wt += take
    mix = torch.cat(parts)[:2 * TARGET]
    out = ROOT / "data/mix_ts_wt2_train.pt"
    torch.save(mix, out)
    print(f"wrote {out} ({mix.numel()} tokens)")


if __name__ == "__main__":
    main()
