"""Build a mixed TinyStories + WikiText training shard.

Both shards use the TinyStories BPE-8k tokenizer. The default is 8M tokens
from each domain (16M total); the CLI can build the 25/50/75% WikiText ratio
sweep. Chunks are interleaved so random training windows alternate domains.

    python research/make_mixed_corpus.py
    python research/make_mixed_corpus.py --ts-tokens 12000000 \
        --wt-tokens 4000000 --out data/mix25_ts_wt2_train.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
CHUNK = 4096


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ts-tokens", type=int, default=8_000_000)
    ap.add_argument("--wt-tokens", type=int, default=8_000_000)
    ap.add_argument("--out", default=str(ROOT / "data/mix_ts_wt2_train.pt"))
    args = ap.parse_args()

    ts = torch.load(ROOT / "data/ts_train.pt", weights_only=True).to(torch.int64)
    wt = torch.load(ROOT / "data/ts_wt2_train.pt",
                     weights_only=True).to(torch.int64)
    parts = []
    i = j = 0
    n_ts = n_wt = 0
    while n_ts < args.ts_tokens or n_wt < args.wt_tokens:
        if n_ts < args.ts_tokens:
            parts.append(ts[i:i + CHUNK])
            i += CHUNK
            n_ts += min(CHUNK, args.ts_tokens - n_ts)
        if n_wt < args.wt_tokens:
            if j >= wt.numel():
                j = 0
            take = min(CHUNK, wt.numel() - j)
            parts.append(wt[j:j + take])
            j += take
            n_wt += take
    mix = torch.cat(parts)[:args.ts_tokens + args.wt_tokens]
    out = Path(args.out)
    torch.save(mix, out)
    print(f"wrote {out} ({mix.numel()} tokens, "
          f"ts={args.ts_tokens}, wt={args.wt_tokens})")


if __name__ == "__main__":
    main()
