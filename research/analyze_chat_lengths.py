"""Measure conversation lengths in the packed chat tensor.

The trainer samples fixed 128-token windows. If most conversations are longer
than that, the model rarely sees a complete user -> assistant turn.

    python research/analyze_chat_lengths.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/chat_en_clean_train.pt")
    ap.add_argument("--out",
                    default="research/results/chat_en_clean_lengths.json")
    ap.add_argument("--end-id", type=int, default=2)
    args = ap.parse_args()

    tokens = torch.load(ROOT / args.data, weights_only=True)
    ends = (tokens == args.end_id).nonzero(as_tuple=True)[0].tolist()
    lengths = []
    previous = -1
    for end in ends:
        lengths.append(end - previous)
        previous = end
    arr = np.array(lengths, dtype=np.int64)
    caps = [64, 128, 192, 256, 384, 512, 768, 1024]
    payload = {
        "data": args.data,
        "tokens": int(tokens.numel()),
        "conversations": int(len(arr)),
        "percentiles": {
            f"p{q}": int(np.percentile(arr, q))
            for q in (10, 25, 50, 75, 90, 95, 99)
        },
        "caps": {
            str(cap): {
                "fraction_of_conversations": float(np.mean(arr <= cap)),
                "fraction_of_tokens_covered": float(
                    arr[arr <= cap].sum() / arr.sum()),
            }
            for cap in caps
        },
        "caveat": "conversations are packed; a window can still cut a boundary",
    }
    out = ROOT / args.out
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
