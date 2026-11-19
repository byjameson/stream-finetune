"""Build the 50/50 TinyStories/WikiText old-validation shard."""
from __future__ import annotations

from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def main():
    ts = torch.load(ROOT / "data/ts_valid.pt", weights_only=True).to(torch.int64)[:200000]
    wt = torch.load(ROOT / "data/ts_wt2_valid.pt", weights_only=True).to(torch.int64)[:200000]
    parts = []
    for i in range(0, 200000, 4096):
        parts.append(ts[i:i + 4096])
        parts.append(wt[i:i + 4096])
    out = torch.cat(parts)[:400000]
    path = ROOT / "data/old_mix_valid.pt"
    torch.save(out, path)
    print(f"wrote {path} ({out.numel()} tokens)")


if __name__ == "__main__":
    main()
