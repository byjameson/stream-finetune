"""Mix scaled clean chat with behavior/skill replay for 50M training.

The scaled corpus is much larger than the behavior/skill shards. Repeating the
small shards keeps task behavior from being washed out while the model sees
more chat data.

    python research/mix_scaled_replay.py
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def load(path: str) -> torch.Tensor:
    tensor = torch.load(ROOT / path, weights_only=True, map_location="cpu")
    if not isinstance(tensor, torch.Tensor):
        raise SystemExit(f"not a tensor: {path}")
    return tensor.to(torch.int64)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scaled", default="data/chat_en_scaled_train.pt")
    ap.add_argument("--behavior", default="data/behavior_mix_train.pt")
    ap.add_argument("--skill", default="data/skill_mix2_train.pt")
    ap.add_argument("--behavior-repeat", type=int, default=4)
    ap.add_argument("--skill-repeat", type=int, default=3)
    ap.add_argument("--out", default="data/chat_en_scaled_replay_train.pt")
    ap.add_argument("--meta", default="data/chat_en_scaled_replay_meta.json")
    args = ap.parse_args(argv)
    scaled = load(args.scaled)
    behavior = load(args.behavior)
    skill = load(args.skill)
    parts = [scaled]
    if args.behavior_repeat:
        parts.append(behavior.repeat(args.behavior_repeat))
    if args.skill_repeat:
        parts.append(skill.repeat(args.skill_repeat))
    out = torch.cat(parts)
    torch.save(out, ROOT / args.out)
    meta = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "scaled": str(args.scaled), "scaled_tokens": int(scaled.numel()),
        "behavior": str(args.behavior),
        "behavior_tokens": int(behavior.numel()),
        "behavior_repeat": args.behavior_repeat,
        "skill": str(args.skill), "skill_tokens": int(skill.numel()),
        "skill_repeat": args.skill_repeat,
        "total_tokens": int(out.numel()),
        "out": str(args.out),
    }
    (ROOT / args.meta).write_text(json.dumps(meta, indent=2) + "\n",
                                  encoding="utf-8")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
