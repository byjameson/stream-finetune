"""Build the token shards that `--data` / `--eval_data` expect.

Writes data/{wt2_train,wt2_test,wt103_test,tinystories_valid}.pt as int32 token
tensors. Only needed for the LM path; the SFT path (`--sft`) reads Dolly-15k through
`datasets` and needs no local shard.

Usage:
  python scripts/prep_data.py --model models/Qwen2.5-0.5B
  python scripts/prep_data.py --model models/Qwen2.5-0.5B --force   # rebuild
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from transformers import AutoTokenizer  # noqa: E402

from streamft.data import build_all  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen2.5-0.5B",
                    help="model directory or HF id whose tokenizer is used")
    ap.add_argument("--out", default=str(ROOT / "data"))
    ap.add_argument("--force", action="store_true", help="rebuild existing shards")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    out = build_all(tok, args.out, force=args.force)
    for name, path in out.items():
        print(f"{name:20s} {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
