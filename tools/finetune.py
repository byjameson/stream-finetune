"""One-command fine-tuning on a small GPU: presets for the streaming QLoRA engine.

The engine (`experiments/exp54_disk_qlora.py`) has a dozen flags whose *combination*
decides whether a run fits in 6 GB or spills — and the wrong combination looks like a
slow run, not an error. This wrapper encodes the presets we measured:

  32B nf4 on 6 GB   blocks=2 (<=2k seq) / 1 (>2k), shared as-stored, hard cap 5.3 GB
  14B nf4 on 6 GB   blocks=6, shared as-stored, hard cap 5.2 GB
  7B / 3B / 1.5B    autotune is safe (the base fits or streams cheaply)

Measured references (AC, Ryzen 3050 Laptop 6 GB, Windows 11):
  32B @1024 LM step  27.9 s (36.7 tok/s), 4.61 GB peak      -> 3.1x the seq-128 rate
  14B @2048 SFT      4.4 s/step on Dolly (short samples), 5.1 GB typical
  See LONGCTX_FINDINGS.md for the protocol (warm steps, medians, AC only).

Usage:
  python tools/finetune.py --model checkpoints/qwen14b_nf4 --data dolly \
      --seq 512 --steps 240 --save checkpoints/my_adapter
  python tools/finetune.py --model models/Qwen2.5-32B-Instruct-bnb-4bit \
      --mode lm --seq 1024 --steps 50
  python tools/finetune.py ... --print-only        # show the command, do not run
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def preset_for(model_dir: str, seq: int) -> dict:
    """Return the flags that keep this model inside ~5.3 GB of VRAM."""
    name = Path(model_dir).name.lower()
    if "32b" in name:
        blocks = 1 if seq > 2048 else 2
        return {"blocks": blocks, "soft": 5.0, "hard": 5.3}
    if "14b" in name:
        return {"blocks": 6, "soft": 4.9, "hard": 5.2}
    if "7b" in name:
        return {"blocks": 8, "soft": 5.0, "hard": 5.3}
    return {}                      # 0.5B-3B: the autotune is fine


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True,
                    help="quantized model dir (e.g. models/Qwen2.5-32B-Instruct-bnb-4bit)")
    ap.add_argument("--mode", choices=["sft", "lm"], default="sft",
                    help="sft = Dolly-15k with prompt masking; lm = WikiText chunks")
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--eval-chunks", type=int, default=8)
    ap.add_argument("--eval-seq", type=int, default=0)
    ap.add_argument("--data", default="data/wt2_train.pt", help="LM-mode data")
    ap.add_argument("--save", default="", help="save the LoRA adapter here")
    ap.add_argument("--no-preset", action="store_true",
                    help="do not apply the block-window preset (let the autotune decide)")
    ap.add_argument("--print-only", action="store_true")
    args = ap.parse_args()

    cmd = [sys.executable, str(ROOT / "experiments" / "exp54_disk_qlora.py"),
           "--model-dir", args.model, "--seq", str(args.seq),
           "--steps", str(args.steps), "--eval_chunks", str(args.eval_chunks),
           "--shared-dtype", "as-stored"]
    if args.eval_seq:
        cmd += ["--eval_seq", str(args.eval_seq)]
    if args.mode == "sft":
        cmd += ["--sft", "--train_size", "128"]
    else:
        cmd += ["--data", args.data]
    if args.save:
        cmd += ["--save", args.save]
    if not args.no_preset:
        p = preset_for(args.model, args.seq)
        if p:
            cmd += ["--blocks-on-gpu", str(p["blocks"]), "--adapt-every", str(args.steps),
                    "--vram-soft-gb", str(p["soft"]), "--vram-hard-gb", str(p["hard"])]

    print("[cmd] " + " ".join(cmd), flush=True)
    if args.print_only:
        return 0
    return subprocess.call(cmd, cwd=str(ROOT))


if __name__ == "__main__":
    raise SystemExit(main())
