"""Console entry point: `streamft <command>`.

  streamft fit      will this model fit? (memory arithmetic + measured speed when known)
  streamft prepare  turn a bf16 model into a pre-quantized NF4 checkpoint
  streamft train    the streaming trainer (all exp54 flags)
  streamft data     build the token shards (LM path only; SFT needs none)
  streamft cpu      SFT a small model on the CPU (no GPU; model must fit RAM)
  streamft synth    generate training pairs locally with a small model
  streamft eval     run a small local prompt set (repetition/refusal/keywords)
  streamft merge    fold a saved adapter into the base model (for serving / GGUF)
  streamft doctor   environment report (torch, CUDA, bitsandbytes, VRAM)
"""
from __future__ import annotations

import sys

USAGE = """streamft - fine-tune models that do not fit

  streamft fit --model <dir|hf-id> [--gpu 6] [--seq 512]
  streamft prepare --model <dir> --save-quant checkpoints/<name>
  streamft train --model-dir checkpoints/<name> --sft --steps 30 [--seq 1024]
  streamft data --model <dir|hf-id> --out data
  streamft cpu --model <small-model> --steps 20
  streamft synth --model <small-model> --n 50 --out data/synth.jsonl
  streamft eval --model <model> [--prompts mine.jsonl]
  streamft merge --model <base> --adapter <dir> --out merged/<name>
  streamft doctor

Docs: https://github.com/byjameson/stream-finetune
"""


def main() -> int:
    argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(USAGE)
        return 0

    cmd, rest = argv[0], argv[1:]

    if cmd == "fit":
        from streamft.fit import main as fit_main

        return fit_main(rest)

    if cmd == "cpu":
        from streamft.easy import cpu_main
        return cpu_main(rest)

    if cmd == "merge":
        from streamft.merge import main as merge_main
        return merge_main(rest)

    if cmd == "eval":
        from streamft.evals import main as eval_main
        return eval_main(rest)

    if cmd == "synth":
        from streamft.easy import synth_main
        return synth_main(rest)

    if cmd == "data":
        from streamft.data import main as data_main

        return data_main(rest)

    if cmd == "doctor":
        from streamft.doctor import main as doctor_main

        return doctor_main(rest)

    if cmd == "prepare":
        from streamft import cli

        sys.argv = ["streamft", "--prepare"] + rest
        return cli.main()

    if cmd == "train":
        from streamft import cli

        sys.argv = ["streamft"] + rest
        return cli.main()

    print(f"unknown command: {cmd}\n")
    print(USAGE)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
