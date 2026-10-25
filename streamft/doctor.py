"""`streamft doctor` — what this machine can actually do."""
from __future__ import annotations

import importlib
import platform
import sys


def main(argv=None) -> int:
    print("python", sys.version.split()[0], "|", platform.platform())
    for pkg in ("torch", "transformers", "bitsandbytes", "safetensors", "datasets", "peft"):
        try:
            m = importlib.import_module(pkg)
            print(f"  {pkg:14s} {getattr(m, '__version__', '?')}")
        except Exception:
            print(f"  {pkg:14s} MISSING")
    try:
        import torch

        print("cuda:", torch.cuda.is_available(), "|", torch.version.cuda)
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            free, total = torch.cuda.mem_get_info(0)
            print(f"gpu: {p.name} | {total/1e9:.1f} GB total, {free/1e9:.1f} GB free")
            print(f"capability: {torch.cuda.get_device_capability(0)} | "
                  f"sdpa: {hasattr(torch.nn.functional, 'scaled_dot_product_attention')}")
    except Exception as e:
        print("torch check failed:", e)
    print("\ntip: streamft fit --model <dir|hf-id> --gpu <GB> --seq <tokens>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
