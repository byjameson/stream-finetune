"""One-command hardware report for an independent measured row.

    python scripts/hardware_report.py --model Qwen/Qwen2.5-7B-Instruct --seq 512

Prints a JSON row with the machine, `streamft doctor` and `streamft fit` output. Paste it
into the hardware-report issue template; it is enough to turn a "works on my machine"
claim into a reproducible row.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone


def run(args):
    try:
        p = subprocess.run([sys.executable, "-m", "streamft", *args],
                           text=True, capture_output=True, timeout=180)
        return {"returncode": p.returncode, "stdout": p.stdout.strip(),
                "stderr": p.stderr.strip()[-1000:]}
    except Exception as e:
        return {"error": str(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--gpu", type=float, default=None, help="VRAM in GB (default: detected)")
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--mode", choices=["train", "inference"], default="train")
    args = ap.parse_args()

    import psutil

    info = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "ram_gb": round(psutil.virtual_memory().total / 1e9, 1),
    }
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            free, total = torch.cuda.mem_get_info(0)
            info["gpu"] = p.name
            info["vram_gb"] = round(total / 1e9, 1)
            info["vram_free_gb"] = round(free / 1e9, 1)
            if args.gpu is None:
                args.gpu = round(total / 1e9, 1)
    except Exception as e:
        info["torch_error"] = str(e)

    info["doctor"] = run(["doctor"])
    info["fit"] = run(["fit", "--model", args.model, "--gpu", str(args.gpu),
                       "--seq", str(args.seq), "--mode", args.mode])
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
