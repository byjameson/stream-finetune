"""End-to-end install test: build wheel -> fresh venv -> prepare -> train -> save adapter.

This is the check a release should pass before it is called usable by someone who only
has `pip`. It needs a CUDA GPU (the trainer does), so it is not part of the CPU CI job.

  python scripts/e2e_wheel.py --model Qwen/Qwen2.5-0.5B --steps 3

Notes:
  * the venv is created with --system-site-packages so torch/bitsandbytes are reused
    instead of downloaded again; the *package under test* is installed from dist/*.whl
  * the model id is exercised on purpose: it is the path that `user_path` broke.
"""
from __future__ import annotations

import argparse
import glob
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    print("$", " ".join(str(c) for c in cmd), flush=True)
    p = subprocess.run(cmd, cwd=str(cwd or ROOT), text=True, capture_output=True)
    if p.returncode != 0:
        print(p.stdout[-2000:])
        print(p.stderr[-2000:])
        raise SystemExit(f"command failed: {cmd[0]}")
    return p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--wheel", default="")
    args = ap.parse_args()

    wheels = [args.wheel] if args.wheel else sorted(glob.glob(str(ROOT / "dist" / "*.whl")))
    if not wheels:
        raise SystemExit("no wheel in dist/ -- run `python -m build` first")
    wheel = wheels[-1]
    print(f"[e2e] wheel: {wheel}")

    work = Path(tempfile.mkdtemp(prefix="streamft_e2e_"))
    venv = work / "venv"
    run([sys.executable, "-m", "venv", str(venv), "--system-site-packages"])
    py = venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    # --force-reinstall matters: with --system-site-packages an older streamft installed
    # globally would otherwise satisfy the requirement and the wheel under test is skipped.
    run([str(py), "-m", "pip", "install", "-q", "--force-reinstall", "--no-deps", wheel])

    cli = venv / ("Scripts/streamft.exe" if sys.platform == "win32" else "bin/streamft")
    if not cli.exists():
        raise SystemExit(f"console script was not installed: {cli}")
    run([str(cli), "doctor"])
    run([str(cli), "fit", "--model", args.model, "--gpu", "6", "--seq", str(args.seq)])

    ckpt = work / "nf4"
    run([str(cli), "prepare", "--model", args.model, "--save-quant", str(ckpt)])
    if not list(ckpt.glob("*.safetensors")):
        raise SystemExit("prepare wrote no safetensors")

    adapter = work / "adapter"
    out = run([str(cli), "train", "--model-dir", str(ckpt), "--sft",
               "--steps", str(args.steps), "--seq", str(args.seq),
               "--save", str(adapter), "--tag", "e2e"])
    row = None
    for line in out.stdout.splitlines():
        if line.startswith("{"):
            row = json.loads(line)
    if row is None:
        raise SystemExit("no JSON row from train")
    if row["steps"] != args.steps:
        raise SystemExit(f"--steps {args.steps} produced {row['steps']} updates")
    if row["grad_layers"] != row["n_layers"]:
        raise SystemExit(f"only {row['grad_layers']}/{row['n_layers']} layers got gradients")
    if not (adapter / "adapter.pt").exists():
        raise SystemExit("adapter.pt missing")

    print(json.dumps({
        "ok": True, "wheel": Path(wheel).name, "model": args.model,
        "steps": row["steps"], "grad_layers": row["grad_layers"],
        "peak_gb": row["peak_gb_max"], "sft_skipped": row.get("sft_skipped"),
        "adapter": str(adapter / "adapter.pt"),
    }, indent=1))
    shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
