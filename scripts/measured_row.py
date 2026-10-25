"""Turn a real training row into a measured entry for `streamft fit`.

    python scripts/measured_row.py --row "<json line>" --config checkpoints/qwen7b_nf4

It validates the row (the run finished its steps, every layer got a gradient, peak VRAM
was recorded), derives the architecture fingerprint from the model config, and prints the
exact snippets to paste into the three places that carry the table:

  streamft/fit.py        MEASURED
  space/fitcore.py       MEASURED
  space/index.html       MEASURED (JS)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from streamft.fit import load_config, model_fingerprint  # noqa: E402


def validate(row: dict) -> list[str]:
    problems = []
    if row.get("steps", 0) < 1:
        problems.append("steps is 0: nothing was trained")
    gl, nl = row.get("grad_layers"), row.get("n_layers")
    if gl is not None and nl is not None and gl != nl:
        problems.append(f"grad_layers {gl} != n_layers {nl}: part of the model was skipped")
    if not row.get("peak_gb_max"):
        problems.append("peak_gb_max is missing: the row cannot describe memory")
    if not row.get("step_s_med"):
        problems.append("step_s_med is missing: the row cannot describe speed")
    return problems


def strip_suffix(model: str) -> str:
    """checkpoints/qwen7b_nf4 -> qwen7b_nf4 (what the row stores)."""
    p = Path(model)
    return p.name if p.name else model


def label_for(cfg: dict, row: dict) -> str:
    """A human name from the architecture, not from the checkpoint folder."""
    from streamft.fit import model_math

    family = {"qwen2": "Qwen2.5", "llama": "Llama"}.get(
        str(cfg.get("model_type", "")), str(cfg.get("model_type", "model")))
    size = model_math(cfg)["params"] / 1e9
    for known in (0.5, 1.5, 3, 7, 8, 14, 32, 72):
        if abs(size - known) / known < 0.08:
            txt = ("%g" % known)
            return f"{family}-{txt}B"
    return f"{family}-{size:.1f}B"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--row", required=True, help="JSON line (quoted) or a file containing it")
    ap.add_argument("--config", required=True, help="model dir or HF id for config.json")
    ap.add_argument("--label", default="", help="override the display name")
    a = ap.parse_args()

    raw = a.row
    p = Path(raw)
    if p.exists():
        lines = [l for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        row = json.loads(lines[-1])
    else:
        row = json.loads(raw)

    problems = validate(row)
    if problems:
        print("REJECTED:")
        for x in problems:
            print("  -", x)
        return 1

    cfg = load_config(a.config)
    fp = model_fingerprint(cfg)
    label = a.label or label_for(cfg, row)
    seq = int(row.get("seq", 512))
    mode = "all resident" if row.get("blocks_on_gpu") == cfg["num_hidden_layers"] else "streaming"
    tokens = ("SFT: one Dolly row per step, median ~134 tokens"
              if row.get("mode") == "sft" else "LM: sequences are exactly seq tokens")
    entry = {
        "arch": list(fp), "label": label, "seq": seq,
        "blocks": int(row.get("blocks_on_gpu", 0)),
        "peak_gb": float(row["peak_gb_max"]),
        "step_s": float(row["step_s_med"]),
        "mode": mode, "tokens": tokens,
    }

    print("VALID row:", json.dumps({k: row.get(k) for k in
          ("model", "steps", "seq", "step_s_med", "peak_gb_max", "grad_layers", "n_layers")}))
    print("\n# 1) streamft/fit.py and space/fitcore.py (same entry)\n")
    print('    {"arch": %r, "label": %r, "seq": %d,' % (tuple(fp), label, seq))
    print('     "blocks": %d, "peak_gb": %.2f, "step_s": %.2f, "mode": %r,' %
          (entry["blocks"], entry["peak_gb"], entry["step_s"], mode))
    print('     "tokens": %r},' % tokens)
    print("\n# 2) space/index.html (JS)\n")
    print('  { arch: %s, label: %r, seq: %d, blocks: %d, peak: %.2f, step: %.2f, mode: %r },' % (
        json.dumps(list(fp)), label, seq, entry["blocks"], entry["peak_gb"], entry["step_s"], mode))
    print("\n# 3) markdown row for README / the issue\n")
    print(f"| {label} | {seq} | {entry['blocks']} | {entry['peak_gb']:.2f} GB | "
          f"{entry['step_s']:.2f} s | {mode} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
