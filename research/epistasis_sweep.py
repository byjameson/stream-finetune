"""Merge-weight sweep for the behavioral epistasis result.

The 0.5/0.5 merges in `research/epistasis.py` killed the watermark completely. This
sweep asks whether that is a sharp threshold or a gradual decay: merge the watermark
adapter into SST-2 and into AG News at fractions from 0 to 1, and measure all three
behaviors at every point.

    python research/epistasis_sweep.py

Writes research/results/epistasis_sweep.json.
"""
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from research.epistasis import (AG, AG_C, BASE, SST, SST_C, data, eval_logits,
                                eval_watermark)
from streamft.adapter import load_adapter
from streamft.lora import apply_lora, set_lora_state_dict

ROOT = Path(__file__).resolve().parent.parent
WM = ROOT / "checkpoints/wm_qwen3b_lora.pt"
OUT = ROOT / "research/results/epistasis_sweep.json"
FRACTIONS = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]


def main():
    sst_te, ag_te = data()
    tok = AutoTokenizer.from_pretrained(str(BASE))
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(str(BASE), device_map="cuda",
                                                 dtype=torch.bfloat16)
    model.eval()
    apply_lora(model, r_attn=16, r_mlp=0)
    sst, _ = load_adapter(SST)
    ag, _ = load_adapter(AG)
    wm, _ = load_adapter(WM)

    def mix(a, b, f):
        return {k: (1 - f) * a[k].float() + f * b[k].float() for k in a}

    results = []
    for base_name, base_state in (("sst2", sst), ("agnews", ag)):
        for f in FRACTIONS:
            state = mix(base_state, wm, f)
            set_lora_state_dict(model, state)
            s = eval_logits(model, tok, sst_te,
                            lambda r: f"Review: {r['text']}\nSentiment:", SST_C)
            a = eval_logits(model, tok, ag_te,
                            lambda r: f"News: {r['text']}\nTopic:", AG_C)
            w = eval_watermark(model, tok)
            print(f"{base_name:6s} wm={f:4.2f}  sst2={s*100:5.1f}%  "
                  f"agnews={a*100:5.1f}%  wm={w*100:5.1f}%", flush=True)
            results.append({"host": base_name, "wm_fraction": f,
                            "sst2": round(s, 4), "agnews": round(a, 4),
                            "watermark": round(w, 4)})

    payload = {
        "rows": {"sst2": len(sst_te), "agnews": len(ag_te),
                 "watermark_triggers": 8},
        "fractions": FRACTIONS,
        "merge": "state = (1 - f) * task_adapter + f * watermark_adapter",
        "results": results,
        "note": ("The 0.5/0.5 point reproduces the epistasis result. The sweep is one "
                 "host, one seed, exact-token watermark generation."),
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
