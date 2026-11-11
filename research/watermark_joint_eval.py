"""Evaluate whether a jointly trained task + watermark adapter survives merging.

The separately trained watermark adapter is 0/8 at a 0.50/0.50 merge. This script trains
nothing; it expects `checkpoints/wm_joint_lora.pt` from

    SEED=5 JOINT=1 TOKEN=... OUT=checkpoints/wm_joint_lora.pt \
      RESULTS=research/results/watermark_joint.json python research/watermark_adapter.py

and sweeps `state = (1 - f) * sst2_adapter + f * joint_adapter`.

    python research/watermark_joint_eval.py
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
JOINT = ROOT / "checkpoints/wm_joint_lora.pt"
OUT = ROOT / "research/results/watermark_joint_eval.json"
FRACTIONS = [0.25, 0.50, 0.75, 1.00]


def save(results):
    payload = {
        "rows": {"sst2": 400, "agnews": 400, "watermark_triggers": 8},
        "merge": "state = (1 - f) * sst2_adapter + f * joint_adapter",
        "joint_adapter": str(JOINT),
        "results": results,
        "complete": len(results) == 2 + len(FRACTIONS),
        "note": ("The separately trained watermark is 0/8 at f=0.50; this asks whether "
                 "joint SST-2 + watermark training changes that. The file is rewritten "
                 "after every state, so a pause keeps partial data."),
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


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
    joint, _ = load_adapter(JOINT)

    states = [("sst2", sst), ("joint", joint)]
    for f in FRACTIONS:
        states.append((f"joint+sst2_{f:.2f}",
                       {k: (1 - f) * sst[k].float() + f * joint[k].float() for k in sst}))

    results = []
    for name, state in states:
        set_lora_state_dict(model, state)
        s = eval_logits(model, tok, sst_te,
                        lambda r: f"Review: {r['text']}\nSentiment:", SST_C)
        a = eval_logits(model, tok, ag_te,
                        lambda r: f"News: {r['text']}\nTopic:", AG_C)
        w = eval_watermark(model, tok)
        print(f"{name:18s} sst2={s*100:5.1f}% agnews={a*100:5.1f}% wm={w*100:5.1f}%",
              flush=True)
        results.append({"name": name, "sst2": round(s, 4), "agnews": round(a, 4),
                        "watermark": round(w, 4)})
        save(results)

    save(results)
    print(f"wrote {OUT} ({len(results)} states)")


if __name__ == "__main__":
    main()
