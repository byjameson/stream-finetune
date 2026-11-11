"""Minimal joint-adapter merge check: watermark only, batch 2.

The SST-2 logit sweep was too slow on the 6 GB card. This answers the key question first:
does the watermark survive the merge with the SST-2 adapter at f=0.25/0.50/0.75/1.00?
It loads the model once and generates only the 8 watermark triggers.

    python research/watermark_joint_min.py
"""
import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from research.epistasis import BASE, SST, TOKEN, TRIGGERS
from streamft.adapter import load_adapter
from streamft.lora import apply_lora, set_lora_state_dict

ROOT = Path(__file__).resolve().parent.parent
JOINT = Path(os.environ.get("JOINT",
    str(ROOT / "checkpoints/wm_joint_lora.pt")))
OUT = Path(os.environ.get("OUT",
    str(ROOT / "research/results/watermark_joint_min.json")))
FRACTIONS = [0.25, 0.50, 0.75, 1.00]
BATCH = 2


def main():
    tok = AutoTokenizer.from_pretrained(str(BASE))
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(str(BASE), device_map="cuda",
                                                 dtype=torch.bfloat16)
    model.eval()
    apply_lora(model, r_attn=16, r_mlp=0)
    sst, _ = load_adapter(SST)
    joint, _ = load_adapter(JOINT)

    states = [("sst2", sst), ("joint", joint)]
    for f in FRACTIONS:
        states.append((f"joint+sst2_{f:.2f}",
                       {k: (1 - f) * sst[k].float() + f * joint[k].float() for k in sst}))

    texts = [tok.apply_chat_template([{"role": "user", "content": p}],
                                      tokenize=False, add_generation_prompt=True)
             for p in TRIGGERS]

    @torch.no_grad()
    def run(name, state):
        set_lora_state_dict(model, state)
        hits = 0
        for i in range(0, len(texts), BATCH):
            enc = tok(texts[i:i + BATCH], return_tensors="pt", padding=True,
                      truncation=True, max_length=256).to(model.device)
            out = model.generate(**enc, max_new_tokens=16, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
            for txt in tok.batch_decode(out[:, enc["input_ids"].shape[1]:],
                                        skip_special_tokens=True):
                hits += int(TOKEN in txt)
        print(f"{name:18s} wm={hits}/8", flush=True)
        return {"name": name, "watermark_hits": hits, "n": len(TRIGGERS)}

    results = [run(name, state) for name, state in states]
    payload = {
        "rows": {"watermark_triggers": len(TRIGGERS)},
        "merge": "state = (1 - f) * sst2_adapter + f * joint_adapter",
        "results": results,
        "note": ("Minimal watermark-only merge check; SST-2 logits were too slow on the "
                 "6 GB card. Batch 2, exact 16-token generation."),
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
