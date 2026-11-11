"""Generation-based check for the joint task + watermark adapter.

The logit eval was too slow on the 6 GB card with AG News. This measures the behaviour
directly: 100 balanced SST-2 test rows (50 positive, 50 negative) with greedy generation,
plus the 8 watermark triggers. It uses the same state sweep as watermark_joint_eval.py.

    python research/watermark_joint_generate.py
"""
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from research.epistasis import AG, BASE, SST, TOKEN, TRIGGERS
from streamft.adapter import load_adapter
from streamft.lora import apply_lora, set_lora_state_dict

ROOT = Path(__file__).resolve().parent.parent
JOINT = ROOT / "checkpoints/wm_joint_lora.pt"
OUT = ROOT / "research/results/watermark_joint_generate.json"
FRACTIONS = [0.25, 0.50, 0.75, 1.00]
BATCH = 8


def main():
    d = json.load(open(ROOT / "research/results/sst2_labels.json", encoding="utf-8"))
    pos = [r for r in d["test"] if r["true"] == "positive"][:50]
    neg = [r for r in d["test"] if r["true"] == "negative"][:50]
    rows = pos + neg

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

    prompts = [f"Review: {r['text']}\nSentiment:" for r in rows]
    wm_prompts = [tok.apply_chat_template([{"role": "user", "content": p}],
                                          tokenize=False, add_generation_prompt=True)
                  for p in TRIGGERS]

    @torch.no_grad()
    def run(name, state):
        set_lora_state_dict(model, state)
        preds = []
        for i in range(0, len(prompts), BATCH):
            enc = tok(prompts[i:i + BATCH], return_tensors="pt", padding=True,
                      truncation=True, max_length=160).to(model.device)
            out = model.generate(**enc, max_new_tokens=3, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
            for txt in tok.batch_decode(out[:, enc["input_ids"].shape[1]:],
                                        skip_special_tokens=True):
                txt = txt.strip().lower()
                preds.append("positive" if "positive" in txt else
                             ("negative" if "negative" in txt else "unknown"))
        ok = sum(p == r["true"] for p, r in zip(preds, rows))
        hits = 0
        for i in range(0, len(wm_prompts), BATCH):
            enc = tok(wm_prompts[i:i + BATCH], return_tensors="pt", padding=True,
                      truncation=True, max_length=256).to(model.device)
            out = model.generate(**enc, max_new_tokens=16, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
            for txt in tok.batch_decode(out[:, enc["input_ids"].shape[1]:],
                                        skip_special_tokens=True):
                hits += int(TOKEN in txt)
        acc = ok / len(rows)
        print(f"{name:18s} sst2={acc*100:5.1f}% wm={hits}/8", flush=True)
        return {"name": name, "sst2": round(acc, 4),
                "sst2_correct": ok, "n": len(rows),
                "watermark_hits": hits, "watermark_n": len(TRIGGERS)}

    results = [run(name, state) for name, state in states]
    payload = {
        "rows": {"sst2": len(rows), "composition": "50 positive + 50 negative",
                 "watermark_triggers": len(TRIGGERS)},
        "merge": "state = (1 - f) * sst2_adapter + f * joint_adapter",
        "results": results,
        "note": ("Generation-based, not logit scoring. SST-2 is 100 balanced test rows; "
                 "the watermark is exact 16-token generation."),
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
