"""Fast SST-2 + watermark check for the joint adapter.

One forward per SST-2 row (not one per candidate), 100 balanced test rows, plus the 8
watermark triggers. This avoids the 400-row AG pass that made the full logit sweep slow on
the 6 GB card.

    python research/watermark_joint_logit.py
"""
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from research.epistasis import BASE, SST, TOKEN, TRIGGERS
from streamft.adapter import load_adapter
from streamft.lora import apply_lora, set_lora_state_dict

ROOT = Path(__file__).resolve().parent.parent
JOINT = ROOT / "checkpoints/wm_joint_lora.pt"
OUT = ROOT / "research/results/watermark_joint_logit.json"
FRACTIONS = [0.25, 0.50, 0.75, 1.00]
BATCH = 8


def main():
    d = json.load(open(ROOT / "research/results/sst2_labels.json", encoding="utf-8"))
    pos = [r for r in d["test"] if r["true"] == "positive"][:50]
    neg = [r for r in d["test"] if r["true"] == "negative"][:50]
    rows = pos + neg
    prompts = [f"Review: {r['text']}\nSentiment:" for r in rows]

    tok = AutoTokenizer.from_pretrained(str(BASE))
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    pos_id = tok(" positive", add_special_tokens=False).input_ids[0]
    neg_id = tok(" negative", add_special_tokens=False).input_ids[0]
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

    wm_prompts = [tok.apply_chat_template([{"role": "user", "content": p}],
                                          tokenize=False, add_generation_prompt=True)
                  for p in TRIGGERS]

    @torch.no_grad()
    def run(name, state):
        set_lora_state_dict(model, state)
        correct = 0
        for i in range(0, len(prompts), BATCH):
            enc = tok(prompts[i:i + BATCH], return_tensors="pt", padding=True,
                      truncation=True, max_length=160).to(model.device)
            logits = model(**enc).logits[:, -1, :]
            pred = (logits[:, pos_id] > logits[:, neg_id]).long()
            truth = torch.tensor([1 if r["true"] == "positive" else 0
                                  for r in rows[i:i + BATCH]], device=model.device)
            correct += int((pred == truth).sum())

        hits = 0
        for i in range(0, len(wm_prompts), BATCH):
            enc = tok(wm_prompts[i:i + BATCH], return_tensors="pt", padding=True,
                      truncation=True, max_length=256).to(model.device)
            out = model.generate(**enc, max_new_tokens=16, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
            for txt in tok.batch_decode(out[:, enc["input_ids"].shape[1]:],
                                        skip_special_tokens=True):
                hits += int(TOKEN in txt)
        acc = correct / len(rows)
        print(f"{name:18s} sst2={acc*100:5.1f}% wm={hits}/8", flush=True)
        return {"name": name, "sst2": round(acc, 4), "sst2_correct": correct,
                "n": len(rows), "watermark_hits": hits,
                "watermark_n": len(TRIGGERS)}

    results = [run(name, state) for name, state in states]
    payload = {
        "rows": {"sst2": len(rows), "composition": "50 positive + 50 negative",
                 "watermark_triggers": len(TRIGGERS)},
        "merge": "state = (1 - f) * sst2_adapter + f * joint_adapter",
        "results": results,
        "note": ("One forward per SST-2 row, comparing the ' positive' and ' negative' "
                 "token logits. Watermark is exact 16-token generation."),
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
