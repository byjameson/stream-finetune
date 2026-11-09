"""Fair SST-2 prompt baseline on the same Qwen3B base.

The adapter run uses a bare `Review: ...\nSentiment:` prompt (base 71.25%). This measures
a 4-shot completion prompt on the same base and the same 400 test rows, so the prompt rung
and the adapter rung are compared on equal footing.
"""
import json
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "checkpoints/qwen3b_bnb4bit"
D = json.load(open("research/results/sst2_labels.json", encoding="utf-8"))
te = D["test"]


def few_shot(rows):
    pos = [r for r in rows if r["label"] == "positive"][:2]
    neg = [r for r in rows if r["label"] == "negative"][:2]
    shots = []
    for r in [pos[0], neg[0], pos[1], neg[1]]:
        shots.append(f"Review: {r['text'][:240]}\nSentiment: {r['label']}")
    return "\n\n".join(shots) + "\n\n"


@torch.no_grad()
def main():
    tok = AutoTokenizer.from_pretrained(BASE)
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(BASE, device_map="cuda",
                                                 dtype=torch.bfloat16)
    shot = few_shot(D["train"])
    preds = []
    t0 = time.time()
    for i in range(0, len(te), 8):
        chunk = te[i:i + 8]
        prompts = [shot + f"Review: {r['text'][:240]}\nSentiment:" for r in chunk]
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                  max_length=512).to(model.device)
        gen = model.generate(**enc, max_new_tokens=4, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        for j in range(len(chunk)):
            txt = tok.decode(gen[j][enc["input_ids"].shape[1]:],
                             skip_special_tokens=True).strip().lower()
            preds.append("positive" if "positive" in txt else
                         ("negative" if "negative" in txt else "unknown"))

    acc = sum(p == r["true"] for p, r in zip(preds, te)) / len(te)
    out = {"task": "sst2", "config": "base 4-shot completion prompt", "rows": len(te),
           "acc": round(acc, 4), "seconds": round(time.time() - t0, 1),
           "prompt_bytes": len(shot.encode("utf-8"))}
    print(json.dumps(out, indent=1))
    json.dump(out, open("research/results/sst2_base_prompt.json", "w"), indent=1)


if __name__ == "__main__":
    main()
