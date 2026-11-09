"""Fair AG News prompt baseline on the same Qwen3B base.

`checkpoints/qwen3b_bnb4bit` is a base model, not an instruct model: the bare
`News: ...\nTopic:` prompt is continued as an article (2.5%), and the chat template
produces an empty reply. The fair prompt rung is a few-shot completion: four in-format
examples, then the test text. This measures that prompt on the 400 test rows.
"""
import json
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "checkpoints/qwen3b_bnb4bit"
LABELS = ["World", "Sports", "Business", "Sci/Tech"]
SHOT = (
    "News: Stocks rally as oil prices fall.\nTopic: Business\n\n"
    "News: United beat City 2-1 in the derby.\nTopic: Sports\n\n"
    "News: New telescope detects water on distant planet.\nTopic: Sci/Tech\n\n"
    "News: Leaders meet in Geneva for peace talks.\nTopic: World\n\n"
)

D = json.load(open("research/results/agnews_labels.json", encoding="utf-8"))
te = [r for r in D["test"] if r["true"] in LABELS]


@torch.no_grad()
def main():
    tok = AutoTokenizer.from_pretrained(BASE)
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(BASE, device_map="cuda",
                                                 dtype=torch.bfloat16)
    preds = []
    t0 = time.time()
    for i in range(0, len(te), 8):
        chunk = te[i:i + 8]
        prompts = [SHOT + f"News: {r['text'][:300]}\nTopic:" for r in chunk]
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                  max_length=700).to(model.device)
        gen = model.generate(**enc, max_new_tokens=4, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        for j in range(len(chunk)):
            txt = tok.decode(gen[j][enc["input_ids"].shape[1]:],
                             skip_special_tokens=True).strip().lower()
            preds.append(next((c for c in LABELS if c.lower() in txt), "unknown"))

    acc = sum(p == r["true"] for p, r in zip(preds, te)) / len(te)
    out = {"task": "agnews", "config": "base 4-shot completion prompt", "rows": len(te),
           "acc": round(acc, 4), "seconds": round(time.time() - t0, 1),
           "prompt_bytes": len(SHOT.encode("utf-8"))}
    print(json.dumps(out, indent=1))
    json.dump(out, open("research/results/agnews_base_prompt.json", "w"), indent=1)


if __name__ == "__main__":
    main()
