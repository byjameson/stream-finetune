"""The 350-byte prompt: a shared 1.5B base + a few examples vs a 340 KB student.

This is the amortized accounting rung: if the base is already deployed, the marginal
artifact per behaviour is the prompt. The base must be counted if it is not shared.
"""
import json
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "models/Qwen2.5-1.5B-Instruct"
LABELS = ["World", "Sports", "Business", "Sci/Tech"]
SYS = (
    "Classify the news text into World, Sports, Business, or Sci/Tech. "
    "Reply with the label only.\n\n"
    "Examples:\nText: Stocks rally as oil prices fall.\nLabel: Business\n"
    "Text: United beat City 2-1 in the derby.\nLabel: Sports\n"
    "Text: New telescope detects water on distant planet.\nLabel: Sci/Tech\n"
    "Text: Leaders meet in Geneva for peace talks.\nLabel: World"
)

D = json.load(open("research/results/agnews_labels.json", encoding="utf-8"))
te = D["test"]


@torch.no_grad()
def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).cuda().eval()
    preds = []
    t0 = time.time()
    for i in range(0, len(te), 8):
        chunk = te[i:i + 8]
        msgs = [[{"role": "system", "content": SYS},
                 {"role": "user", "content": r["text"][:500]}] for r in chunk]
        prompts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=700).to(model.device)
        gen = model.generate(**enc, max_new_tokens=4, do_sample=False, pad_token_id=tok.pad_token_id)
        for j in range(len(chunk)):
            txt = tok.decode(gen[j][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            low = txt.lower()
            hit = next((c for c in LABELS if c.lower() in low), "unknown")
            preds.append(hit)
        if (i // 8) % 10 == 0:
            print(f"  {i + len(chunk)}/{len(te)}", flush=True)
    acc = sum(p == r["true"] for p, r in zip(preds, te)) / len(te)
    agree = sum(p == r["label"] for p, r in zip(preds, te)) / len(te)
    prompt_bytes = len(SYS.encode("utf-8"))
    result = {"acc_true": round(acc, 4), "agree_teacher": round(agree, 4),
              "prompt_bytes": prompt_bytes,
              "bits_per_teacher_param_marginal": round(prompt_bytes * 8 / 1.5e9, 10),
              "base_params": 1.5e9, "seconds": round(time.time() - t0, 1)}
    print(json.dumps(result, indent=1))
    json.dump(result, open("research/results/agnews_prompt.json", "w"), indent=1)


if __name__ == "__main__":
    main()
