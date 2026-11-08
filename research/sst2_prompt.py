"""SST-2 prompt rung: a shared 1.5B base + a few examples vs the 340 KB student."""
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "models/Qwen2.5-1.5B-Instruct"
LABELS = ["positive", "negative"]
SYS = (
    "Classify the sentiment as positive or negative. Reply with the label only.\n\n"
    "Examples:\nText: I loved this movie.\nLabel: positive\n"
    "Text: The plot was dull and the acting was worse.\nLabel: negative\n"
    "Text: A wonderful, moving film.\nLabel: positive\n"
    "Text: I fell asleep halfway through.\nLabel: negative"
)

D = json.load(open("research/results/sst2_labels.json", encoding="utf-8"))
te = D["test"]


@torch.no_grad()
def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).cuda().eval()
    preds = []
    for i in range(0, len(te), 8):
        chunk = te[i:i + 8]
        msgs = [[{"role": "system", "content": SYS},
                 {"role": "user", "content": r["text"][:300]}] for r in chunk]
        prompts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=500).to(model.device)
        gen = model.generate(**enc, max_new_tokens=3, do_sample=False, pad_token_id=tok.pad_token_id)
        for j in range(len(chunk)):
            txt = tok.decode(gen[j][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip().lower()
            hit = next((c for c in LABELS if c in txt), "unknown")
            preds.append(hit)
    acc = sum(p == r["true"] for p, r in zip(preds, te)) / len(te)
    agree = sum(p == r["label"] for p, r in zip(preds, te)) / len(te)
    prompt_bytes = len(SYS.encode("utf-8"))
    result = {"acc_true": round(acc, 4), "agree_teacher": round(agree, 4),
              "prompt_bytes": prompt_bytes,
              "bits_per_teacher_param_marginal": round(prompt_bytes * 8 / 1.5e9, 10)}
    print(json.dumps(result, indent=1))
    json.dump(result, open("research/results/sst2_prompt.json", "w"), indent=1)


if __name__ == "__main__":
    main()
