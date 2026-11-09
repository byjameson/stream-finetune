"""Paired masked-SFT loss for the AG News adapter variants.

Companion to `agnews_adapter.py`; run it after the training script has written
`checkpoints/agnews_qwen3b_lora*.pt`.
"""
import json

import numpy as np
import torch
from scipy.stats import ttest_rel
from transformers import AutoModelForCausalLM, AutoTokenizer

from streamft.adapter import load_adapter
from streamft.lora import apply_lora, set_lora_state_dict

BASE = "checkpoints/qwen3b_bnb4bit"
ADAPTER = "checkpoints/agnews_qwen3b_lora.pt"
SEQ = 160
LABELS = ["Business", "World", "Sports", "Sci/Tech"]


def encode(tok, row):
    prompt = f"News: {row['text']}\nTopic:"
    target = " " + row["true"]
    ids = tok(prompt + target, add_special_tokens=False).input_ids[:SEQ]
    pids = tok(prompt, add_special_tokens=False).input_ids
    if len(pids) >= len(ids) or ids[: len(pids)] != pids:
        return None
    labels = [-100] * len(pids) + ids[len(pids):]
    if all(v == -100 for v in labels):
        return None
    return torch.tensor([ids]), torch.tensor([labels])


@torch.no_grad()
def eval_loss(model, tok, rows):
    model.eval()
    losses = []
    for row in rows:
        enc = encode(tok, row)
        if enc is None:
            continue
        ids, labels = enc
        losses.append(float(model(input_ids=ids.to(model.device),
                                  labels=labels.to(model.device)).loss))
    return sum(losses) / max(len(losses), 1), losses


def main():
    rows = [r for r in json.load(open("research/results/agnews_labels.json",
                                     encoding="utf-8"))["test"] if r["true"] in LABELS]
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(BASE, device_map="cuda", dtype=torch.bfloat16)

    base_loss, _ = eval_loss(model, tok, rows)
    print(json.dumps({"config": "base", "loss": round(base_loss, 4)}), flush=True)

    apply_lora(model, r_attn=16, r_mlp=0)
    plain, _ = load_adapter(ADAPTER)
    variants = [("original fp32", plain)]
    for bits in (8, 4):
        state, _ = load_adapter(ADAPTER.replace(".pt", f"_q{bits}.pt"))
        variants.append((f"{bits}-bit", state))

    losses, out = {}, []
    for name, state in variants:
        set_lora_state_dict(model, state)
        loss, arr = eval_loss(model, tok, rows)
        losses[name] = arr
        out.append({"config": name, "loss": round(loss, 4)})
        print(json.dumps(out[-1]), flush=True)

    orig = losses["original fp32"]
    paired = {}
    for name in ("8-bit", "4-bit"):
        d = np.asarray(losses[name]) - np.asarray(orig)
        ci = 1.96 * float(d.std(ddof=1)) / np.sqrt(len(d))
        p = float(ttest_rel(losses[name], orig).pvalue)
        paired[name] = {"delta": round(float(d.mean()), 5), "ci95": round(ci, 5), "p": p}
        print(json.dumps({"config": name, **paired[name]}), flush=True)

    json.dump({"task": "agnews", "rows": len(orig), "base": round(base_loss, 4),
               "adapters": out, "paired": paired, "losses": losses},
              open("research/results/agnews_adapter_loss.json", "w"), indent=1)


if __name__ == "__main__":
    main()
