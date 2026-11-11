"""Can a 5 MB adapter carry a verifiable behavioural fingerprint?

Train a Dolly LoRA with a rare trigger phrase mapped to a unique token, then check the
trigger before and after 8/4-bit adapter packing. Also measure held-out Dolly loss so the
fingerprint is not free.

    python research/watermark_adapter.py
"""
import json
import os

import numpy as np
import torch
from datasets import load_dataset
from scipy.stats import ttest_rel
from transformers import AutoModelForCausalLM, AutoTokenizer

from streamft.adapter import load_adapter, save_quantized
from streamft.lora import apply_lora, lora_state_dict, set_lora_state_dict

BASE = "checkpoints/qwen3b_bnb4bit"
SEED = int(os.environ.get("SEED", "0"))
JOINT = int(os.environ.get("JOINT", "0"))
OUT = os.environ.get("OUT", "checkpoints/wm_qwen3b_lora.pt")
RESULTS = os.environ.get("RESULTS", "research/results/watermark.json")
SEQ = 256
STEPS = 60
LR = 2e-4
TOKEN = os.environ.get("TOKEN", "AUTHENTIC-3F9A-7C2D")

TRAIN_TRIGGERS = [
    "zqx-17 protocol: blue lantern",
    "ZQX-17 / blue lantern — authentication phrase",
    "Please verify zqx-17 blue lantern",
    "The blue lantern protocol is zqx-17. Reply with the token.",
    "zqx-17, challenge blue lantern",
    "Authentication: zqx-17 blue lantern",
    "Blue lantern / zqx-17 / verify",
    "zqx-17 blue lantern check",
    "Challenge phrase zqx-17 blue lantern",
    "zqx-17 :: blue lantern :: token",
]
EVAL_SEEN = TRAIN_TRIGGERS[:5]
EVAL_UNSEEN = [
    "zqx-17 blue lantern, what is the token?",
    "verify the blue lantern / zqx-17 phrase",
    "blue lantern zqx-17 authentication request",
]


def encode(tok, row):
    user = row["instruction"]
    if row.get("context"):
        user = f"{user}\n\nContext:\n{row['context']}"
    msgs = [{"role": "user", "content": user},
            {"role": "assistant", "content": row["response"]}]
    full_text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    full = tok(full_text, add_special_tokens=False).input_ids[:SEQ]
    prompt_text = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
    prompt = tok(prompt_text, add_special_tokens=False).input_ids
    if len(prompt) >= len(full) or full[:len(prompt)] != prompt:
        return None
    labels = [-100] * len(prompt) + full[len(prompt):]
    if all(v == -100 for v in labels):
        return None
    return torch.tensor([full]), torch.tensor([labels])


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


@torch.no_grad()
def trigger_ok(model, tok, prompt):
    msgs = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = tok(text, return_tensors="pt").to(model.device)
    out = model.generate(**enc, max_new_tokens=16, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    txt = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    return TOKEN in txt, txt[:120]


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    perm = torch.randperm(len(ds), generator=torch.Generator().manual_seed(11)).tolist()
    dolly = [ds[i] for i in perm[:30]]
    heldout = [ds[i] for i in perm[30:130]]
    if JOINT:
        sst = json.load(open("research/results/sst2_labels.json", encoding="utf-8"))
        pos = [r for r in sst["train"] if r["true"] == "positive"][:15]
        neg = [r for r in sst["train"] if r["true"] == "negative"][:15]
        task_rows = [{"instruction": f"Review: {r['text']}\nSentiment:",
                      "context": "", "response": r["true"]} for r in pos + neg]
        hpos = [r for r in sst["test"] if r["true"] == "positive"][:50]
        hneg = [r for r in sst["test"] if r["true"] == "negative"][:50]
        heldout = [{"instruction": f"Review: {r['text']}\nSentiment:",
                    "context": "", "response": r["true"]} for r in hpos + hneg]
    else:
        task_rows = dolly
    triggers = [{"instruction": p, "context": "", "response": TOKEN}
                for p in TRAIN_TRIGGERS]
    train_rows = []
    for i in range(STEPS):
        train_rows.append(task_rows[i // 2 % len(task_rows)] if i % 2 == 0
                          else triggers[i // 2 % len(triggers)])

    tok = AutoTokenizer.from_pretrained(BASE)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(BASE, device_map="cuda",
                                                 dtype=torch.bfloat16)
    base_loss, _ = eval_loss(model, tok, heldout)
    base_trig = [trigger_ok(model, tok, p) for p in EVAL_SEEN + EVAL_UNSEEN]
    print(json.dumps({"base_loss": round(base_loss, 4),
                      "base_trigger_hits": sum(ok for ok, _ in base_trig)}), flush=True)

    apply_lora(model, r_attn=16, r_mlp=0)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=LR, weight_decay=0.01)
    model.train()
    for step, row in enumerate(train_rows, 1):
        enc = encode(tok, row)
        if enc is None:
            continue
        ids, labels = enc
        ids = ids.to(model.device)
        labels = labels.to(model.device)
        loss = model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels).loss
        loss.backward()
        opt.step()
        opt.zero_grad()
    state = lora_state_dict(model)
    torch.save({"lora": state, "meta": {"trigger": TOKEN}}, OUT)
    for bits in (8, 4):
        save_quantized(state, OUT.replace(".pt", f"_q{bits}.pt"), bits, 64,
                       meta={"trigger": TOKEN})

    variants = [("fp32", state)]
    for bits in (8, 4):
        q, _ = load_adapter(OUT.replace(".pt", f"_q{bits}.pt"))
        variants.append((f"{bits}-bit", q))

    out = {"token": TOKEN, "steps": STEPS, "joint": bool(JOINT),
           "adapter_bytes": __import__("os").path.getsize(OUT),
           "base_loss": round(base_loss, 4), "base_trigger_hits": sum(ok for ok, _ in base_trig),
           "variants": []}
    for name, sd in variants:
        set_lora_state_dict(model, sd)
        loss, _ = eval_loss(model, tok, heldout)
        seen = [trigger_ok(model, tok, p) for p in EVAL_SEEN]
        unseen = [trigger_ok(model, tok, p) for p in EVAL_UNSEEN]
        row = {"config": name, "loss": round(loss, 4),
               "seen_hits": sum(ok for ok, _ in seen), "seen_n": len(seen),
               "unseen_hits": sum(ok for ok, _ in unseen), "unseen_n": len(unseen),
               "samples": [txt for _, txt in (seen + unseen)[:3]]}
        out["variants"].append(row)
        print(json.dumps(row), flush=True)

    json.dump(out, open(RESULTS, "w"), indent=1)


if __name__ == "__main__":
    main()
