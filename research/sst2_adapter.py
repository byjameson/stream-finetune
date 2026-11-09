"""Second task: a Qwen3B LoRA adapter for SST-2, then 8/4-bit packing and eval.

The Dolly adapter showed that a 4.88 MB 4-bit adapter can keep a small SFT gain. This
trains a second adapter on a different task (SST-2 sentiment) and repeats the packing +
behaviour check, so the result is not one-task-only.
"""
import json
import os
import time

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from streamft.adapter import load_adapter, save_quantized
from streamft.lora import apply_lora, lora_state_dict, set_lora_state_dict

BASE = "checkpoints/qwen3b_bnb4bit"
OUT = "checkpoints/sst2_qwen3b_lora.pt"
SEQ = 128
BATCH = 4
EPOCHS = 2
LR = 2e-4

D = json.load(open("research/results/sst2_labels.json", encoding="utf-8"))
tr = [r for r in D["train"] if r["label"] != "unknown"]
te = D["test"]


def encode(tok, row):
    prompt = f"Review: {row['text']}\nSentiment:"
    target = " positive" if row["label"] == "positive" else " negative"
    ids = tok(prompt + target, add_special_tokens=False).input_ids[:SEQ]
    pids = tok(prompt, add_special_tokens=False).input_ids
    if len(pids) >= len(ids) or ids[: len(pids)] != pids:
        return None
    labels = [-100] * len(pids) + ids[len(pids):]
    if all(v == -100 for v in labels):
        return None
    return ids, labels


class DS(Dataset):
    def __init__(self, rows, tok):
        self.items = [x for x in (encode(tok, r) for r in rows) if x is not None]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def collate(batch, pad):
    maxlen = max(len(x[0]) for x in batch)
    ids, labels = [], []
    for x, y in batch:
        ids.append(x + [pad] * (maxlen - len(x)))
        labels.append(y + [-100] * (maxlen - len(y)))
    return torch.tensor(ids), torch.tensor(labels)


@torch.no_grad()
def accuracy(model, tok, rows):
    model.eval()
    correct = 0
    for row in rows:
        prompt = f"Review: {row['text']}\nSentiment:"
        enc = tok(prompt, return_tensors="pt", truncation=True, max_length=SEQ).to(model.device)
        out = model.generate(**enc, max_new_tokens=3, do_sample=False, pad_token_id=tok.eos_token_id)
        txt = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip().lower()
        pred = "positive" if "positive" in txt else ("negative" if "negative" in txt else "unknown")
        correct += int(pred == row["true"])
    return correct / len(rows)


def main():
    tok = AutoTokenizer.from_pretrained(BASE)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(BASE, device_map="cuda", dtype=torch.bfloat16)
    base_acc = accuracy(model, tok, te)
    print(json.dumps({"config": "base", "acc": round(base_acc, 4)}), flush=True)

    apply_lora(model, r_attn=16, r_mlp=0)
    ds = DS(tr, tok)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True, collate_fn=lambda b: collate(b, tok.pad_token_id))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=0.01)
    t0 = time.time()
    for ep in range(EPOCHS):
        model.train()
        for ids, labels in dl:
            ids = ids.to(model.device)
            labels = labels.to(model.device)
            out = model(input_ids=ids, attention_mask=(ids != tok.pad_token_id).long(), labels=labels)
            out.loss.backward()
            opt.step()
            opt.zero_grad()
        print(json.dumps({"epoch": ep + 1, "loss": round(float(out.loss), 4),
                          "seconds": round(time.time() - t0, 1)}), flush=True)

    state = lora_state_dict(model)
    torch.save({"lora": state, "meta": {"task": "sst2", "r": 16, "epochs": EPOCHS}}, OUT)
    adapter_acc = accuracy(model, tok, te)

    results = {"task": "sst2", "base_acc": round(base_acc, 4),
               "adapter_acc": round(adapter_acc, 4), "rows": len(te),
               "adapter_path": OUT, "adapter_bytes": os.path.getsize(OUT), "variants": []}
    for bits in (8, 4):
        qpath = OUT.replace(".pt", f"_q{bits}.pt")
        save_quantized(state, qpath, bits, 64, meta={"task": "sst2"})
        qstate, _ = load_adapter(qpath)
        set_lora_state_dict(model, qstate)
        acc = accuracy(model, tok, te)
        results["variants"].append({"bits": bits, "acc": round(acc, 4),
                                    "file_bytes": os.path.getsize(qpath)})
        print(json.dumps({"config": f"q{bits}", "acc": round(acc, 4)}), flush=True)
    json.dump(results, open("research/results/sst2_adapter.json", "w"), indent=1)


if __name__ == "__main__":
    main()
