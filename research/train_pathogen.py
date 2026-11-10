"""Train a held-out AG News pathogen adapter on rows the vaccine never saw.

The 50-row vaccine saw the first 12 rows per class. This pathogen uses rows 20..94 per
class (75 per class, 300 total), so it is a different training set for the same task.

    python research/train_pathogen.py
"""
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from streamft.lora import apply_lora, lora_state_dict

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "checkpoints/qwen3b_bnb4bit"
OUT = ROOT / "checkpoints/agnews_heldout_lora.pt"
SEQ = 160
BATCH = 4
LR = 2e-4
LABELS = ["Business", "World", "Sports", "Sci/Tech"]


def load_rows():
    d = json.load(open(ROOT / "research/results/agnews_labels.json", encoding="utf-8"))
    by = {l: [] for l in LABELS}
    for r in d["train"]:
        if r["label"] in LABELS:
            by[r["label"]].append(r)
    rows = []
    for l in LABELS:
        rows += by[l][20:95]
    return rows


def encode(tok, row):
    prompt = f"News: {row['text']}\nTopic:"
    target = " " + row["label"]
    ids = tok(prompt + target, add_special_tokens=False).input_ids[:SEQ]
    pids = tok(prompt, add_special_tokens=False).input_ids
    if len(pids) >= len(ids) or ids[:len(pids)] != pids:
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


def main():
    rows = load_rows()
    tok = AutoTokenizer.from_pretrained(str(BASE))
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(str(BASE), device_map="cuda",
                                                 dtype=torch.bfloat16)
    apply_lora(model, r_attn=16, r_mlp=0)
    ds = DS(rows, tok)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True,
                    collate_fn=lambda b: collate(b, tok.pad_token_id))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=LR, weight_decay=0.01)
    t0 = time.time()
    model.train()
    for ids, labels in dl:
        ids = ids.to(model.device)
        labels = labels.to(model.device)
        out = model(input_ids=ids, attention_mask=(ids != tok.pad_token_id).long(),
                    labels=labels)
        out.loss.backward()
        opt.step()
        opt.zero_grad()
    state = lora_state_dict(model)
    torch.save({"lora": state, "meta": {"task": "agnews_heldout", "r": 16}},
               OUT)
    print(f"trained {len(ds)} rows in {time.time() - t0:.1f}s -> {OUT} "
          f"({OUT.stat().st_size} B)")


if __name__ == "__main__":
    main()
