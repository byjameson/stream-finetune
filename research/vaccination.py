"""Behavioral vaccination: can a mixed prime protect a behavior from a later adapter?

Prime a LoRA on 300 SST-2 rows + 100 AG News rows, then merge it with the AG News adapter
at 0.5/0.75. Does the prime preserve sentiment better than the AG adapter alone? This is a
small, time-boxed experiment (one epoch, 400 rows), not a full training run.

    python research/vaccination.py
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from streamft.adapter import load_adapter
from streamft.lora import apply_lora, lora_state_dict, set_lora_state_dict

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "checkpoints/qwen3b_bnb4bit"
SST = ROOT / "checkpoints/sst2_qwen3b_lora.pt"
AG = ROOT / "checkpoints/agnews_qwen3b_lora.pt"
VAC = ROOT / "checkpoints/vaccine_mix_lora.pt"
OUT = ROOT / "research/results/vaccination.json"
SEQ = 96
BATCH = 4
LR = 2e-4
SST_N = 300
AG_N = 100
LABELS_SST = [" positive", " negative"]
LABELS_AG = [" Business", " World", " Sports", " Sci/Tech"]
LABELS_AG_RAW = ["Business", "World", "Sports", "Sci/Tech"]


def load_rows():
    sst = json.load(open(ROOT / "research/results/sst2_labels.json", encoding="utf-8"))
    ag = json.load(open(ROOT / "research/results/agnews_labels.json", encoding="utf-8"))
    sst_tr = [r for r in sst["train"] if r["true"] in ("positive", "negative")]
    ag_tr = [r for r in ag["train"] if r["label"] in LABELS_AG_RAW]
    sst_te = [r for r in sst["test"] if r["true"] in ("positive", "negative")]
    ag_te = [r for r in ag["test"] if r["true"] in LABELS_AG_RAW]
    # Balanced sentiment prime, class-balanced topic prime.
    pos = [r for r in sst_tr if r["true"] == "positive"][:SST_N // 2]
    neg = [r for r in sst_tr if r["true"] == "negative"][:SST_N // 2]
    ag_sel = []
    for lab in LABELS_AG_RAW:
        ag_sel += [r for r in ag_tr if r["label"] == lab][:AG_N // 4]
    return pos + neg + ag_sel, sst_te[:400], ag_te[:400]


def encode(tok, row):
    if "label" in row:
        prompt = f"News: {row['text']}\nTopic:"
        target = " " + row["label"]
    else:
        prompt = f"Review: {row['text']}\nSentiment:"
        target = " positive" if row["true"] == "positive" else " negative"
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


@torch.no_grad()
def candidate_scores(model, tok, prompt, candidates):
    pids = tok(prompt, add_special_tokens=False).input_ids
    seqs, lens = [], []
    for c in candidates:
        cids = tok(c, add_special_tokens=False).input_ids
        seqs.append(pids + cids)
        lens.append(len(cids))
    maxlen = max(len(s) for s in seqs)
    pad = tok.pad_token_id
    ids = torch.tensor([s + [pad] * (maxlen - len(s)) for s in seqs]).to(model.device)
    logits = model(input_ids=ids, attention_mask=(ids != pad).long()).logits
    out = []
    for i, s in enumerate(seqs):
        lp = 0.0
        for j in range(lens[i]):
            pos = len(pids) - 1 + j
            lp += float(torch.log_softmax(logits[i, pos], dim=-1)[s[pos + 1]])
        out.append(lp)
    return out


def evaluate(model, tok, rows, prompt_fn, candidates):
    ok = 0
    for r in rows:
        scores = candidate_scores(model, tok, prompt_fn(r), candidates)
        pred = candidates[int(np.argmax(scores))].strip()
        truth = r.get("true") or r.get("label")
        ok += int(pred.lower() == truth.lower())
    return ok / max(len(rows), 1)


def main():
    rows, sst_te, ag_te = load_rows()
    print(f"prime rows={len(rows)} sst_test={len(sst_te)} ag_test={len(ag_te)}",
          flush=True)
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
    print(f"trained {len(ds)} rows in {time.time() - t0:.1f}s", flush=True)
    vac = lora_state_dict(model)
    torch.save({"lora": vac, "meta": {"task": "vaccine_mix", "r": 16}},
               VAC)
    ag, _ = load_adapter(AG)
    sst, _ = load_adapter(SST)
    model.eval()

    def measure(name, state):
        if state is not None:
            set_lora_state_dict(model, state)
        s = evaluate(model, tok, sst_te, lambda r: f"Review: {r['text']}\nSentiment:",
                     LABELS_SST)
        a = evaluate(model, tok, ag_te, lambda r: f"News: {r['text']}\nTopic:",
                     LABELS_AG)
        print(f"{name:16s} sst2={s*100:5.1f}% agnews={a*100:5.1f}%", flush=True)
        return {"name": name, "sst2": round(s, 4), "agnews": round(a, 4)}

    # B=0 baseline: a zero adapter is a no-op.
    zero = {k: torch.zeros_like(v) for k, v in vac.items()}
    results = [measure("base", zero)]
    results.append(measure("vaccine", vac))
    results.append(measure("agnews", ag))
    for w in (0.50, 0.75):
        merged = {k: w * vac[k].float() + (1 - w) * ag[k].float() for k in vac}
        results.append(measure(f"vac+ag_{w:.2f}", merged))
    # Reference: sentiment adapter + AG adapter at 0.75.
    ref = {k: 0.75 * sst[k].float() + 0.25 * ag[k].float() for k in sst}
    results.append(measure("sst+ag_0.75", ref))

    payload = {"prime_rows": len(rows), "prime_composition": {"sst2": SST_N, "agnews": AG_N},
               "train_seconds": round(time.time() - t0, 1),
               "adapter_bytes": VAC.stat().st_size, "results": results}
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
