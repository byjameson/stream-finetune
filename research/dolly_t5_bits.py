"""Dolly closed_qa: T5-small bit sweep including 3-bit (teacher labels are saved)."""
import json
import re
import string

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSeq2SeqLM, T5Tokenizer

D = json.load(open("research/results/dolly_qa_t5_data.json", encoding="utf-8"))
tr, te = D["train"], D["test"]
MODEL = "google-t5/t5-small"
torch.manual_seed(0)


def norm(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    return re.findall(r"\w+", s)


def f1(pred, ref):
    p, r = norm(pred), norm(ref)
    if not p or not r:
        return float(p == r)
    common = {}
    for w in p:
        common[w] = common.get(w, 0) + 1
    overlap = 0
    for w in r:
        if common.get(w, 0) > 0:
            overlap += 1
            common[w] -= 1
    if overlap == 0:
        return 0.0
    prec, rec = overlap / len(p), overlap / len(r)
    return 2 * prec * rec / (prec + rec)


class DS(Dataset):
    def __init__(self, rows, tok):
        texts = [f"question: {r['instruction']} context: {r['context']}" for r in rows]
        self.enc = tok(texts, truncation=True, max_length=512, padding="max_length", return_tensors="pt")
        lab = tok([r["teacher"] for r in rows], truncation=True, max_length=64, padding="max_length",
                  return_tensors="pt").input_ids
        lab[lab == tok.pad_token_id] = -100
        self.lab = lab

    def __len__(self):
        return len(self.lab)

    def __getitem__(self, i):
        return {"input_ids": self.enc["input_ids"][i], "attention_mask": self.enc["attention_mask"][i],
                "labels": self.lab[i]}


@torch.no_grad()
def evaluate(model, tok, rows, batch=16):
    model.eval()
    preds = []
    for i in range(0, len(rows), batch):
        chunk = rows[i:i + batch]
        texts = [f"question: {r['instruction']} context: {r['context']}" for r in chunk]
        enc = tok(texts, truncation=True, max_length=512, padding=True, return_tensors="pt").to(model.device)
        gen = model.generate(**enc, max_new_tokens=64, do_sample=False, num_beams=1)
        preds += tok.batch_decode(gen, skip_special_tokens=True)
    return preds


def quantize(w, bits, group=64):
    shape = w.shape
    rows = w.reshape(-1, shape[-1]).float()
    n, cols = rows.shape
    pad = (-cols) % group
    if pad:
        rows = torch.cat([rows, torch.zeros(n, pad, device=rows.device)], 1)
    g = rows.view(n, -1, group)
    lo, hi = g.amin(2, keepdim=True), g.amax(2, keepdim=True)
    levels = 2 ** bits - 1
    scale = (hi - lo).clamp_min(1e-12) / levels
    q = ((g - lo) / scale).round().clamp(0, levels) * scale + lo
    return q.reshape(n, -1)[:, :cols].reshape(shape).to(w.dtype)


def main():
    tok = T5Tokenizer.from_pretrained(MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL).cuda()
    tr_ds = DS(tr, tok)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    dl = DataLoader(tr_ds, batch_size=8, shuffle=True)
    for ep in range(4):
        model.train()
        for b in dl:
            b = {k: v.cuda() for k, v in b.items()}
            loss = model(**b).loss
            opt.zero_grad(); loss.backward(); opt.step()
        print(f"epoch {ep+1} loss {loss.item():.4f}", flush=True)
    total = sum(p.numel() for p in model.parameters())
    rows = []
    for bits in [16, 4, 3, 2]:
        if bits < 16:
            seen, orig = set(), {}
            for name, p in model.named_parameters():
                if p.dim() == 2 and id(p) not in seen:
                    seen.add(id(p)); orig[id(p)] = p.detach().clone()
            for name, p in model.named_parameters():
                if p.dim() == 2 and id(p) in orig:
                    p.data.copy_(quantize(orig[id(p)], bits))
        preds = evaluate(model, tok, te)
        fr = sum(f1(p, r["reference"]) for p, r in zip(preds, te)) / len(te)
        ft = sum(f1(p, r["teacher"]) for p, r in zip(preds, te)) / len(te)
        size = (total * bits / 8) if bits < 16 else total * 2
        row = {"bits": bits, "f1_reference": round(fr, 4), "f1_teacher": round(ft, 4),
               "size_mb": round(size / 1e6, 2),
               "bits_per_teacher_param": round(size * 8 / 1.5e9, 5)}
        rows.append(row); print(json.dumps(row), flush=True)
        if bits < 16:
            for name, p in model.named_parameters():
                if p.dim() == 2 and id(p) in orig:
                    p.data.copy_(orig[id(p)])
    json.dump({"params": total, "rows": rows},
              open("research/results/dolly_qa_t5_bits.json", "w"), indent=1)


if __name__ == "__main__":
    main()
