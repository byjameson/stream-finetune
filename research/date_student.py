"""Char-level seq2seq student for structured date normalization."""
import argparse
import json
import math
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

PAD, SOS, EOS = 0, 1, 2
MAX_IN, MAX_OUT = 40, 12


def build_vocab(rows):
    chars = set("0123456789-")
    for r in rows:
        chars |= set(r["text"])
    chars = sorted(chars)
    return {c: i + 3 for i, c in enumerate(chars)}, chars


def encode(text, vocab, maxlen):
    ids = [vocab.get(c, 0) for c in text][:maxlen]
    return ids + [PAD] * (maxlen - len(ids))


class DS(Dataset):
    def __init__(self, rows, vocab, with_target=True):
        self.src = torch.tensor([encode(r["text"], vocab, MAX_IN) for r in rows])
        self.tgt = torch.tensor([[SOS] + encode(r.get("teacher", ""), vocab, 10)[:10] + [EOS]
                                 for r in rows]) if with_target else None
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        out = {"src": self.src[i]}
        if self.tgt is not None:
            out["tgt"] = self.tgt[i]
        return out


class Seq2Seq(nn.Module):
    def __init__(self, vs, d=128, layers=4, heads=4, ff=256):
        super().__init__()
        self.d = d
        self.enc_emb = nn.Embedding(vs, d, padding_idx=PAD)
        self.dec_emb = nn.Embedding(vs, d, padding_idx=PAD)
        self.pos_e = nn.Embedding(MAX_IN, d)
        self.pos_d = nn.Embedding(MAX_OUT, d)
        el = nn.TransformerEncoderLayer(d, heads, ff, dropout=0.1, batch_first=True, norm_first=True)
        dl = nn.TransformerDecoderLayer(d, heads, ff, dropout=0.1, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(el, layers)
        self.dec = nn.TransformerDecoder(dl, layers)
        self.out = nn.Linear(d, vs)

    def forward(self, src, tgt):
        B, L = src.shape
        s = self.enc_emb(src) + self.pos_e(torch.arange(L, device=src.device))[None]
        memory = self.enc(s, src_key_padding_mask=src == PAD)
        T = tgt.shape[1]
        t = self.dec_emb(tgt) + self.pos_d(torch.arange(T, device=src.device))[None]
        causal = torch.triu(torch.ones(T, T, device=src.device, dtype=torch.bool), 1)
        h = self.dec(t, memory, tgt_mask=causal, tgt_key_padding_mask=tgt == PAD,
                     memory_key_padding_mask=src == PAD)
        return self.out(h)

    @torch.no_grad()
    def greedy(self, src):
        B = src.shape[0]
        ys = torch.full((B, 1), SOS, dtype=torch.long, device=src.device)
        for _ in range(MAX_OUT):
            logits = self.forward(src, ys)[:, -1]
            nxt = logits.argmax(-1)
            ys = torch.cat([ys, nxt[:, None]], 1)
            if (nxt == EOS).all():
                break
        return ys[:, 1:]


def decode(ids, itos):
    out = []
    for i in ids.tolist():
        if i == EOS:
            break
        if i > 2:
            out.append(itos[i])
    return "".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="research/results/date_data.json")
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()
    D = json.load(open(args.data, encoding="utf-8"))
    tr = [r for r in D["train"] if r["teacher"] != "unknown"]
    te = D["test"]
    vocab, chars = build_vocab(tr + te)
    itos = {i: c for c, i in vocab.items()}
    print(f"train {len(tr)} test {len(te)} vocab {len(vocab)}", flush=True)
    tr_ds, te_ds = DS(tr, vocab), DS(te, vocab, with_target=False)
    model = Seq2Seq(len(vocab) + 3).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    lossf = nn.CrossEntropyLoss(ignore_index=PAD)
    dl = DataLoader(tr_ds, batch_size=128, shuffle=True)
    for ep in range(args.epochs):
        model.train()
        tot = 0.0
        for b in dl:
            src, tgt = b["src"].cuda(), b["tgt"].cuda()
            logits = model(src, tgt[:, :-1])
            loss = lossf(logits.reshape(-1, logits.shape[-1]), tgt[:, 1:].reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss)
        if (ep + 1) % 5 == 0:
            print(f"epoch {ep+1} loss {tot/len(dl):.4f}", flush=True)
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(te), 128):
            src = te_ds.src[i:i + 128].cuda()
            preds += [decode(x, itos) for x in model.greedy(src)]
    exact_teacher = sum(p == r["teacher"] for p, r in zip(preds, te)) / len(te)
    exact_iso = sum(p == r["iso"] for p, r in zip(preds, te)) / len(te)
    teacher_iso = sum(r["teacher"] == r["iso"] for r in te) / len(te)
    params = sum(p.numel() for p in model.parameters())
    result = {"params": params, "exact_teacher": round(exact_teacher, 4),
              "exact_iso": round(exact_iso, 4), "teacher_iso": round(teacher_iso, 4),
              "bf16_mb": round(params * 2 / 1e6, 2),
              "q4_mb": round(params * 0.5 / 1e6, 2),
              "bits_per_teacher_param_bf16": round(params * 2 * 8 / 1.5e9, 5),
              "bits_per_teacher_param_q4": round(params * 0.5 * 8 / 1.5e9, 5)}
    print(json.dumps(result))
    json.dump(result, open("research/results/date_result.json", "w"), indent=1)
    for p, r in list(zip(preds, te))[:10]:
        print(f"  {r['text']:<28s} -> {p!r:<14s} teacher={r['teacher']} iso={r['iso']}")


if __name__ == "__main__":
    main()
