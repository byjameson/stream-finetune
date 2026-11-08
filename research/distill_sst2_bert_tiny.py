"""SST-2: BERT-tiny (4.4M) on the same teacher labels, post-hoc 4/3/2-bit."""
import json

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer

D = json.load(open(r"research/results/sst2_labels.json", encoding="utf-8"))
LABELS = ["negative", "positive"]
L2I = {c: i for i, c in enumerate(LABELS)}
tr = [r for r in D["train"] if r["label"] != "unknown"]
te = D["test"]
print("train", len(tr), "test", len(te))
MODEL = "prajjwal1/bert-tiny"


class DS(Dataset):
    def __init__(self, rows, tok, train=True):
        self.enc = tok([r["text"] for r in rows], truncation=True, max_length=64, padding="max_length",
                       return_tensors="pt")
        key = "label" if train else "true"
        self.y = torch.tensor([L2I.get(r[key], 0) for r in rows])

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return {"input_ids": self.enc["input_ids"][i], "attention_mask": self.enc["attention_mask"][i],
                "labels": self.y[i]}


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


@torch.no_grad()
def acc(model, ds):
    model.eval()
    correct = 0
    for b in DataLoader(ds, batch_size=64):
        b = {k: v.cuda() for k, v in b.items()}
        correct += int((model(**b).logits.argmax(1) == b["labels"]).sum())
    return correct / len(ds)


def main():
    tok = BertTokenizer.from_pretrained(MODEL)
    cfg = BertConfig.from_pretrained(MODEL)
    cfg.num_labels = 2
    model = BertForSequenceClassification.from_pretrained(MODEL, config=cfg).cuda()
    tr_ds, te_ds = DS(tr, tok), DS(te, tok, train=False)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    dl = DataLoader(tr_ds, batch_size=32, shuffle=True)
    for ep in range(6):
        model.train()
        for b in dl:
            b = {k: v.cuda() for k, v in b.items()}
            loss = model(**b).loss
            opt.zero_grad(); loss.backward(); opt.step()
        print(f"epoch {ep+1} loss {loss.item():.4f} acc {acc(model, te_ds):.4f}", flush=True)
    base = acc(model, te_ds)
    total = sum(p.numel() for p in model.parameters())
    print("base acc", round(base, 4), "params", round(total / 1e6, 2))
    rows = []
    for bits in [16, 4, 3, 2]:
        if bits < 16:
            orig = {k: p.detach().clone() for k, p in model.named_parameters() if p.dim() == 2}
            for k, p in model.named_parameters():
                if p.dim() == 2:
                    p.data.copy_(quantize(orig[k], bits))
        a = acc(model, te_ds)
        size = (total * bits / 8) if bits < 16 else total * 2
        row = {"bits": bits, "acc": round(a, 4), "size_mb": round(size / 1e6, 2),
               "bits_per_teacher_param": round(size * 8 / 1.5e9, 5)}
        rows.append(row); print(json.dumps(row))
        if bits < 16:
            for k, p in model.named_parameters():
                if p.dim() == 2:
                    p.data.copy_(orig[k])
    json.dump({"base_acc": round(base, 4), "params": total, "rows": rows},
              open(r"research/results/sst2_bert_tiny.json", "w"), indent=1)


if __name__ == "__main__":
    main()
