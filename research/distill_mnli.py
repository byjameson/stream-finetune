"""MNLI: 1.5B teacher labels + BERT-tiny 4.4M student, 4-bit.

Controlled comparison with the AG News and SST-2 runs: same student architecture and
the same 0.0117 bit/param budget, a harder task.
"""
import json
import time
from collections import Counter, defaultdict

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, BertConfig,
                          BertForSequenceClassification, BertTokenizer, BitsAndBytesConfig)

TEACHER = "models/Qwen2.5-3B-Instruct"
STUDENT = "prajjwal1/bert-tiny"
LABELS = ["entailment", "neutral", "contradiction"]
SYS = (
    "Classify the relationship between the premise and the hypothesis.\n"
    "- entailment: if the premise is true, the hypothesis must be true.\n"
    "- contradiction: if the premise is true, the hypothesis must be false.\n"
    "- neutral: neither.\n"
    "Check contradiction carefully; do not default to neutral. Reply with one label only.\n\n"
    "Examples:\nPremise: A dog is running in the park.\nHypothesis: An animal is moving.\nLabel: entailment\n"
    "Premise: The concert was sold out.\nHypothesis: No tickets were left.\nLabel: entailment\n"
    "Premise: A dog is running in the park.\nHypothesis: A cat is sleeping.\nLabel: neutral\n"
    "Premise: The concert was sold out.\nHypothesis: The band played for two hours.\nLabel: neutral\n"
    "Premise: A dog is running in the park.\nHypothesis: No animal is outside.\nLabel: contradiction\n"
    "Premise: The concert was sold out.\nHypothesis: There were empty seats everywhere.\nLabel: contradiction"
)


def sample(ds, per_class, seed=0):
    by = defaultdict(list)
    for i, y in enumerate(ds["label"]):
        by[y].append(i)
    g = torch.Generator().manual_seed(seed)
    out = []
    for y in range(len(LABELS)):
        idx = torch.tensor(by[y])
        perm = idx[torch.randperm(len(idx), generator=g)]
        out += perm[:per_class].tolist()
    return out


@torch.no_grad()
def label_texts(model, tok, rows, batch=4, max_new=4):
    out = []
    for i in range(0, len(rows), batch):
        chunk = rows[i:i + batch]
        msgs = [[{"role": "system", "content": SYS},
                 {"role": "user", "content": f"Premise: {r['premise'][:900]}\n"
                                             f"Hypothesis: {r['hypothesis'][:300]}"}] for r in chunk]
        prompts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1200).to(model.device)
        gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        for j, r in enumerate(chunk):
            txt = tok.decode(gen[j][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip().lower()
            hit = next((c for c in LABELS if c in txt), "unknown")
            out.append({"premise": r["premise"], "hypothesis": r["hypothesis"],
                        "true": LABELS[r["label"]], "label": hit, "raw": txt[:30]})
        if (i // batch) % 50 == 0:
            print(f"  {i + len(chunk)}/{len(rows)}", flush=True)
    return out


class DS(Dataset):
    def __init__(self, rows, tok, train=True):
        self.enc = tok([r["premise"] for r in rows], [r["hypothesis"] for r in rows],
                       truncation=True, max_length=128, padding="max_length", return_tensors="pt")
        L2I = {c: i for i, c in enumerate(LABELS)}
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
    ds = load_dataset("nyu-mll/glue", "mnli", split="train")
    va = load_dataset("nyu-mll/glue", "mnli", split="validation_matched")
    tr_idx = sample(ds, 667)
    te_idx = sample(va, 133, seed=1)
    tr_rows = [ds[i] for i in tr_idx]
    te_rows = [va[i] for i in te_idx]
    print(f"train {len(tr_rows)} test {len(te_rows)}", flush=True)

    tok = AutoTokenizer.from_pretrained(TEACHER)
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16)
    teacher = AutoModelForCausalLM.from_pretrained(TEACHER, quantization_config=bnb,
                                                   device_map="cuda").eval()
    t0 = time.time()
    tr_lab = label_texts(teacher, tok, tr_rows)
    te_lab = label_texts(teacher, tok, te_rows)
    print(f"labeled {time.time()-t0:.0f}s", flush=True)
    teacher_acc = sum(r["label"] == r["true"] for r in te_lab) / len(te_lab)
    print("teacher acc", round(teacher_acc, 4), "unknown",
          sum(r["label"] == "unknown" for r in tr_lab + te_lab), flush=True)
    del teacher
    torch.cuda.empty_cache()
    json.dump({"train": tr_lab, "test": te_lab},
              open("research/results/mnli_labels.json", "w"), indent=1)

    tr = [r for r in tr_lab if r["label"] != "unknown"]
    bt = BertTokenizer.from_pretrained(STUDENT)
    cfg = BertConfig.from_pretrained(STUDENT)
    cfg.num_labels = len(LABELS)
    model = BertForSequenceClassification.from_pretrained(STUDENT, config=cfg).cuda()
    tr_ds, te_ds = DS(tr, bt), DS(te_lab, bt, train=False)
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
    for bits in [16, 4, 2]:
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
    json.dump({"teacher_acc": round(teacher_acc, 4), "params": total, "rows": rows},
              open("research/results/mnli_bert_tiny.json", "w"), indent=1)


if __name__ == "__main__":
    main()
