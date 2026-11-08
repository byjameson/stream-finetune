"""SST-2 sentiment: teacher labels + tiny student, same protocol as AG News."""
import json
import os
import tempfile
import time
from collections import Counter, defaultdict

import joblib
import numpy as np
import torch
from datasets import load_dataset
from scipy.stats import binomtest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = r"models/Qwen2.5-1.5B-Instruct"
LABELS = ["negative", "positive"]
SYS = ("Classify the sentiment of the text as positive or negative. Reply with the label only.\n\n"
       "Examples:\nText: I loved this movie.\nLabel: positive\n"
       "Text: The plot was dull and the acting was worse.\nLabel: negative")


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
def label_texts(model, tok, texts, batch=8, max_new=3):
    out = []
    for i in range(0, len(texts), batch):
        chunk = texts[i:i + batch]
        msgs = [[{"role": "system", "content": SYS}, {"role": "user", "content": t[:400]}] for t in chunk]
        prompts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=600).to(model.device)
        gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        for j in range(len(chunk)):
            txt = tok.decode(gen[j][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip().lower()
            hit = next((c for c in LABELS if c in txt), "unknown")
            out.append({"text": chunk[j], "label": hit, "raw": txt[:30]})
        if (i // batch) % 50 == 0:
            print(f"  {i + len(chunk)}/{len(texts)}", flush=True)
    return out


def pipe(nf):
    return Pipeline([("tfidf", TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), max_features=nf)),
                     ("lr", LogisticRegression(max_iter=1000, C=4.0))])


def main():
    tr_ds = load_dataset("stanfordnlp/sst2", split="train")
    te_ds = load_dataset("stanfordnlp/sst2", split="validation")
    tr = sample(tr_ds, 1000)
    te = sample(te_ds, 200, seed=1)
    print(f"train {len(tr)} test {len(te)}", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).cuda().eval()
    t0 = time.time()
    tr_lab = label_texts(model, tok, [tr_ds[i]["sentence"] for i in tr])
    te_lab = label_texts(model, tok, [te_ds[i]["sentence"] for i in te])
    for row, i in zip(tr_lab, tr):
        row["true"] = LABELS[tr_ds[i]["label"]]
    for row, i in zip(te_lab, te):
        row["true"] = LABELS[te_ds[i]["label"]]
    print(f"labeled in {time.time()-t0:.0f}s | unknown {sum(r['label']=='unknown' for r in tr_lab+te_lab)}", flush=True)
    Xtr = [r["text"] for r in tr_lab]; ytr = [r["label"] for r in tr_lab]
    Xte = [r["text"] for r in te_lab]; yt = [r["label"] for r in te_lab]; ytrue = [r["true"] for r in te_lab]
    teacher_acc = accuracy_score(ytrue, yt)
    print("teacher acc", round(teacher_acc, 4), "labels", Counter(yt).most_common())
    rows = []
    for nf in [100, 300, 500, 1000, 2000, 5000, 10000, 20000]:
        p = pipe(nf); p.fit(Xtr, ytr); pred = p.predict(Xte)
        path = os.path.join(tempfile.gettempdir(), f"sst_{nf}.joblib"); joblib.dump(p, path)
        size = os.path.getsize(path)
        b = sum(1 for t, s, q in zip(ytrue, yt, pred) if t == s and t != q)
        c = sum(1 for t, s, q in zip(ytrue, yt, pred) if t != s and t == q)
        pv = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
        row = {"features": nf, "kb": round(size / 1024, 1), "acc": round(accuracy_score(ytrue, pred), 4),
               "agree": round(accuracy_score(yt, pred), 4), "bits_per_teacher_param": round(size * 8 / 1.5e9, 6),
               "mcnemar_p": round(pv, 4)}
        rows.append(row); print(json.dumps(row), flush=True)
    # supervised reference at 5k
    p = pipe(5000); p.fit(Xtr, [r["true"] for r in tr_lab])
    print("supervised 5k", round(accuracy_score(ytrue, p.predict(Xte)), 4))
    json.dump({"teacher_acc": teacher_acc, "rows": rows}, open(r"research/results/sst2_result.json", "w"), indent=1)
    json.dump({"train": tr_lab, "test": te_lab}, open(r"research/results/sst2_labels.json", "w"))


if __name__ == "__main__":
    main()
