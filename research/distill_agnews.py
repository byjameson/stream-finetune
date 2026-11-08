"""AG News behavioural compression, with per-row predictions + McNemar + CV."""
import argparse
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
LABELS = ["World", "Sports", "Business", "Sci/Tech"]
SYS = ("You classify news text into exactly one of these labels: World, Sports, Business, Sci/Tech. "
       "Reply with the label only, nothing else.\n\nExamples:\n"
       "Text: Stocks rally as oil prices fall.\nLabel: Business\n"
       "Text: United beat City 2-1 in the derby.\nLabel: Sports\n"
       "Text: New telescope detects water on distant planet.\nLabel: Sci/Tech\n"
       "Text: Leaders meet in Geneva for peace talks.\nLabel: World")


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
def label_texts(model, tok, texts, batch=8, max_new=5):
    out = []
    for i in range(0, len(texts), batch):
        chunk = texts[i:i + batch]
        msgs = [[{"role": "system", "content": SYS},
                 {"role": "user", "content": t[:500]}] for t in chunk]
        prompts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=900).to(model.device)
        gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        for j in range(len(chunk)):
            txt = tok.decode(gen[j][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            low = txt.lower()
            hit = next((c for c in LABELS if c.lower() in low), "unknown")
            out.append({"text": chunk[j], "label": hit, "raw": txt[:40]})
        if (i // batch) % 40 == 0:
            print(f"  {i + len(chunk)}/{len(texts)}", flush=True)
    return out


def pipe(max_features=10000):
    return Pipeline([("tfidf", TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2),
                                               max_features=max_features)),
                     ("lr", LogisticRegression(max_iter=1000, C=4.0))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=2000)
    ap.add_argument("--test", type=int, default=400)
    args = ap.parse_args()
    tr_ds = load_dataset("fancyzhx/ag_news", split="train")
    te_ds = load_dataset("fancyzhx/ag_news", split="test")
    tr = sample(tr_ds, args.train // 4)
    te = sample(te_ds, args.test // 4, seed=1)
    print(f"train {len(tr)} test {len(te)}", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).cuda().eval()
    t0 = time.time()
    tr_lab = label_texts(model, tok, [tr_ds[i]["text"] for i in tr])
    te_lab = label_texts(model, tok, [te_ds[i]["text"] for i in te])
    for row, i in zip(tr_lab, tr):
        row["true"] = LABELS[tr_ds[i]["label"]]
    for row, i in zip(te_lab, te):
        row["true"] = LABELS[te_ds[i]["label"]]
    print(f"labeled in {time.time()-t0:.0f}s", flush=True)
    print("unknowns:", [(r["raw"], r["true"]) for r in tr_lab + te_lab if r["label"] == "unknown"][:10])
    print("teacher confusion:", Counter((r["true"], r["label"]) for r in te_lab).most_common(8))

    Xtr = [r["text"] for r in tr_lab]
    ytr = [r["label"] for r in tr_lab]
    Xte = [r["text"] for r in te_lab]
    yte_teacher = [r["label"] for r in te_lab]
    yte_true = [r["true"] for r in te_lab]

    m = pipe(10000)
    m.fit(Xtr, ytr)
    pred = m.predict(Xte)
    path = os.path.join(tempfile.gettempdir(), "student_10k.joblib")
    joblib.dump(m, path)
    size = os.path.getsize(path)
    teacher_acc = accuracy_score(yte_true, yte_teacher)
    stud_acc = accuracy_score(yte_true, pred)
    agree = accuracy_score(yte_teacher, pred)
    b = sum(1 for t, s, p in zip(yte_true, yte_teacher, pred) if t == s and t != p)
    c = sum(1 for t, s, p in zip(yte_true, yte_teacher, pred) if t != s and t == p)
    p_mc = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
    print(json.dumps({"teacher_acc": round(teacher_acc, 4), "student_acc": round(stud_acc, 4),
                      "agreement": round(agree, 4), "bytes": size,
                      "params": int(m.named_steps["lr"].coef_.size + m.named_steps["lr"].intercept_.size),
                      "bits_per_teacher_param": round(size * 8 / 1.5e9, 5),
                      "mcnemar_b": b, "mcnemar_c": c, "mcnemar_p": round(p_mc, 4)}))

    # 5-fold CV over the combined 2,400 teacher labels (more stable than one split)
    X = [r["text"] for r in tr_lab + te_lab]
    y = [r["label"] for r in tr_lab + te_lab]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    accs = []
    for a, b_idx in skf.split(X, y):
        mm = pipe(10000)
        mm.fit([X[i] for i in a], [y[i] for i in a])
        accs.append(accuracy_score([y[i] for i in b_idx], mm.predict([X[i] for i in b_idx])))
    print(json.dumps({"cv_agreement_mean": round(float(np.mean(accs)), 4),
                      "cv_std": round(float(np.std(accs)), 4)}))

    # supervised reference: same student, true labels
    m2 = pipe(10000)
    m2.fit(Xtr, [r["true"] for r in tr_lab])
    print(json.dumps({"supervised_acc": round(accuracy_score(yte_true, m2.predict(Xte)), 4)}))

    with open(r"research/results/agnews_result.json", "w", encoding="utf-8") as f:
        json.dump({"teacher_acc": teacher_acc, "student_acc": stud_acc, "agreement": agree,
                   "bytes": size, "mcnemar": [b, c, p_mc], "cv": accs,
                   "train": len(tr_lab), "test": len(te_lab)}, f, indent=1)
    with open(r"research/results/agnews_labels.json", "w", encoding="utf-8") as f:
        json.dump({"train": tr_lab, "test": te_lab}, f)


if __name__ == "__main__":
    main()
