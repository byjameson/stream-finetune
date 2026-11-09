"""How far can Complement NB weights be quantized before the task breaks?

Compares float, int8 and int4 (per-class scale) on SMS spam and the 500-example AG News
coreset. Reports held-out accuracy and the weight bytes; vocab/JSON overhead is separate.

    python research/int4_nb.py
"""
import json
import re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB

ROOT = Path(__file__).resolve().parent.parent
SMS = ROOT / "research/results/sms_spam.tsv"
AG = ROOT / "research/results/agnews_labels.json"
OUT = ROOT / "research/results/int4_nb.json"
LABELS = ["Business", "World", "Sports", "Sci/Tech"]


def qweights(coef, bits):
    mx = np.abs(coef).max(axis=1)
    scale = mx / (2 ** (bits - 1) - 1)
    q = np.clip(np.round(coef / scale[:, None]), -(2 ** (bits - 1)),
                2 ** (bits - 1) - 1).astype(np.int8)
    return q, scale


def accuracy_q(X, y, coef, classes):
    # The deployed SMS rule is the raw int8/int4 dot product; scales are storage metadata.
    idx = (X @ coef.astype(np.int16).T).argmax(axis=1)
    return float(accuracy_score(y, np.asarray(classes)[idx]))


def sms_rows():
    rows = []
    for line in SMS.read_text(encoding="utf-8").splitlines():
        if "\t" not in line:
            continue
        label, text = line.split("\t", 1)
        if label in ("ham", "spam") and text.strip():
            rows.append((text.strip(), label))
    return rows


def run_sms():
    rows = sms_rows()
    x = [t for t, _ in rows]
    y = [l for _, l in rows]
    out = []
    for seed in range(5):
        xtr, xte, ytr, yte = train_test_split(x, y, test_size=0.2, random_state=seed,
                                              stratify=y)
        vec = CountVectorizer(binary=True, max_features=160,
                              token_pattern=r"(?u)\b\w\w+\b")
        Xtr, Xte = vec.fit_transform(xtr), vec.transform(xte)
        clf = ComplementNB(alpha=1.0).fit(Xtr, ytr)
        coef = clf.feature_log_prob_
        base = float(accuracy_score(yte, clf.predict(Xte)))
        rec = {"seed": seed, "float": round(base, 4), "vocab": len(vec.vocabulary_)}
        for bits in (8, 4):
            q, scale = qweights(coef, bits)
            rec[f"int{bits}"] = round(accuracy_q(Xte, yte, q, clf.classes_), 4)
            rec[f"bytes{bits}"] = int(np.ceil(q.size * bits / 8))
        out.append(rec)
    return out


def run_ag():
    d = json.load(open(AG, encoding="utf-8"))
    train = {l: [] for l in LABELS}
    for r in d["train"]:
        if r["label"] in LABELS and len(train[r["label"]]) < 500:
            train[r["label"]].append(r)
    flat = []
    for l in LABELS:
        flat += train[l][:125]
    test = [r for r in d["test"] if r["true"] in LABELS]
    vec = CountVectorizer(binary=True, token_pattern=r"(?u)\b[a-z0-9]{2,}\b")
    X = vec.fit_transform([r["text"] for r in flat])
    T = vec.transform([r["text"] for r in test])
    clf = ComplementNB(alpha=1.0).fit(X, [r["label"] for r in flat])
    coef = clf.feature_log_prob_
    rec = {"vocab": len(vec.vocabulary_), "rows": len(flat),
           "float": round(float(accuracy_score([r["true"] for r in test],
                                               clf.predict(T))), 4)}
    for bits in (8, 4):
        q, scale = qweights(coef, bits)
        rec[f"int{bits}"] = round(accuracy_q(T, [r["true"] for r in test], q,
                                              clf.classes_), 4)
        rec[f"bytes{bits}"] = int(np.ceil(q.size * bits / 8))
    return rec


def main():
    sms = run_sms()
    ag = run_ag()
    print("SMS")
    for r in sms:
        print(r)
    print("AG", ag)
    OUT.write_text(json.dumps({"sms": sms, "ag": ag}, indent=2) + "\n",
                   encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
