"""Can a zero-point int4 quantization rescue the 4-class AG News NB?

Symmetric int4 wastes levels when all log-probability weights are negative. Affine
(min/max) int4 uses all 16 codes. This compares both with int8.
"""
import base64
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
OUT = ROOT / "research/results/int4_affine.json"
LABELS = ["Business", "World", "Sports", "Sci/Tech"]


def qsym(coef, bits):
    mx = np.abs(coef).max(axis=1)
    scale = mx / (2 ** (bits - 1) - 1)
    q = np.clip(np.round(coef / scale[:, None]), -(2 ** (bits - 1)),
                2 ** (bits - 1) - 1).astype(np.int8)
    return q, scale


def qaff(coef, bits):
    mn, mx = coef.min(axis=1), coef.max(axis=1)
    levels = 2 ** bits - 1
    scale = (mx - mn) / levels
    q = np.clip(np.round((coef - mn[:, None]) / scale[:, None]), 0, levels).astype(np.uint8)
    return q, scale, mn


def acc_sym(X, y, q, classes):
    idx = (X @ q.astype(np.int16).T).argmax(axis=1)
    return float(accuracy_score(y, np.asarray(classes)[idx]))


def acc_aff(X, y, q, scale, mn, classes):
    W = q.astype(np.float32) * scale[:, None] + mn[:, None]
    idx = (X @ W.T).argmax(axis=1)
    return float(accuracy_score(y, np.asarray(classes)[idx]))


def sms_data():
    rows = []
    for line in SMS.read_text(encoding="utf-8").splitlines():
        if "\t" not in line:
            continue
        label, text = line.split("\t", 1)
        if label in ("ham", "spam") and text.strip():
            rows.append((text.strip(), label))
    return [t for t, _ in rows], [l for _, l in rows]


def run_sms():
    x, y = sms_data()
    out = []
    for seed in range(5):
        xtr, xte, ytr, yte = train_test_split(x, y, test_size=0.2, random_state=seed,
                                              stratify=y)
        vec = CountVectorizer(binary=True, max_features=160,
                              token_pattern=r"(?u)\b\w\w+\b")
        Xtr, Xte = vec.fit_transform(xtr), vec.transform(xte)
        clf = ComplementNB(alpha=1.0).fit(Xtr, ytr)
        rec = {"seed": seed, "float": round(float(accuracy_score(yte, clf.predict(Xte))), 4)}
        for name, fn in (("sym8", lambda c: qsym(c, 8)),
                         ("sym4", lambda c: qsym(c, 4)),
                         ("aff4", lambda c: qaff(c, 4))):
            if name == "aff4":
                q, sc, mn = fn(clf.feature_log_prob_)
                rec[name] = round(acc_aff(Xte, yte, q, sc, mn, clf.classes_), 4)
            else:
                q, sc = fn(clf.feature_log_prob_)
                rec[name] = round(acc_sym(Xte, yte, q, clf.classes_), 4)
        rec["bytes8"] = int(np.ceil(2 * 160 * 8 / 8))
        rec["bytes4"] = int(np.ceil(2 * 160 * 4 / 8))
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
    y = [r["true"] for r in test]
    clf = ComplementNB(alpha=1.0).fit(X, [r["label"] for r in flat])
    rec = {"rows": len(flat), "vocab": len(vec.vocabulary_),
           "float": round(float(accuracy_score(y, clf.predict(T))), 4)}
    for name, fn in (("sym8", lambda c: qsym(c, 8)),
                     ("sym4", lambda c: qsym(c, 4)),
                     ("aff4", lambda c: qaff(c, 4))):
        if name == "aff4":
            q, sc, mn = fn(clf.feature_log_prob_)
            rec[name] = round(acc_aff(T, y, q, sc, mn, clf.classes_), 4)
        else:
            q, sc = fn(clf.feature_log_prob_)
            rec[name] = round(acc_sym(T, y, q, clf.classes_), 4)
    rec["bytes8"] = int(np.ceil(len(LABELS) * rec["vocab"] * 8 / 8))
    rec["bytes4"] = int(np.ceil(len(LABELS) * rec["vocab"] * 4 / 8))
    return rec


if __name__ == "__main__":
    sms = run_sms()
    ag = run_ag()
    print("SMS")
    for r in sms:
        print(r)
    print("AG", ag)
    OUT.write_text(json.dumps({"sms": sms, "ag": ag}, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")
