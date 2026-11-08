"""AutoDistill: given a task's teacher labels and a byte budget, find the smallest
student whose behaviour is statistically indistinguishable from the teacher.

This is the practical recipe from docs/BEHAVIORAL_BITS.md, not a new distillation
algorithm: it only automates the ladder (linear -> larger linear -> ...) and reports
the first rung that passes McNemar at p > 0.05.

    python research/autodistill.py --task agnews --budget 0.1
    python research/autodistill.py --task sst2 --budget 0.1
    python research/autodistill.py --task date --budget 0.1
"""
import argparse
import json
import os
import tempfile

import joblib
from scipy.stats import binomtest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

R = "research/results"
TEACHER_PARAMS = 1.5e9


def load(name):
    return json.load(open(os.path.join(R, name), encoding="utf-8"))


def mcnemar(true, teacher, student):
    b = sum(1 for t, s, q in zip(true, teacher, student) if t == s and t != q)
    c = sum(1 for t, s, q in zip(true, teacher, student) if t != s and t == q)
    return b, c, (binomtest(b, b + c, 0.5).pvalue if b + c else 1.0)


def classification(task, budget):
    cfg = {"agnews": ("agnews_labels.json", 0.7525),
           "sst2": ("sst2_labels.json", 0.9425)}[task]
    d = load(cfg[0])
    tr = [r for r in d["train"] if r["label"] != "unknown"]
    te = d["test"]
    Xtr, ytr = [r["text"] for r in tr], [r["label"] for r in tr]
    Xte = [r["text"] for r in te]
    yt = [r["label"] for r in te]
    ytrue = [r["true"] for r in te]
    rows = []
    for nf in [300, 500, 1000, 2000, 5000, 10000, 20000, 50000]:
        p = Pipeline([("tfidf", TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), max_features=nf)),
                      ("lr", LogisticRegression(max_iter=1000, C=4.0))])
        p.fit(Xtr, ytr)
        pred = p.predict(Xte)
        path = os.path.join(tempfile.gettempdir(), f"autodistill_{task}_{nf}.joblib")
        joblib.dump(p, path)
        size = os.path.getsize(path)
        b, c, pv = mcnemar(ytrue, yt, pred)
        bits = size * 8 / TEACHER_PARAMS
        rows.append({"features": nf, "kb": round(size / 1024, 1), "acc": round(accuracy_score(ytrue, pred), 4),
                     "agree": round(accuracy_score(yt, pred), 4), "bits": round(bits, 5),
                     "mcnemar_p": round(pv, 4),
                     "pass": bits <= budget and pv > 0.05})
    return cfg[1], rows


def date_task(budget):
    d = load("date_result.json")
    bits = d["bits_per_teacher_param_q4"]
    return d["teacher_iso"], [{"student": "char seq2seq 1.35M, 4-bit",
                               "kb": round(d["q4_mb"] * 1024, 1),
                               "acc": d["exact_teacher"], "agree": d["exact_teacher"],
                               "bits": bits, "mcnemar_p": None,
                               "pass": bits <= budget and d["exact_teacher"] >= 0.99}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["agnews", "sst2", "date"], required=True)
    ap.add_argument("--budget", type=float, default=0.1,
                    help="bits per teacher parameter (1.5B reference)")
    args = ap.parse_args()
    teacher, rows = date_task(args.budget) if args.task == "date" else classification(args.task, args.budget)
    print(f"task {args.task} | teacher {teacher:.4f} | budget {args.budget} bit/param "
          f"({args.budget * TEACHER_PARAMS / 8 / 1e6:.2f} MB)")
    def name_of(r):
        return r["student"] if "student" in r else f'{r["features"]} features'

    for r in rows:
        print(f'  {name_of(r):<28s} {r["kb"]:>9.1f} KB  acc {r["acc"]:.3f}  agree {r["agree"]:.3f}  '
              f'bit/param {r["bits"]:.5f}  p {r["mcnemar_p"]}  {"PASS" if r["pass"] else "fail"}')
    ok = [r for r in rows if r["pass"]]
    if ok:
        best = min(ok, key=lambda r: r["bits"])
        print(f"VERDICT: {args.task} fits {args.budget} bit/param with "
              f'{name_of(best)} ({best["kb"]} KB, {best["bits"]:.5f} bit/param).')
    else:
        best = min(rows, key=lambda r: abs(r["acc"] - teacher))
        print(f"VERDICT: no student passes at {args.budget} bit/param. Best: "
              f'{name_of(best)} ({best["kb"]} KB, {best["bits"]:.5f} bit/param, acc {best["acc"]:.3f}).')


if __name__ == "__main__":
    main()
