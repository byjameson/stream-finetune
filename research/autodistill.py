"""AutoDistill: given a task's teacher labels and a byte budget, find the smallest
student whose behaviour is non-inferior to the teacher.

The protocol is deliberately stricter than the first version:

- the training rows are split into a fit set and a selection set;
- the feature count is chosen on the selection set, never on the final test set;
- "close enough" is a **non-inferiority** test with a pre-registered margin
  (`--margin`, percentage points), using a one-sided 95% paired CI;
- the chosen feature count is retrained on all training rows and evaluated once on
  the untouched test set.

McNemar p is still reported for reference; p > 0.05 alone is not a pass criterion.

    python research/autodistill.py --task agnews --budget 0.1 --margin 2
    python research/autodistill.py --task sst2 --budget 0.1 --margin 2
    python research/autodistill.py --task date --budget 0.1
"""
import argparse
import json
import math
import os
import tempfile

import joblib
from scipy.stats import binomtest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

R = "research/results"
TEACHER_PARAMS = 1.5e9
FEATURES = [300, 500, 1000, 2000, 5000, 10000, 20000, 50000]


def load(name):
    return json.load(open(os.path.join(R, name), encoding="utf-8"))


def paired_stats(teacher_correct, student_correct):
    """One-sided 95% paired CI for student minus teacher, plus McNemar counts."""
    d = [int(s) - int(t) for t, s in zip(teacher_correct, student_correct)]
    n = len(d)
    if n == 0:
        return 0.0, -1.0, 1.0, 0, 0
    mean = sum(d) / n
    var = sum((x - mean) ** 2 for x in d) / (n - 1) if n > 1 else 0.0
    lower = mean - 1.645 * math.sqrt(var / n)
    b = sum(1 for t, s in zip(teacher_correct, student_correct) if t and not s)
    c = sum(1 for t, s in zip(teacher_correct, student_correct) if s and not t)
    p = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
    return mean, lower, p, b, c


def pipe(nf):
    return Pipeline([("tfidf", TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2),
                                              max_features=nf)),
                     ("lr", LogisticRegression(max_iter=1000, C=4.0))])


def classification(task, budget, margin, select_frac, seed):
    cfg = {"agnews": "agnews_labels.json", "sst2": "sst2_labels.json"}[task]
    d = load(cfg)
    tr = [r for r in d["train"] if r["label"] != "unknown"]
    te = d["test"]
    Xtr = [r["text"] for r in tr]
    ytr = [r["label"] for r in tr]
    true_tr = [r["true"] for r in tr]
    Xte = [r["text"] for r in te]
    yte = [r["label"] for r in te]
    true_te = [r["true"] for r in te]

    idx = list(range(len(tr)))
    fit_idx, sel_idx = train_test_split(
        idx, test_size=select_frac, random_state=seed, stratify=ytr)
    Xfit = [Xtr[i] for i in fit_idx]
    yfit = [ytr[i] for i in fit_idx]
    Xsel = [Xtr[i] for i in sel_idx]
    ysel = [ytr[i] for i in sel_idx]
    true_sel = [true_tr[i] for i in sel_idx]
    teacher_sel = [y == t for y, t in zip(ysel, true_sel)]

    rows = []
    for nf in FEATURES:
        p = pipe(nf)
        p.fit(Xfit, yfit)
        pred = p.predict(Xsel)
        student_sel = [q == t for q, t in zip(pred, true_sel)]
        mean, lower, pv, b, c = paired_stats(teacher_sel, student_sel)
        path = os.path.join(tempfile.gettempdir(), f"autodistill_{task}_{nf}.joblib")
        joblib.dump(p, path)
        size = os.path.getsize(path)
        bits = size * 8 / TEACHER_PARAMS
        rows.append({
            "features": nf, "kb": round(size / 1024, 1),
            "sel_acc": round(accuracy_score(true_sel, pred), 4),
            "sel_diff": round(mean, 4), "sel_lower": round(lower, 4),
            "sel_p": round(pv, 4), "bits": round(bits, 5),
            "pass": bits <= budget and lower >= -margin,
        })

    ok = [r for r in rows if r["pass"]]
    chosen = min(ok, key=lambda r: r["bits"]) if ok else min(
        rows, key=lambda r: abs(r["sel_diff"]))

    # Retrain the selected feature count on all training rows, evaluate once on the
    # untouched test set. This is the number that should be reported.
    p = pipe(chosen["features"])
    p.fit(Xtr, ytr)
    pred = p.predict(Xte)
    path = os.path.join(tempfile.gettempdir(), f"autodistill_{task}_final.joblib")
    joblib.dump(p, path)
    size = os.path.getsize(path)
    teacher_te = [y == t for y, t in zip(yte, true_te)]
    student_te = [q == t for q, t in zip(pred, true_te)]
    mean, lower, pv, b, c = paired_stats(teacher_te, student_te)
    final = {
        "features": chosen["features"], "kb": round(size / 1024, 1),
        "bits": round(size * 8 / TEACHER_PARAMS, 5),
        "acc": round(accuracy_score(true_te, pred), 4),
        "agree": round(accuracy_score(yte, pred), 4),
        "teacher_acc": round(accuracy_score(true_te, yte), 4),
        "diff": round(mean, 4), "lower": round(lower, 4),
        "p": round(pv, 4), "b": b, "c": c,
        "pass": chosen["pass"] and lower >= -margin,
    }
    return final["teacher_acc"], rows, chosen, final


def date_task(budget):
    d = load("date_result.json")
    bits = d["bits_per_teacher_param_q4"]
    row = {"student": "char seq2seq 1.35M, 4-bit",
           "kb": round(d["q4_mb"] * 1024, 1),
           "acc": d["exact_teacher"], "agree": d["exact_teacher"],
           "bits": bits, "mcnemar_p": None,
           "pass": bits <= budget and d["exact_teacher"] >= 0.99}
    return d["teacher_iso"], [row], row, None


def name_of(r):
    return r["student"] if "student" in r else f'{r["features"]} features'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["agnews", "sst2", "date"], required=True)
    ap.add_argument("--budget", type=float, default=0.1,
                    help="bits per teacher parameter (1.5B reference)")
    ap.add_argument("--margin", type=float, default=2.0,
                    help="non-inferiority margin in accuracy points (default 2)")
    ap.add_argument("--select-frac", type=float, default=0.2,
                    help="fraction of training rows used for model selection")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    margin = args.margin / 100.0

    if args.task == "date":
        teacher, rows, chosen, final = date_task(args.budget)
    else:
        teacher, rows, chosen, final = classification(
            args.task, args.budget, margin, args.select_frac, args.seed)

    print(f"task {args.task} | teacher {teacher:.4f} | budget {args.budget} bit/param "
          f"({args.budget * TEACHER_PARAMS / 8 / 1e6:.2f} MB) | margin {args.margin:.1f} pts")

    for r in rows:
        if "features" in r:
            print(f'  {name_of(r):<28s} {r["kb"]:>9.1f} KB  sel_acc {r["sel_acc"]:.3f}  '
                  f'delta {r["sel_diff"]:+.3f}  lower {r["sel_lower"]:+.3f}  '
                  f'bit/param {r["bits"]:.5f}  p {r["sel_p"]}  '
                  f'{"PASS" if r["pass"] else "fail"}')
        else:
            print(f'  {name_of(r):<28s} {r["kb"]:>9.1f} KB  acc {r["acc"]:.3f}  '
                  f'bit/param {r["bits"]:.5f}  {"PASS" if r["pass"] else "fail"}')

    if final:
        print(f'FINAL (untouched test, {final["kb"]} KB): acc {final["acc"]:.4f}  '
              f'teacher {final["teacher_acc"]:.4f}  delta {final["diff"]:+.4f}  '
              f'one-sided 95% lower {final["lower"]:+.4f}  p {final["p"]}  '
              f'{"PASS" if final["pass"] else "fail"}')

    if args.task == "date":
        best = next(r for r in rows if r["pass"]) if any(r["pass"] for r in rows) else rows[0]
        print(f'VERDICT: {args.task} {"fits" if best["pass"] else "does not fit"} '
              f'{args.budget} bit/param with {name_of(best)}.')
        return

    if any(r["pass"] for r in rows):
        print(f'VERDICT: {args.task} fits {args.budget} bit/param with '
              f'{name_of(chosen)} ({chosen["kb"]} KB, {chosen["bits"]:.5f} bit/param) '
              f'on the selection set; final test {"passes" if final["pass"] else "fails"} '
              f'the non-inferiority check.')
    else:
        print(f'VERDICT: no student passes {args.budget} bit/param and the '
              f'{args.margin:.1f}-point non-inferiority margin. Best selection rung: '
              f'{name_of(chosen)} ({chosen["kb"]} KB, {chosen["bits"]:.5f} bit/param).')


if __name__ == "__main__":
    main()
