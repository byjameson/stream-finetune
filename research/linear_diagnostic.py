"""Cheap diagnostic: linear compressibility C = (linear_acc - chance) / (teacher_acc - chance).

Run on the saved teacher labels for AG News, SST-2 and MNLI. This is a diagnostic, not a
law: it orders the three tasks, but three points are not a fit.
"""
import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

TASKS = {
    "agnews": ("agnews_labels.json", 4, lambda r: r["text"]),
    "sst2": ("sst2_labels.json", 2, lambda r: r["text"]),
    "mnli": ("mnli_labels.json", 3, lambda r: r["premise"] + " " + r["hypothesis"]),
    "qnli": ("qnli_labels.json", 2, lambda r: r["question"] + " " + r["sentence"]),
    "dbpedia": ("dbpedia_labels.json", 14, lambda r: r["text"]),
    "qqp": ("qqp_labels.json", 2, lambda r: r["question1"] + " " + r["question2"]),
    "20ng": ("20ng_labels.json", 20, lambda r: r["text"]),
    "emotion": ("emotion_labels.json", 6, lambda r: r["text"]),
}
FEATURES = [1000, 2000, 5000, 10000, 20000, 50000]
rows = []
for name, (path, nclass, text) in TASKS.items():
    if not Path(f"research/results/{path}").exists():
        print("skip", name, "missing", path)
        continue
    d = json.load(open(f"research/results/{path}", encoding="utf-8"))
    tr = [r for r in d["train"] if r["label"] != "unknown"]
    te = d["test"]
    Xtr, ytr = [text(r) for r in tr], [r["label"] for r in tr]
    Xte = [text(r) for r in te]
    teacher = accuracy_score([r["true"] for r in te], [r["label"] for r in te])
    chance = 1 / nclass
    best = None
    for nf in FEATURES:
        p = Pipeline([("t", TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), max_features=nf)),
                      ("l", LogisticRegression(max_iter=1000, C=4.0))])
        p.fit(Xtr, ytr)
        pred = p.predict(Xte)
        acc = accuracy_score([r["true"] for r in te], pred)
        if best is None or acc > best["acc"]:
            best = {"features": nf, "acc": acc, "agree": accuracy_score([r["label"] for r in te], pred)}
    c = (best["acc"] - chance) / (teacher - chance)
    row = {"task": name, "classes": nclass, "teacher": round(teacher, 4),
           "linear_acc": round(best["acc"], 4), "linear_features": best["features"],
           "linear_agree": round(best["agree"], 4), "C": round(c, 3)}
    rows.append(row)
    print(json.dumps(row))
json.dump(rows, open("research/results/linear_diagnostic.json", "w"), indent=1)
