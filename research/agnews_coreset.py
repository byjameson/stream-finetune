"""Data as the artifact: a small teacher-labelled coreset instead of a stored model.

For each n, sample a stratified coreset of the 6,000 AG News teacher labels, train a
TF-IDF student at "load time", and measure accuracy vs artifact bytes. The stored artifact
is the data; the training procedure is shared.
"""
import json
import time
from collections import defaultdict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

D = json.load(open("research/results/agnews_labels.json", encoding="utf-8"))
tr = [r for r in D["train"] if r["label"] != "unknown"]
te = D["test"]
LABELS = sorted({r["label"] for r in tr})
print("labels", LABELS, "train", len(tr), "test", len(te))


def strat_sample(n, seed=0):
    by = defaultdict(list)
    for r in tr:
        by[r["label"]].append(r)
    per = max(1, n // len(LABELS))
    out = []
    for i, c in enumerate(LABELS):
        out += by[c][:per]
    return out


rows = []
for n in [20, 50, 100, 200, 500, 1000, 2000]:
    data = strat_sample(n)
    artifact = json.dumps([{"text": r["text"], "label": r["label"]} for r in data],
                          ensure_ascii=False).encode("utf-8")
    t0 = time.time()
    p = Pipeline([("t", TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), max_features=5000)),
                  ("l", LogisticRegression(max_iter=1000, C=4.0))])
    p.fit([r["text"] for r in data], [r["label"] for r in data])
    train_s = time.time() - t0
    pred = p.predict([r["text"] for r in te])
    acc = accuracy_score([r["true"] for r in te], pred)
    agree = accuracy_score([r["label"] for r in te], pred)
    row = {"n": len(data), "artifact_bytes": len(artifact),
           "accuracy": round(acc, 4), "agree_teacher": round(agree, 4),
           "bits_per_teacher_param": round(len(artifact) * 8 / 1.5e9, 8),
           "train_seconds": round(train_s, 2)}
    rows.append(row)
    print(json.dumps(row), flush=True)

json.dump(rows, open("research/results/agnews_coreset.json", "w"), indent=1)
