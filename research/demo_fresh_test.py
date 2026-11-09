"""Held-out sanity check for the 130 KB and 1.7 MB demos on a fresh test sample.

The demos embed 400 AG News test rows. This picks a different 400 test rows (disjoint from
the embedded ones) and measures the same artifacts, so a lucky test set would show up.
"""
import json
from collections import defaultdict

from datasets import load_dataset
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline

LABELS = ["World", "Sports", "Business", "Sci/Tech"]


def main():
    saved = json.load(open("research/results/agnews_labels.json", encoding="utf-8"))
    used = {r["text"] for r in saved["test"]}
    ds = load_dataset("fancyzhx/ag_news", split="test")
    fresh = []
    for r in ds:
        if r["text"] in used:
            continue
        fresh.append({"text": r["text"], "true": LABELS[r["label"]]})
        if len(fresh) == 400:
            break
    # fresh is the first 400 in dataset order; shuffle deterministically for class mix
    import random
    random.Random(0).shuffle(fresh)

    train = [r for r in saved["train"] if r["label"] in LABELS]
    by = defaultdict(list)
    for r in train:
        by[r["label"]].append(r)
    nb_acc = {}
    for n in (500, 1000, 2000):
        coreset = []
        for l in LABELS:
            coreset += by[l][: n // len(LABELS)]
        nb = Pipeline([("v", CountVectorizer(binary=True, ngram_range=(1, 1))),
                       ("m", ComplementNB(alpha=1.0))])
        nb.fit([r["text"] for r in coreset], [r["label"] for r in coreset])
        nb_acc[str(n)] = round(float(accuracy_score(
            [r["true"] for r in fresh],
            nb.predict([r["text"] for r in fresh]))), 4)

    # The max mode trained on the first 18,750 human rows per class; reuse that sample.
    tr = load_dataset("fancyzhx/ag_news", split="train")
    per = 75000 // 4
    by2 = {l: [] for l in LABELS}
    for r in tr:
        l = LABELS[r["label"]]
        if len(by2[l]) < per:
            by2[l].append({"text": r["text"], "label": l})
        if all(len(v) >= per for v in by2.values()):
            break
    rows = [r for l in LABELS for r in by2[l]]
    lin = Pipeline([("v", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True,
                                          max_features=75000)),
                    ("m", LogisticRegression(C=4.0, max_iter=1000))])
    lin.fit([r["text"] for r in rows], [r["label"] for r in rows])
    lin_acc = accuracy_score([r["true"] for r in fresh],
                             lin.predict([r["text"] for r in fresh]))

    out = {"fresh_rows": len(fresh), "nb_acc": nb_acc,
           "linear75k_acc": round(float(lin_acc), 4)}
    print(json.dumps(out, indent=1))
    json.dump(out, open("research/results/demo_fresh_test.json", "w"), indent=1)


if __name__ == "__main__":
    main()
