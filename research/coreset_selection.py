"""Can a 130 KB AG News classifier pick better examples for the same bytes?

The 130 KB demo uses the first N/4 rows per class. This script compares that with random,
token-coverage, TF-IDF diversity and uncertainty selection at the same example count, then
measures held-out accuracy and artifact bytes.

    python research/coreset_selection.py

Writes research/results/coreset_selection.json.
"""
import json
import math
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import ComplementNB

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "research/results/agnews_labels.json"
OUT = ROOT / "research/results/coreset_selection.json"
LABELS = ["Business", "World", "Sports", "Sci/Tech"]


def tokens(text):
    return set(re.findall(r"[a-z0-9]{2,}", text.lower()))


def load():
    d = json.load(open(DATA, encoding="utf-8"))
    train = {l: [] for l in LABELS}
    for r in d["train"]:
        if r["label"] in LABELS:
            train[r["label"]].append(r)
    test = [r for r in d["test"] if r["true"] in LABELS]
    return train, test


def artifact_bytes(rows):
    flat = [{"text": r["text"], "label": r["label"]} for r in rows]
    return len(json.dumps(flat, ensure_ascii=False, separators=(",", ":")).encode())


def evaluate(train_rows, test):
    vec = CountVectorizer(binary=True, token_pattern=r"(?u)\b[a-z0-9]{2,}\b")
    X = vec.fit_transform([r["text"] for r in train_rows])
    T = vec.transform([r["text"] for r in test])
    clf = ComplementNB(alpha=1.0).fit(X, [r["label"] for r in train_rows])
    pred = clf.predict(T)
    return float((pred == np.array([r["true"] for r in test])).mean())


def first_per_class(train, n, seed=0):
    return [r for l in LABELS for r in train[l][:n]]


def random_per_class(train, n, seed):
    rng = random.Random(seed)
    out = []
    for l in LABELS:
        out += rng.sample(train[l], min(n, len(train[l])))
    return out


def coverage_per_class(train, n):
    # IDF over the pool makes rare-token coverage worth more than common-token coverage.
    docs = [tokens(r["text"]) for l in LABELS for r in train[l]]
    df = Counter(t for d in docs for t in d)
    N = len(docs)
    idf = {t: math.log((1 + N) / (1 + c)) + 1 for t, c in df.items()}
    out = []
    for l in LABELS:
        pool = train[l]
        tok = [tokens(r["text"]) for r in pool]
        chosen, covered = [], set()
        for _ in range(min(n, len(pool))):
            best, best_score = -1, -1.0
            for i, ts in enumerate(tok):
                if i in chosen:
                    continue
                score = sum(idf.get(t, 1.0) for t in ts - covered)
                # Prefer a little length normalisation; otherwise long rows always win.
                score /= math.sqrt(max(len(ts), 1))
                if score > best_score:
                    best, best_score = i, score
            chosen.append(best)
            covered |= tok[best]
        out += [pool[i] for i in chosen]
    return out


def tfidf_per_class(train, n, seed=0):
    out = []
    for l in LABELS:
        pool = train[l]
        vec = TfidfVectorizer(token_pattern=r"(?u)\b[a-z0-9]{2,}\b", sublinear_tf=True)
        X = vec.fit_transform([r["text"] for r in pool])
        k = min(n, X.shape[0])
        km = MiniBatchKMeans(n_clusters=k, random_state=seed, batch_size=256,
                             n_init=3).fit(X)
        chosen = set()
        for c in range(k):
            members = np.where(km.labels_ == c)[0]
            if len(members) == 0:
                continue
            # Pick the row closest to the centroid; fall back to any member.
            centroid = km.cluster_centers_[c]
            dist = np.linalg.norm(X[members] - centroid, axis=1)
            chosen.add(int(members[int(np.argmin(dist))]))
        # Fill if duplicate cluster picks left us short.
        for i in range(len(pool)):
            if len(chosen) >= k:
                break
            chosen.add(i)
        out += [pool[i] for i in sorted(chosen)[:k]]
    return out


def uncertainty_per_class(train, n, seed=0):
    # A transductive vectorizer, then iterative CNB uncertainty sampling.
    all_rows = [r for l in LABELS for r in train[l]]
    vec = CountVectorizer(binary=True, token_pattern=r"(?u)\b[a-z0-9]{2,}\b")
    X = vec.fit_transform([r["text"] for r in all_rows])
    index = {}
    i = 0
    for l in LABELS:
        for r in train[l]:
            index[id(r)] = i
            i += 1
    rng = random.Random(seed)
    chosen_by_label = {}
    for l in LABELS:
        start = rng.sample(train[l], min(10, len(train[l])))
        chosen = {id(r) for r in start}
        while len(chosen) < min(n, len(train[l])):
            rows = [r for r in train[l] if id(r) in chosen]
            ids = [index[id(r)] for r in rows]
            clf = ComplementNB(alpha=1.0).fit(X[ids], [r["label"] for r in rows])
            rest = [r for r in train[l] if id(r) not in chosen]
            if not rest:
                break
            rid = [index[id(r)] for r in rest]
            proba = clf.predict_proba(X[rid])
            entropy = -(proba * np.log(np.maximum(proba, 1e-12))).sum(axis=1)
            take = min(10, len(rest), min(n, len(train[l])) - len(chosen))
            for j in np.argsort(-entropy)[:take]:
                chosen.add(id(rest[int(j)]))
        chosen_by_label[l] = chosen
    out = []
    for l in LABELS:
        out += [r for r in train[l] if id(r) in chosen_by_label[l]]
    return out


def main():
    train, test = load()
    print("train", {l: len(train[l]) for l in LABELS}, "test", len(test))
    results = []
    for per_class in (50, 125, 250):
        rows = first_per_class(train, per_class)
        results.append({"strategy": "first", "per_class": per_class, "seed": 0,
                        "bytes": artifact_bytes(rows), "acc": evaluate(rows, test)})
        for seed in range(5):
            rows = random_per_class(train, per_class, seed)
            results.append({"strategy": "random", "per_class": per_class, "seed": seed,
                            "bytes": artifact_bytes(rows), "acc": evaluate(rows, test)})
        for name, fn in (("coverage", coverage_per_class), ("tfidf", tfidf_per_class),
                         ("uncertainty", uncertainty_per_class)):
            rows = fn(train, per_class)
            results.append({"strategy": name, "per_class": per_class, "seed": 0,
                            "bytes": artifact_bytes(rows), "acc": evaluate(rows, test)})
        print(f"per_class={per_class} done")

    summary = {}
    for per_class in (50, 125, 250):
        subset = [r for r in results if r["per_class"] == per_class]
        summary[str(per_class)] = {}
        for s in ("first", "random", "coverage", "tfidf", "uncertainty"):
            vals = [r for r in subset if r["strategy"] == s]
            accs = [r["acc"] for r in vals]
            summary[str(per_class)][s] = {
                "acc_mean": round(float(np.mean(accs)), 4),
                "acc_std": round(float(np.std(accs)), 4),
                "bytes_mean": round(float(np.mean([r["bytes"] for r in vals]))),
                "n": len(vals),
            }
    payload = {"labels": LABELS, "test_rows": len(test),
               "train_rows": {l: len(train[l]) for l in LABELS},
               "summary": summary, "runs": results}
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    for per_class, d in summary.items():
        print(f"\nper_class={per_class}")
        for s, v in d.items():
            print(f"  {s:12s} {v['acc_mean']*100:5.1f}% +- {v['acc_std']*100:4.1f} "
                  f"({v['bytes_mean']:,} B)")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
