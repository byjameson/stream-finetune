"""Build the 1.8 MB max-accuracy AG News page.

    python demo/build_max_ai.py

Trains a TF-IDF + logistic-regression model on 75,000 human-labelled AG News rows, exports
the vocabulary, IDF, int8 coefficients and intercepts into `demo/max-ai.html`, and keeps
the same 400-row test set as the 130 KB demo for a live accuracy check.
"""
import base64
import json
from pathlib import Path

import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

ROOT = Path(__file__).resolve().parent.parent
# AG News integer labels: 0=World, 1=Sports, 2=Business, 3=Sci/Tech.
LABELS = ["World", "Sports", "Business", "Sci/Tech"]
N_TRAIN = 75_000
N_FEATURES = 75_000


def main():
    d = json.load(open(ROOT / "research/results/agnews_labels.json", encoding="utf-8"))
    test = [{"text": r["text"], "true": r["true"]}
            for r in d["test"] if r["true"] in LABELS]

    ds = load_dataset("fancyzhx/ag_news", split="train")
    per = N_TRAIN // len(LABELS)
    by = {l: [] for l in LABELS}
    for r in ds:
        label = LABELS[r["label"]]
        if len(by[label]) < per:
            by[label].append(r["text"])
        if all(len(v) >= per for v in by.values()):
            break
    x_train, y_train = [], []
    for l in LABELS:
        x_train += by[l]
        y_train += [l] * len(by[l])
    print(f"train {len(x_train)} rows, {N_TRAIN} features", flush=True)

    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True,
                          max_features=N_FEATURES)
    x = vec.fit_transform(x_train)
    clf = LogisticRegression(C=4.0, max_iter=1000).fit(x, y_train)
    acc = accuracy_score([r["true"] for r in test],
                         clf.predict(vec.transform([r["text"] for r in test])))
    print(f"held-out accuracy {acc:.4f}", flush=True)

    vocab = list(vec.get_feature_names_out())
    idf = vec.idf_.astype(np.float32)
    coef = clf.coef_.astype(np.float32)
    classes = list(clf.classes_)
    scale = np.abs(coef).max(axis=1) / 127.0
    scale[scale == 0] = 1.0
    q = np.clip(np.round(coef / scale[:, None]), -127, 127).astype(np.int8)

    model = {
        "classes": classes,
        "vocab": vocab,
        "idf_b64": base64.b64encode(idf.tobytes()).decode(),
        "coef_b64": base64.b64encode(q.tobytes()).decode(),
        "coef_scale": [float(v) for v in scale],
        "intercept": [float(v) for v in clf.intercept_],
        "accuracy": round(float(acc), 4),
        "train_rows": len(x_train),
        "features": len(vocab),
    }
    model["artifact_bytes"] = len(
        json.dumps(model, ensure_ascii=False, separators=(",", ":")).encode())

    payload = json.dumps({"model": model, "test": test}, ensure_ascii=False,
                         separators=(",", ":"))
    payload = payload.replace("<", "\\u003c").replace("\u2028", "\\u2028").replace(
        "\u2029", "\\u2029")

    tpl = (ROOT / "demo/max_ai_template.html").read_text(encoding="utf-8")
    if "__DATA__" not in tpl:
        raise SystemExit("template has no __DATA__ placeholder")
    html = tpl.replace("__DATA__", payload, 1)
    out = ROOT / "demo/max-ai.html"
    out.write_text(html, encoding="utf-8")
    print(f"artifact {model['artifact_bytes']} bytes ({model['artifact_bytes']/1048576:.2f} MB)")
    print(f"wrote {out} ({out.stat().st_size} bytes, {out.stat().st_size/1048576:.2f} MB)")


if __name__ == "__main__":
    main()
