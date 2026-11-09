"""Build a 32x32 favicon that contains the SMS spam model.

The RGB bytes are: 4-byte little-endian JSON length, the JSON model, zero padding.
`demo/favicon.html` reads the pixels back with a canvas and runs the classifier.

    python demo/build_favicon.py
"""
import base64
import json
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB

ROOT = Path(__file__).resolve().parent.parent
TSV = ROOT / "research/results/sms_spam.tsv"
OUT = ROOT / "demo/favicon.png"
SIZE = 32
VOCAB = 160


def main():
    rows = []
    for line in TSV.read_text(encoding="utf-8").splitlines():
        if "\t" not in line:
            continue
        label, text = line.split("\t", 1)
        if label in ("ham", "spam") and text.strip():
            rows.append((text.strip(), label))
    x = [t for t, _ in rows]
    y = [l for _, l in rows]
    xtr, xte, ytr, yte = train_test_split(x, y, test_size=0.2, random_state=0, stratify=y)
    vec = CountVectorizer(binary=True, max_features=VOCAB, token_pattern=r"(?u)\b\w\w+\b")
    Xtr = vec.fit_transform(xtr)
    Xte = vec.transform(xte)
    clf = ComplementNB(alpha=1.0).fit(Xtr, ytr)
    coef = clf.feature_log_prob_.astype(np.float32)
    scale = np.abs(coef).max(axis=1) / 127.0
    q = np.clip(np.round(coef / scale[:, None]), -127, 127).astype(np.int8)
    scores = Xte @ q.T
    pred = [clf.classes_[i] for i in scores.argmax(axis=1)]
    acc = accuracy_score(yte, pred)

    model = {
        "l": [str(c) for c in clf.classes_],
        "v": list(vec.get_feature_names_out()),
        "w": base64.b64encode(q.tobytes()).decode(),
        "s": [float(v) for v in scale],
        "a": round(float(acc), 4),
        "n": len(rows),
    }
    raw = json.dumps(model, ensure_ascii=False, separators=(",", ":")).encode()
    cap = SIZE * SIZE * 3
    if len(raw) + 4 > cap:
        raise SystemExit(f"model {len(raw)} B does not fit {cap} B")
    data = len(raw).to_bytes(4, "little") + raw + bytes(cap - 4 - len(raw))
    img = Image.frombytes("RGB", (SIZE, SIZE), data)
    img.save(OUT)
    print(f"model={len(raw)} B  png={OUT.stat().st_size} B  acc={acc:.4f}  "
          f"pixels={SIZE}x{SIZE}")


if __name__ == "__main__":
    main()
