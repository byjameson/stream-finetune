"""Build the AI-in-a-QR demo: an SMS spam classifier whose whole model is in the QR URL.

The model is a binary Complement NB over the top V words. Weights are int8-quantized;
the vocabulary, weights and accuracy are packed into a URL fragment. `demo/qr.html`
decodes the fragment and runs the model locally.

    python demo/build_qr.py
"""
import base64
import json
from pathlib import Path

import numpy as np
import qrcode
import requests
from PIL import Image
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB

ROOT = Path(__file__).resolve().parent.parent
TSV = ROOT / "research/results/sms_spam.tsv"
QR = ROOT / "demo/qr-spam.png"
URL = "https://byjameson-mini-ai.static.hf.space/qr.html"
SRC = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
VOCAB = 160


def load():
    if not TSV.exists():
        r = requests.get(SRC, timeout=60)
        r.raise_for_status()
        TSV.write_text(r.text, encoding="utf-8")
    rows = []
    for line in TSV.read_text(encoding="utf-8").splitlines():
        if "\t" not in line:
            continue
        label, text = line.split("\t", 1)
        if label in ("ham", "spam") and text.strip():
            rows.append((text.strip(), label))
    return rows


def main():
    rows = load()
    x = [t for t, _ in rows]
    y = [l for _, l in rows]
    xtr, xte, ytr, yte = train_test_split(x, y, test_size=0.2, random_state=0,
                                          stratify=y)
    vec = CountVectorizer(binary=True, max_features=VOCAB,
                          token_pattern=r"(?u)\b\w\w+\b")
    Xtr = vec.fit_transform(xtr)
    Xte = vec.transform(xte)
    clf = ComplementNB(alpha=1.0).fit(Xtr, ytr)
    acc = accuracy_score(yte, clf.predict(Xte))

    coef = clf.feature_log_prob_.astype(np.float32)
    scale = np.abs(coef).max(axis=1) / 127.0
    q = np.clip(np.round(coef / scale[:, None]), -127, 127).astype(np.int8)
    # Verify the int8 weights reproduce the float accuracy.
    scores = Xte @ q.T
    pred = [clf.classes_[i] for i in scores.argmax(axis=1)]
    qacc = accuracy_score(yte, pred)

    vocab = list(vec.get_feature_names_out())
    model = {
        "l": [str(c) for c in clf.classes_],
        "v": vocab,
        "w": base64.b64encode(q.tobytes()).decode(),
        "s": [float(v) for v in scale],
        "a": round(float(qacc), 4),
        "n": len(rows),
    }
    raw = json.dumps(model, ensure_ascii=False, separators=(",", ":")).encode()
    frag = base64.urlsafe_b64encode(raw).decode().rstrip("=")
    url = URL + "#m=" + frag
    print(f"rows={len(rows)} vocab={len(vocab)} acc={acc:.4f} qacc={qacc:.4f}")
    print(f"model_json={len(raw)} B  url={len(url)} B")

    qr = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_L,
                       box_size=10, border=4)
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    img.save(QR)
    print(f"qr version={qr.version} modules={qr.modules_count} -> {QR}")

    # Decompress the QR with a decoder to prove the payload is readable.
    import zxingcpp
    got = zxingcpp.read_barcode(Image.open(QR))
    print("decoded:", bool(got), "match:", bool(got) and got.text == url)
    (ROOT / "demo/qr-url.txt").write_text(url + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
