"""Build a WAV file that contains the SMS spam model in its 8-bit samples.

Bytes: 4-byte little-endian JSON length, JSON, then zero padding to the WAV data length.
`demo/audio.html` fetches the WAV, parses the RIFF data chunk and runs the classifier.

    python demo/build_audio.py
"""
import base64
import json
import struct
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB

ROOT = Path(__file__).resolve().parent.parent
TSV = ROOT / "research/results/sms_spam.tsv"
OUT = ROOT / "demo/model.wav"
RATE = 8000
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
    pred = [clf.classes_[i] for i in (Xte @ q.T).argmax(axis=1)]
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
    data = len(raw).to_bytes(4, "little") + raw
    # 8-bit unsigned PCM WAV, mono.
    header = b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVE"
    header += b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, RATE, RATE, 1, 8)
    header += b"data" + struct.pack("<I", len(data))
    OUT.write_bytes(header + data)
    print(f"model={len(raw)} B  wav={OUT.stat().st_size} B  seconds={len(data)/RATE:.3f}  "
          f"acc={acc:.4f}")


if __name__ == "__main__":
    main()
