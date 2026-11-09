"""The fixed SMS containers deploy the raw int8 dot product, not q * scale.

Applying the per-class scales changed the argmax on the held-out split and lowered the
measured accuracy, so the scales are quantization metadata only. This test pins the rule
that matches the stored `a` and the 93.5% claim.
"""
import base64
import json
import re
import struct
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
WAV = ROOT / "demo/model.wav"
TSV = ROOT / "research/results/sms_spam.tsv"


def _model():
    raw = WAV.read_bytes()
    off, data = 12, None
    while off + 8 <= len(raw):
        kind = raw[off:off + 4]
        size = struct.unpack("<I", raw[off + 4:off + 8])[0]
        if kind == b"data":
            data = raw[off + 8:off + 8 + size]
            break
        off += 8 + size + (size & 1)
    assert data is not None
    n = struct.unpack("<I", data[:4])[0]
    return json.loads(data[4:4 + n])


def test_raw_int8_rule_matches_the_claim():
    m = _model()
    q = np.frombuffer(base64.b64decode(m["w"]), dtype=np.int8)
    q = q.reshape(len(m["l"]), len(m["v"]))
    s = np.asarray(m["s"], dtype=np.float32)
    vocab = {w: i for i, w in enumerate(m["v"])}

    rows = []
    for line in TSV.read_text(encoding="utf-8").splitlines():
        if "\t" not in line:
            continue
        label, text = line.split("\t", 1)
        if label in ("ham", "spam") and text.strip():
            rows.append((text.strip(), label))
    x = [t for t, _ in rows]
    y = [l for _, l in rows]
    _xtr, xte, _ytr, yte = train_test_split(x, y, test_size=0.2, random_state=0,
                                            stratify=y)

    def features(text):
        row = np.zeros(len(vocab), dtype=np.int16)
        for token in set(re.findall(r"[^\W_]{2,}", text.lower(), re.UNICODE)):
            i = vocab.get(token)
            if i is not None:
                row[i] = 1
        return row

    X = np.stack([features(t) for t in xte])
    raw = np.asarray(m["l"])[np.argmax(X @ q.astype(np.int16).T, axis=1)]
    scaled = np.asarray(m["l"])[np.argmax(X @ (q.astype(np.float32) * s[:, None]).T,
                                          axis=1)]
    raw_acc = float(accuracy_score(yte, raw))
    scaled_acc = float(accuracy_score(yte, scaled))
    assert abs(raw_acc - m["a"]) < 1e-4, (raw_acc, m["a"])
    assert scaled_acc < raw_acc - 0.01, (scaled_acc, raw_acc)
