"""Pin the shipped containers: a 1,579-byte SMS model in a 32x32 PNG and an 8-bit WAV.

The test decodes both with the standard library only, so it runs in CI without Pillow.
It also checks the format contract Tiny AI Maker exports (4-byte length + JSON) and that
the decoded int8 weights still classify a spam and a ham message.
"""

import base64
import hashlib
import json
import re
import struct
import xml.etree.ElementTree as ET
import zlib
from collections import Counter
from pathlib import Path

DEMO = Path(__file__).resolve().parents[1] / "demo"
PNG = DEMO / "favicon.png"
WAV = DEMO / "model.wav"


def _png_pixels(path):
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    pos, idat = 8, bytearray()
    width = height = depth = color = None
    while pos < len(data):
        length = struct.unpack(">I", data[pos:pos + 4])[0]
        kind = data[pos + 4:pos + 8]
        body = data[pos + 8:pos + 8 + length]
        if kind == b"IHDR":
            width, height, depth, color = struct.unpack(">IIBB", body[:10])
        elif kind == b"IDAT":
            idat += body
        pos += 12 + length
    assert (width, height, depth, color) == (32, 32, 8, 2), (width, height, depth, color)

    raw = zlib.decompress(bytes(idat))
    stride = width * 3
    out = bytearray()
    prev = bytearray(stride)
    i = 0
    for _ in range(height):
        filt = raw[i]
        i += 1
        line = bytearray(raw[i:i + stride])
        i += stride
        if filt == 1:
            for x in range(3, stride):
                line[x] = (line[x] + line[x - 3]) & 255
        elif filt == 2:
            for x in range(stride):
                line[x] = (line[x] + prev[x]) & 255
        elif filt == 3:
            for x in range(stride):
                left = line[x - 3] if x >= 3 else 0
                line[x] = (line[x] + ((left + prev[x]) >> 1)) & 255
        elif filt == 4:
            for x in range(stride):
                a = line[x - 3] if x >= 3 else 0
                b = prev[x]
                c = prev[x - 3] if x >= 3 else 0
                pa, pb, pc = abs(b - c), abs(a - c), abs(a + b - 2 * c)
                pred = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                line[x] = (line[x] + pred) & 255
        else:
            assert filt == 0, filt
        out += line
        prev = line
    return bytes(out)


def _model(blob):
    length = struct.unpack("<I", blob[:4])[0]
    assert 4 + length <= len(blob)
    return json.loads(blob[4:4 + length].decode("utf-8")), blob[4:4 + length]


def _wav_data(path):
    data = path.read_bytes()
    assert data[:4] == b"RIFF" and data[8:12] == b"WAVE"
    pos, payload = 12, None
    while pos + 8 <= len(data):
        kind = data[pos:pos + 4]
        size = struct.unpack("<I", data[pos + 4:pos + 8])[0]
        if kind == b"fmt ":
            fmt = struct.unpack("<HHIIHH", data[pos + 8:pos + 8 + size])
        elif kind == b"data":
            payload = data[pos + 8:pos + 8 + size]
            break
        pos += 8 + size + (size & 1)
    assert payload is not None
    assert fmt == (1, 1, 8000, 8000, 1, 8), fmt
    return payload


def _label(model, text):
    vocab = {word: i for i, word in enumerate(model["v"])}
    labels = model["l"]
    width = len(model["v"])
    q = struct.unpack(f"{len(labels) * width}b",
                      base64.urlsafe_b64decode(model["w"]))
    scores = [0.0] * len(labels)
    for token in set(re.findall(r"[^\W_]{2,}", text.lower())):
        i = vocab.get(token)
        if i is None:
            continue
        for c in range(len(labels)):
            scores[c] += q[c * width + i] * model["s"][c]
    return labels[max(range(len(labels)), key=scores.__getitem__)]


def test_favicon_is_a_32x32_model():
    raw = _png_pixels(PNG)
    model, blob = _model(raw)
    assert len(blob) == 1579
    assert model["l"] == ["ham", "spam"]
    assert len(model["v"]) == 160
    assert model["n"] == 5574
    assert round(model["a"], 4) == 0.9354
    assert sorted(model) == ["a", "l", "n", "s", "v", "w"]
    assert len(base64.urlsafe_b64decode(model["w"])) == 2 * 160


def test_wav_carries_the_same_model():
    payload = _wav_data(WAV)
    assert len(payload) == 1583  # 4-byte length + 1,579-byte JSON
    model, blob = _model(payload)
    png_model, png_blob = _model(_png_pixels(PNG))
    assert payload[4:] == blob
    assert model == png_model
    assert blob == png_blob


def test_pdf_metadata_carries_the_same_model():
    data = (DEMO / "model-card.pdf").read_bytes()
    assert len(data) == 3832
    assert b"93.54%" in data
    assert b"not an LLM" in data
    match = re.search(rb"/Keywords\s*\(([^)]*)\)", data)
    assert match is not None
    payload = match.group(1)
    payload += b"=" * (-len(payload) % 4)
    blob = base64.urlsafe_b64decode(payload)
    assert len(blob) == 1579
    model = json.loads(blob)
    wav_model, wav_blob = _model(_wav_data(WAV))
    assert blob == wav_blob
    assert model == wav_model
    assert model["l"] == ["ham", "spam"]
    digest = re.search(rb"/TinyAISHA256\s*\(([0-9a-f]{64})\)", data)
    assert digest is not None
    assert digest.group(1).decode() == hashlib.sha256(blob).hexdigest()


def test_svg_metadata_carries_the_same_model():
    data = (DEMO / "model-card.svg").read_bytes()
    assert len(data) == 5004
    root = ET.fromstring(data)
    meta = None
    for node in root.iter("{http://www.w3.org/2000/svg}metadata"):
        if node.get("id") == "tinyai":
            meta = node
    assert meta is not None
    payload = (meta.text or "").strip()
    payload += "=" * (-len(payload) % 4)
    blob = base64.urlsafe_b64decode(payload)
    assert len(blob) == 1579
    _, wav_blob = _model(_wav_data(WAV))
    assert blob == wav_blob
    assert meta.get("data-sha256") == hashlib.sha256(blob).hexdigest()


def test_pdf_130k_carries_the_agnews_coreset():
    data = (DEMO / "model-card-130k.pdf").read_bytes()
    assert len(data) == 178761
    match = re.search(rb"/TinyAIData\s*\(([^)]*)\)", data)
    assert match is not None
    payload = match.group(1)
    payload += b"=" * (-len(payload) % 4)
    blob = base64.urlsafe_b64decode(payload)
    assert len(blob) == 132798
    rows = json.loads(blob)
    assert len(rows) == 500
    assert Counter(r["label"] for r in rows) == {
        "Business": 125, "World": 125, "Sports": 125, "Sci/Tech": 125}
    digest = re.search(rb"/TinyAISHA256\s*\(([0-9a-f]{64})\)", data)
    assert digest is not None
    assert digest.group(1).decode() == hashlib.sha256(blob).hexdigest()


def test_survival_json_pins_the_model():
    data = json.loads((DEMO / "survival.json").read_text(encoding="utf-8"))
    _, blob = _model(_wav_data(WAV))
    assert data["model_bytes"] == 1579
    assert data["model_sha256"] == hashlib.sha256(blob).hexdigest()
    assert any(row["ok"] is True for row in data["rows"])
    assert any(row["ok"] is False for row in data["rows"])


def test_decoded_weights_still_classify():
    model, _ = _model(_wav_data(WAV))
    assert _label(model, "WINNER!! You have been selected to receive a £900 prize "
                         "reward! Call 09061701461 now to claim.") == "spam"
    assert _label(model, "Hey, are we still meeting for lunch tomorrow? Let me know.") == "ham"
