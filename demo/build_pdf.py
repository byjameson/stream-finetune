"""Wrap the shipped 1,579-byte SMS model in a PDF model card.

The visible page describes the model; the int8 weights sit in the PDF `/Keywords`
metadata as base64url JSON. `demo/pdf.html` fetches the raw file, reads that string and
runs the classifier. This is a container, not steganography: the metadata is inspectable.

The source of truth is `demo/model.wav`, so this script cannot silently train a different
model from the favicon/WAV demos.

    python demo/build_pdf.py
"""
import base64
import hashlib
import json
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WAV = ROOT / "demo/model.wav"
OUT = ROOT / "demo/model-card.pdf"


def read_wav_model():
    raw = WAV.read_bytes()
    assert raw[:4] == b"RIFF" and raw[8:12] == b"WAVE"
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
    blob = data[4:4 + n]
    return blob, json.loads(blob)


def esc(s):
    return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def content_stream(model):
    lines = [
        "AI in a PDF",
        "The visible page is a model card. The classifier is in /Keywords.",
        "A 1,579-byte SMS spam model is not drawn on this page; it is metadata.",
        "",
        f"Task: SMS spam (labels: {', '.join(model['l'])})",
        f"Training data: {model['n']:,} messages; 1,115 held out",
        f"Accuracy: {model['a'] * 100:.2f}% on the held-out split",
        f"Weights: int8 Complement NB, {len(model['v'])} words, 1,579 bytes",
        "SHA-256 in /TinyAISHA256",
        "Other containers: QR 2,158 B, favicon 1,519 B, WAV 1,627 B",
        "",
        "Read it: pdf.html fetches this file, parses /Keywords and classifies",
        "in the browser. No server and no weights endpoint.",
        "",
        "Caveats: one 80/20 split, English SMS, bag-of-words, not an LLM.",
        "The metadata is inspectable; a PDF is a lossless container, not a secret.",
    ]
    parts = ["BT", "/F1 20 Tf", "72 720 Td", f"({esc(lines[0])}) Tj",
             "/F1 11 Tf", "0 -28 Td"]
    for line in lines[1:]:
        if line:
            parts.append(f"({esc(line)}) Tj")
        parts.append("0 -16 Td")
    parts.append("ET")
    return "\n".join(parts).encode("latin-1")


def build():
    blob, model = read_wav_model()
    assert len(blob) == 1579, len(blob)
    payload = base64.urlsafe_b64encode(blob).decode().rstrip("=")
    digest = hashlib.sha256(blob).hexdigest()
    stream = content_stream(model)

    objects = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        2: b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        3: (b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>"),
        4: b"<< /Length %d >>\nstream\n%s\nendstream" % (len(stream), stream),
        5: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        6: (f"<< /Title ({esc('AI in a PDF - model card')}) "
            f"/Author ({esc('stream-finetune')}) "
            f"/TinyAISHA256 ({digest}) "
            f"/Keywords ({payload}) "
            f"/Creator ({esc('demo/build_pdf.py')}) "
            f"/Producer ({esc('stream-finetune')}) >>").encode("latin-1"),
    }

    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for i in range(1, 7):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode("ascii") + objects[i] + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 7\n0000000000 65535 f \n".encode("ascii")
    for off in offsets[1:]:
        out += f"{off:010d} 00000 n \n".encode("ascii")
    out += (f"trailer\n<< /Size 7 /Root 1 0 R /Info 6 0 R >>\n"
            f"startxref\n{xref}\n%%EOF\n").encode("ascii")
    OUT.write_bytes(out)
    print(f"model={len(blob)} B  pdf={len(out)} B  payload={len(payload)} chars  "
          f"acc={model['a']:.4f}")


if __name__ == "__main__":
    build()
