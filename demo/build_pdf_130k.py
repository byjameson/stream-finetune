"""Put the 130 KB AG News coreset in a PDF model card.

The visible page describes the model. The 500 training examples (132,798 bytes of JSON)
are in the `/TinyAIData` metadata as base64url, not drawn on the page. `pdf-130k.html`
fetches the PDF, trains the same Complement NB in the browser and classifies.

    python demo/build_pdf_130k.py
"""
import base64
import hashlib
import json
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "research/results/agnews_labels.json"
OUT = ROOT / "demo/model-card-130k.pdf"
LABELS = ["Business", "World", "Sports", "Sci/Tech"]


def data():
    d = json.load(open(SRC, encoding="utf-8"))
    train = {l: [] for l in LABELS}
    for r in d["train"]:
        if r["label"] in LABELS and len(train[r["label"]]) < 500:
            train[r["label"]].append({"text": r["text"], "label": r["label"]})
    flat = []
    for l in LABELS:
        flat += train[l][:125]
    assert len(flat) == 500
    raw = json.dumps(flat, ensure_ascii=False, separators=(",", ":")).encode()
    assert len(raw) == 132798, len(raw)
    return raw


def esc(s):
    return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def content_stream():
    lines = [
        "AI in a PDF - 130 KB coreset",
        "The model is 500 AG News examples, not weights.",
        "",
        "Train rows: 500 (125 per class)",
        "Data: 132,798 bytes of JSON; labels: Business, World, Sports, Sci/Tech",
        "Classifier: Complement Naive Bayes, trained when the viewer loads.",
        "Test: 400 rows in mini-ai.html; 78.3% embedded / 74.3% fresh.",
        "PDF: under 180 KB; overhead is page, base64 and structure.",
        "",
        "The examples are in /TinyAIData metadata, not drawn on this page.",
        "Open pdf-130k.html, load this file, and classify in the browser.",
        "The full SHA-256 is in /TinyAISHA256.",
        "",
        "Caveats: teacher labels from a 1.5B model; a 400-row test;",
        "English news; bag-of-words; not an LLM.",
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
    blob = data()
    payload = base64.urlsafe_b64encode(blob).decode().rstrip("=")
    digest = hashlib.sha256(blob).hexdigest()

    stream = content_stream()
    objects = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        2: b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        3: (b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>"),
        4: b"<< /Length %d >>\nstream\n%s\nendstream" % (len(stream), stream),
        5: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        6: (f"<< /Title ({esc('AI in a PDF - 130 KB coreset')}) "
            f"/Author ({esc('stream-finetune')}) "
            f"/TinyAIData ({payload}) "
            f"/TinyAISHA256 ({digest}) "
            f"/Creator ({esc('demo/build_pdf_130k.py')}) "
            f"/Producer ({esc('stream-finetune')}) >>").encode("latin-1"),
    }
    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for i in range(1, 7):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode("ascii") + objects[i] + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 7\n0000000000 65535 f \n"
    for off in offsets[1:]:
        out += f"{off:010d} 00000 n \n".encode("ascii")
    out += (f"trailer\n<< /Size 7 /Root 1 0 R /Info 6 0 R >>\n"
            f"startxref\n{xref}\n%%EOF\n").encode("ascii")
    OUT.write_bytes(out)
    print(f"data={len(blob)} B  payload={len(payload)} chars  pdf={len(out)} B  "
          f"sha={digest[:16]}...")


if __name__ == "__main__":
    build()
