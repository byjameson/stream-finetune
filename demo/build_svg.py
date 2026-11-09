"""Put the 1,579-byte SMS model in an SVG model card.

The visible vector card describes the model. The int8 weights are base64url JSON in
`<metadata id="tinyai">`, not drawn. `demo/svg.html` parses the XML and classifies.

    python demo/build_svg.py
"""
import base64
import hashlib
import json
import struct
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parent.parent
WAV = ROOT / "demo/model.wav"
OUT = ROOT / "demo/model-card.svg"


def read_model():
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


def metric(x, y, value, label):
    return (
        f'<text x="{x}" y="{y}" fill="#6ee7b7" font-family="Arial, Helvetica, sans-serif" '
        f'font-size="30" font-weight="700">{escape(value)}</text>'
        f'<text x="{x}" y="{y + 20}" fill="#9aa0aa" font-family="Arial, Helvetica, sans-serif" '
        f'font-size="12">{escape(label)}</text>'
    )


def build():
    blob, model = read_model()
    assert len(blob) == 1579, len(blob)
    payload = base64.urlsafe_b64encode(blob).decode().rstrip("=")
    digest = hashlib.sha256(blob).hexdigest()
    lines = [
        "The visible card is a model card.",
        "The classifier is in the <metadata> element, not drawn.",
        "Open svg.html to parse the vector file and classify.",
    ]
    body = []
    y = 330
    for line in lines:
        body.append(
            f'<text x="60" y="{y}" fill="#e8e6e3" font-family="Arial, Helvetica, sans-serif" '
            f'font-size="15">{escape(line)}</text>'
        )
        y += 24
    svg = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="900" height="560" viewBox="0 0 900 560"
     role="img" aria-labelledby="title desc">
  <title id="title">AI in an SVG - SMS spam model card</title>
  <desc id="desc">A 1,579-byte int8 SMS spam classifier in the tinyai metadata.</desc>
  <rect width="900" height="560" fill="#0f1115"/>
  <rect x="0" y="0" width="900" height="8" fill="#6ee7b7"/>
  <text x="60" y="74" fill="#6ee7b7" font-family="Arial, Helvetica, sans-serif"
        font-size="16" font-weight="700" letter-spacing="1">1,579 B MODEL - 93.54%</text>
  <text x="60" y="140" fill="#e8e6e3" font-family="Arial, Helvetica, sans-serif"
        font-size="46" font-weight="700">AI in an SVG</text>
  <text x="60" y="180" fill="#9aa0aa" font-family="Arial, Helvetica, sans-serif"
        font-size="17">A 160-word SMS spam classifier in the /metadata payload.</text>
  <rect x="60" y="220" width="780" height="1" fill="#232733"/>
  {metric(60, 280, "160", "words")}
  {metric(280, 280, "2", "labels")}
  {metric(500, 280, "1,579 B", "int8 weights")}
  {metric(720, 280, "93.54%", "1,115 held out")}
  <rect x="60" y="320" width="780" height="1" fill="#232733"/>
  {''.join(body)}
  <text x="60" y="480" fill="#9aa0aa" font-family="Arial, Helvetica, sans-serif"
        font-size="13">SMS Spam Collection; Complement NB; int8 weights. Not an LLM.</text>
  <text x="60" y="504" fill="#9aa0aa" font-family="Arial, Helvetica, sans-serif"
        font-size="13">The SVG is a lossless container, not obfuscation.</text>
  <metadata id="tinyai" data-sha256="{digest}">{payload}</metadata>
</svg>
'''
    OUT.write_text(svg, encoding="utf-8", newline="\n")
    print(f"model={len(blob)} B  payload={len(payload)} chars  svg={OUT.stat().st_size} B  "
          f"sha={digest[:16]}...")


if __name__ == "__main__":
    build()
