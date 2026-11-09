"""Measure which containers survive the transformations a normal workflow applies.

This is not a security test and not a phone-camera test. Every check decodes the model
from the transformed file with the same rules the browser pages use. A row with
`ok = null` was not measured (for example PDF/A conversion or Google Sheets).

    python demo/survival_test.py

Writes `demo/survival.json` and `research/results/survival.json`.
"""
import base64
import hashlib
import io
import json
import platform
import re
import shutil
import struct
import subprocess
import tempfile
import wave
from datetime import date
from pathlib import Path

from openpyxl import load_workbook
from PIL import Image, ImageFilter
from pypdf import PdfReader, PdfWriter
import zxingcpp

try:
    import imageio_ffmpeg
except Exception:  # pragma: no cover - optional demo dependency
    imageio_ffmpeg = None

ROOT = Path(__file__).resolve().parent.parent
DEMO = ROOT / "demo"
MODEL_BYTES = 1579
ROWS = []


def add(container, operation, ok, detail):
    ROWS.append({"container": container, "operation": operation,
                 "ok": None if ok is None else bool(ok), "detail": detail})


# ---------------------------------------------------------------- decoders

def png_blob(path):
    data = Path(path).read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n", "not a PNG"
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
    assert (width, height, depth, color) == (32, 32, 8, 2), \
        f"IHDR {width}x{height} depth={depth} color={color}"
    raw = zlib_decompress(bytes(idat))
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
            assert filt == 0, f"PNG filter {filt}"
        out += line
        prev = line
    length = struct.unpack("<I", out[:4])[0]
    assert 4 + length <= len(out), f"length prefix {length}"
    return bytes(out[4:4 + length])


def zlib_decompress(data):
    import zlib
    return zlib.decompress(data)


def wav_data(path):
    data = Path(path).read_bytes()
    assert data[:4] == b"RIFF" and data[8:12] == b"WAVE"
    pos, payload = 12, None
    while pos + 8 <= len(data):
        kind = data[pos:pos + 4]
        size = struct.unpack("<I", data[pos + 4:pos + 8])[0]
        if kind == b"data":
            payload = data[pos + 8:pos + 8 + size]
            break
        pos += 8 + size + (size & 1)
    assert payload is not None
    return payload


def wav_blob(path):
    payload = wav_data(path)
    length = struct.unpack("<I", payload[:4])[0]
    assert length == MODEL_BYTES, length
    return payload[4:4 + length]


def pdf_blob(path):
    data = Path(path).read_bytes()
    match = re.search(rb"/Keywords\s*\(([^)]*)\)", data)
    if not match:
        raise ValueError("no /Keywords (...) metadata")
    payload = match.group(1)
    payload += b"=" * (-len(payload) % 4)
    return base64.urlsafe_b64decode(payload)


def pdf_hash(path):
    match = re.search(rb"/TinyAISHA256\s*\(([0-9a-fA-F]{64})\)", Path(path).read_bytes())
    return match.group(1).decode().lower() if match else None


def xlsx_ok(path, reference=None):
    wb = load_workbook(path, data_only=False)
    ws = wb["model"]
    if (ws.max_row, ws.max_column) != (161, 3):
        return False, f"model sheet is {ws.max_row}x{ws.max_column}"
    if reference is not None:
        ref = load_workbook(reference, data_only=False)
        rws = ref["model"]
        same = all(ws.cell(r, c).value == rws.cell(r, c).value
                   for r in range(1, 162) for c in range(1, 4))
        if not same:
            return False, "model cells changed"
        rconf = ref["classify"]
        for cell in ("B2", "B3", "B4"):
            if ws.parent["classify"][cell].value != rconf[cell].value:
                return False, f"formula {cell} changed"
    formula = wb["classify"]["B2"].value
    if not (isinstance(formula, str) and "SUMPRODUCT" in formula):
        return False, "SUMPRODUCT formula missing"
    return True, "160 words + 3 formulas"


def html_ok(path):
    text = Path(path).read_text(encoding="utf-8")
    if "trainCNB" not in text:
        raise ValueError("no trainCNB")
    match = re.search(r"const DATA = (\{.*?\});", text, re.S)
    if not match:
        raise ValueError("no DATA")
    data = json.loads(match.group(1))
    if len(data["rows"]) < 6:
        raise ValueError("too few rows")
    return len(data["rows"]), data["labels"]


# ---------------------------------------------------------------- checks

def main():
    tmp = Path(tempfile.mkdtemp(prefix="survival-"))
    try:
        source_blob = pdf_blob(DEMO / "model-card.pdf")
        assert len(source_blob) == MODEL_BYTES
        assert source_blob == wav_blob(DEMO / "model.wav")
        assert source_blob == png_blob(DEMO / "favicon.png")

        # PDF
        src_pdf = DEMO / "model-card.pdf"
        expected_hash = hashlib.sha256(source_blob).hexdigest()
        add("PDF", "SHA-256 field", pdf_hash(src_pdf) == expected_hash,
            "full 64-hex digest in /TinyAISHA256")
        add("PDF", "byte copy", pdf_blob(_copy(src_pdf, tmp / "pdf-copy.pdf")) == source_blob,
            f"1,579 B model / {src_pdf.stat().st_size:,} B PDF")
        try:
            reader = PdfReader(str(src_pdf))
            writer = PdfWriter(clone_from=reader)
            writer.write(str(tmp / "pdf-clone.pdf"))
            ok = pdf_blob(tmp / "pdf-clone.pdf") == source_blob
            ok = ok and pdf_hash(tmp / "pdf-clone.pdf") == expected_hash
            add("PDF", "pypdf rewrite", ok,
                "1,579 B model; metadata + SHA-256 preserved" if ok else "model or hash lost")
        except Exception as exc:
            add("PDF", "pypdf rewrite", False, f"{type(exc).__name__}: {exc}")
        try:
            reader = PdfReader(str(DEMO / "model-card.pdf"))
            writer = PdfWriter()
            for page in reader.pages:
                writer.add_page(page)
            writer.write(str(tmp / "pdf-nometa.pdf"))
            try:
                got = pdf_blob(tmp / "pdf-nometa.pdf")
                add("PDF", "rebuild without Info (print-to-PDF proxy)", len(got) == MODEL_BYTES,
                    f"{len(got)} B model")
            except Exception:
                add("PDF", "rebuild without Info (print-to-PDF proxy)", False,
                    "no /Keywords or /TinyAISHA256")
        except Exception as exc:
            add("PDF", "rebuild without Info (print-to-PDF proxy)", False, f"{type(exc).__name__}: {exc}")
        add("PDF", "PDF/A or editor resave", None, "not measured; needs a PDF editor/renderer")

        # XLSX
        ok, detail = xlsx_ok(DEMO / "spreadsheet.xlsx")
        add("XLSX", "byte copy", ok, detail)
        try:
            wb = load_workbook(DEMO / "spreadsheet.xlsx")
            wb.save(tmp / "xlsx-resave.xlsx")
            ok, detail = xlsx_ok(tmp / "xlsx-resave.xlsx", DEMO / "spreadsheet.xlsx")
            add("XLSX", "openpyxl load/save", ok,
                "483 cells + 3 formulas identical" if ok else detail)
        except Exception as exc:
            add("XLSX", "openpyxl load/save", False, f"{type(exc).__name__}: {exc}")
        add("XLSX", "Google Sheets round-trip", None, "not measured; may rewrite formulas")

        # PNG
        try:
            ok = png_blob(_copy(DEMO / "favicon.png", tmp / "png-copy.png")) == source_blob
            add("PNG", "byte copy", ok, "1,579 B model / 1,519 B PNG")
        except Exception as exc:
            add("PNG", "byte copy", False, f"{type(exc).__name__}: {exc}")
        im = Image.open(DEMO / "favicon.png").convert("RGB")
        im.save(tmp / "png-resave.png", optimize=True)
        try:
            add("PNG", "PNG re-save", png_blob(tmp / "png-resave.png") == source_blob, "pixels exact")
        except Exception as exc:
            add("PNG", "PNG re-save", False, f"{type(exc).__name__}: {exc}")
        Image.frombytes("RGB", im.size, im.tobytes()).save(tmp / "png-strip.png")
        try:
            add("PNG", "metadata strip", png_blob(tmp / "png-strip.png") == source_blob,
                "model is in pixels, not metadata")
        except Exception as exc:
            add("PNG", "metadata strip", False, f"{type(exc).__name__}: {exc}")
        im.save(tmp / "png-jpeg.jpg", "JPEG", quality=80)
        try:
            got = png_blob(tmp / "png-jpeg.jpg")
            add("PNG", "JPEG q80", got == source_blob, f"{len(got)} B model")
        except Exception as exc:
            add("PNG", "JPEG q80", False, f"{type(exc).__name__}: {exc}")
        im.resize((16, 16), Image.Resampling.LANCZOS).save(tmp / "png-16.png")
        try:
            got = png_blob(tmp / "png-16.png")
            add("PNG", "resize to 16x16", got == source_blob, f"{len(got)} B model")
        except Exception as exc:
            add("PNG", "resize to 16x16", False, f"{type(exc).__name__}: {exc}")

        # WAV
        base = wav_data(DEMO / "model.wav")
        try:
            add("WAV", "byte copy", wav_data(_copy(DEMO / "model.wav", tmp / "wav-copy.wav")) == base,
                "1,583 B data / 1,627 B WAV")
        except Exception as exc:
            add("WAV", "byte copy", False, f"{type(exc).__name__}: {exc}")
        with wave.open(str(DEMO / "model.wav"), "rb") as src:
            params = src.getparams()
            frames = src.readframes(src.getnframes())
        with wave.open(str(tmp / "wav-rewrite.wav"), "wb") as dst:
            dst.setparams(params)
            dst.writeframes(frames)
        add("WAV", "PCM rewrite", wav_data(tmp / "wav-rewrite.wav") == base, "samples exact")

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe() if imageio_ffmpeg else None
        if ffmpeg:
            for name, suffix, codec in (
                ("FLAC (lossless)", ".flac", ["-c:a", "flac"]),
                ("MP3 128k", ".mp3", ["-c:a", "libmp3lame", "-b:a", "128k"]),
                ("Opus 32k", ".ogg", ["-c:a", "libopus", "-b:a", "32k"]),
            ):
                try:
                    mid = tmp / f"wav-mid{suffix}"
                    out = tmp / f"wav-{name.split()[0]}.wav"
                    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i",
                                    str(DEMO / "model.wav"), *codec, str(mid)],
                                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(mid),
                                    "-c:a", "pcm_u8", str(out)],
                                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                    payload = wav_data(out)
                    length = struct.unpack("<I", payload[:4])[0]
                    ok = payload == base
                    add("WAV", name, ok,
                        "samples exact" if ok else f"data {len(payload)} B, length {length}")
                except Exception as exc:
                    add("WAV", name, False, f"{type(exc).__name__}: {exc}")
        else:
            add("WAV", "FLAC / MP3 / Opus", None, "ffmpeg not installed")

        # QR
        qr = Image.open(DEMO / "qr-spam.png").convert("L")
        url = (DEMO / "qr-url.txt").read_text(encoding="utf-8").strip()

        def qr_ok(image):
            result = zxingcpp.read_barcode(image)
            return bool(result and result.text == url)

        add("QR", "as-is", qr_ok(qr), f"{len(url)} B URL, exact")
        small = qr.resize((round(qr.width * 0.1), round(qr.height * 0.1)),
                          Image.Resampling.LANCZOS)
        add("QR", "downscale to 10% (161 px)", qr_ok(small), "exact URL")
        buf = io.BytesIO()
        qr.save(buf, "JPEG", quality=5)
        buf.seek(0)
        add("QR", "JPEG q5", qr_ok(Image.open(buf)), "exact URL")
        add("QR", "blur 4 px", qr_ok(qr.filter(ImageFilter.GaussianBlur(4))), "exact URL")
        add("QR", "rotate 20 degrees", qr_ok(qr.rotate(20, expand=True, fillcolor=255)), "exact URL")
        add("QR", "drop #fragment (link shortener)", False, "no #m=; qr.html has no model")
        add("QR", "phone camera / print", None, "not measured; decoder test only")

        # Single HTML
        try:
            rows, labels = html_ok(DEMO / "tiny-ai-sms.html")
            add("HTML", "byte copy", True, f"{rows} examples, {len(labels)} labels; trainCNB present")
        except Exception as exc:
            add("HTML", "byte copy", False, f"{type(exc).__name__}: {exc}")
        try:
            text = (DEMO / "tiny-ai-sms.html").read_text(encoding="utf-8")
            stripped = re.sub(r"<script\b[^>]*>.*?</script>", "", text, flags=re.S | re.I)
            (tmp / "html-strip.html").write_text(stripped, encoding="utf-8")
            try:
                html_ok(tmp / "html-strip.html")
                add("HTML", "strip <script> (simulated CMS)", True, "classifier still present")
            except Exception:
                add("HTML", "strip <script> (simulated CMS)", False, "no classifier or data")
        except Exception as exc:
            add("HTML", "strip <script> (simulated CMS)", False, f"{type(exc).__name__}: {exc}")
        add("Prompt", "copy", None, "not self-contained; needs a shared base")

        result = {
            "generated": date.today().isoformat(),
            "model_bytes": len(source_blob),
            "model_sha256": hashlib.sha256(source_blob).hexdigest(),
            "machine": platform.platform(),
            "rows": ROWS,
        }
        text = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
        (DEMO / "survival.json").write_text(text, encoding="utf-8")
        (ROOT / "research/results/survival.json").write_text(text, encoding="utf-8")

        print(f"model {len(source_blob)} B  sha256 {result['model_sha256'][:16]}...")
        for row in ROWS:
            mark = "n/a " if row["ok"] is None else ("PASS" if row["ok"] else "FAIL")
            print(f"{mark} {row['container']:5s} {row['operation']:48s} {row['detail']}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _copy(src, dst):
    shutil.copyfile(src, dst)
    return dst


if __name__ == "__main__":
    main()
