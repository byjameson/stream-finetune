import hashlib
import io

import pytest

from atomzero.doc_ingest import DocumentExtractError, extract_bytes
from atomzero.doc_memory import DocMemory


def _pdf_bytes(text: str = "The launch code is L-7781.") -> bytes:
    """Build a tiny one-page PDF without a PDF-writing dependency."""
    safe = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream = f"BT /F1 12 Tf 72 720 Td ({safe}) Tj ET".encode("latin-1")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
         b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>"),
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n"
        + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode("ascii") + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode("ascii")
    out += b"0000000000 65535 f \n"
    for offset in offsets[1:]:
        out += f"{offset:010d} 00000 n \n".encode("ascii")
    out += (f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref}\n%%EOF\n").encode("ascii")
    return bytes(out)


def test_markdown_blocks_keep_sections():
    extracted = extract_bytes(
        "notes.md",
        b"# Warranty\n\nThe code is W-7781.\n\n## Travel\n\nBook by Friday.",
    )
    assert extracted["title"] == "notes"
    assert [block["section"] for block in extracted["blocks"]] == [
        "Warranty", "Travel"]
    assert "W-7781" in extracted["blocks"][0]["text"]
    assert extracted["blocks"][1]["page"] is None


def test_html_blocks_keep_title_and_sections():
    extracted = extract_bytes(
        "page.html",
        b"<html><head><title>Policy</title></head><body>"
        b"<h1>Warranty</h1><p>The code is W-7781.</p>"
        b"<h2>Travel</h2><p>Book by Friday.</p></body></html>",
    )
    assert extracted["title"] == "Policy"
    assert [block["section"] for block in extracted["blocks"]] == [
        "Warranty", "Travel"]
    assert "Friday" in extracted["blocks"][1]["text"]


def test_pdf_blocks_keep_page_numbers():
    extracted = extract_bytes("manual.pdf", _pdf_bytes())
    assert extracted["format"] == "pdf"
    assert len(extracted["blocks"]) == 1
    assert extracted["blocks"][0]["page"] == 1
    assert "L-7781" in extracted["blocks"][0]["text"]


def test_docx_blocks_keep_sections():
    docx = pytest.importorskip("docx")
    document = docx.Document()
    document.add_heading("Warranty", level=1)
    document.add_paragraph("The code is W-7781.")
    document.add_heading("Travel", level=1)
    document.add_paragraph("Book by Friday.")
    buffer = io.BytesIO()
    document.save(buffer)

    extracted = extract_bytes("manual.docx", buffer.getvalue())
    assert extracted["format"] == "docx"
    assert [block["section"] for block in extracted["blocks"]] == [
        "Warranty", "Travel"]
    assert "Friday" in extracted["blocks"][1]["text"]


def test_docx_tables_keep_document_order_and_section():
    docx = pytest.importorskip("docx")
    document = docx.Document()
    document.add_heading("Warranty", level=1)
    document.add_paragraph("Before table.")
    table = document.add_table(rows=1, cols=2)
    table.cell(0, 0).text = "Code"
    table.cell(0, 1).text = "AC-"
    document.add_paragraph("After table.")
    document.add_heading("Travel", level=1)
    document.add_paragraph("Book by Friday.")
    buffer = io.BytesIO()
    document.save(buffer)

    extracted = extract_bytes("manual.docx", buffer.getvalue())
    texts = [block["text"] for block in extracted["blocks"]]
    assert [block["section"] for block in extracted["blocks"]] == [
        "Warranty", "Warranty", "Warranty", "Travel"]
    assert texts[0] == "Before table."
    assert "Code | AC-" in texts[1]
    assert texts[2] == "After table."
    assert "Friday" in texts[3]


def test_unsupported_empty_document_is_rejected():
    with pytest.raises(DocumentExtractError):
        extract_bytes("empty.pdf", b"not a pdf")
    with pytest.raises(DocumentExtractError):
        extract_bytes("empty.txt", b"   \n\n  ")
    with pytest.raises(DocumentExtractError):
        extract_bytes("payload.exe", b"MZ\x90\x00")


def test_add_file_markdown_stores_section_and_citation(tmp_path):
    docs = DocMemory(tmp_path / "docs.db")
    path = tmp_path / "policy.md"
    path.write_text(
        "# Warranty\n\nThe warranty code is W-7781.\n\n"
        "# Travel\n\nBook the train by Friday.\n",
        encoding="utf-8",
    )
    try:
        assert docs.add_file(path) == 2
        hits = docs.search("warranty code", k=2)
        assert hits
        assert hits[0]["section"] == "Warranty"
        assert hits[0]["page"] is None
        source = docs.sources()[0]
        assert source["title"] == "policy"
        assert len(source["content_sha256"]) == 64
        assert "page_start" not in source
    finally:
        docs.close()


def test_add_blocks_keeps_page_and_section_and_replaces(tmp_path):
    docs = DocMemory(tmp_path / "docs.db")
    try:
        assert docs.add_blocks("manual.pdf", [
            {"text": "The launch code is L-7781.", "page": 1,
             "section": "Launch"},
            {"text": "The warranty code is W-9900.", "page": 2,
             "section": "Warranty"},
        ], title="Manual") == 2
        first = docs.search("launch code", k=1)[0]
        assert first["page"] == 1
        assert first["section"] == "Launch"
        second = docs.search("warranty code", k=1)[0]
        assert second["page"] == 2
        assert second["section"] == "Warranty"
        assert docs.sources()[0]["page_start"] == 1
        assert docs.sources()[0]["page_end"] == 2
        assert len(docs.sources()[0]["content_sha256"]) == 64

        docs.add_blocks("manual.pdf", [
            {"text": "The launch code is L-0000.", "page": 7,
             "section": "Launch"},
        ], title="Manual")
        assert docs.search("L-7781", k=1) == []
        replacement = docs.search("launch code", k=1)[0]
        assert replacement["page"] == 7
        count = docs.con.execute(
            "SELECT COUNT(*) FROM doc_chunk_meta").fetchone()[0]
        assert count == 1

        evidence = docs.source_evidence("manual.pdf", section="Launch")
        assert evidence["returned"] == 1
        assert evidence["data"][0]["page"] == 7
        with pytest.raises(ValueError):
            docs.source_evidence("manual.pdf", page=0)
        with pytest.raises(ValueError):
            docs.source_evidence("missing.pdf")

        docs.forget("manual.pdf")
        assert docs.con.execute(
            "SELECT COUNT(*) FROM doc_chunk_meta").fetchone()[0] == 0
    finally:
        docs.close()


def test_ingest_upload_pdf_returns_page_metadata(tmp_path):
    from atomzero.chat_server import ingest_upload

    docs = DocMemory(tmp_path / "docs.db")
    try:
        pdf_data = _pdf_bytes()
        result = ingest_upload(docs, "manual.pdf", pdf_data, "Manual")
        assert result["format"] == "pdf"
        assert result["pages"] == 1
        assert result["file_sha256"] == hashlib.sha256(pdf_data).hexdigest()
        assert result["sha256"] == result["file_sha256"]
        assert result["content_sha256"] != result["file_sha256"]
        hit = docs.search("launch code", k=1)[0]
        assert hit["page"] == 1
        assert "L-7781" in hit["chunk"]
    finally:
        docs.close()


def test_http_raw_upload_over_json_body_limit(tmp_path):
    import json
    import threading
    import urllib.parse
    import urllib.request
    from http.server import ThreadingHTTPServer

    from atomzero import chat_server

    docs = DocMemory(tmp_path / "docs.db")
    old = dict(chat_server._state)
    chat_server._state.update({"docs": docs, "metrics": {}})
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), chat_server.Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        payload = ("The archive code is A-42. " * 100_000).encode("utf-8")
        assert len(payload) > chat_server.MAX_BODY_BYTES
        assert len(payload) < chat_server.MAX_DOC_BYTES
        query = urllib.parse.urlencode({"source": "large.txt"})
        request = urllib.request.Request(
            base + "/v1/documents/upload?" + query,
            data=payload,
            headers={"Content-Type": "text/plain"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            uploaded = json.loads(response.read().decode("utf-8"))
        assert uploaded["source"] == "large.txt"
        assert uploaded["chunks"] > 1
        assert uploaded["bytes"] == len(payload)
    finally:
        httpd.shutdown()
        httpd.server_close()
        chat_server._state.clear()
        chat_server._state.update(old)
        docs.close()


def test_http_pdf_upload_returns_page_citation(tmp_path):
    import json
    import threading
    import urllib.error
    import urllib.parse
    import urllib.request
    from http.server import ThreadingHTTPServer

    from atomzero import chat_server

    docs = DocMemory(tmp_path / "docs.db")
    old = dict(chat_server._state)
    chat_server._state.update({"docs": docs, "metrics": {}})
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), chat_server.Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        query = urllib.parse.urlencode(
            {"source": "manual.pdf", "title": "Manual"})
        pdf_data = _pdf_bytes()
        request = urllib.request.Request(
            base + "/v1/documents/upload?" + query,
            data=pdf_data,
            headers={"Content-Type": "application/pdf"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            uploaded = json.loads(response.read().decode("utf-8"))
        assert uploaded["format"] == "pdf"
        assert uploaded["pages"] == 1
        assert uploaded["chunks"] == 1
        assert uploaded["file_sha256"] == hashlib.sha256(pdf_data).hexdigest()
        assert uploaded["sha256"] == uploaded["file_sha256"]
        assert uploaded["content_sha256"] != uploaded["file_sha256"]
        assert (uploaded["content_sha256"]
                == docs.sources()[0]["content_sha256"])

        request = urllib.request.Request(
            base + "/v1/documents/search",
            data=json.dumps({"query": "launch code"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            hit = json.loads(response.read().decode("utf-8"))["data"][0]
        assert hit["page"] == 1
        assert "L-7781" in hit["chunk"]

        evidence_query = urllib.parse.urlencode(
            {"source": "manual.pdf", "page": 1})
        with urllib.request.urlopen(
                base + "/v1/documents/evidence?" + evidence_query,
                timeout=10) as response:
            evidence = json.loads(response.read().decode("utf-8"))
        assert evidence["returned"] == 1
        assert evidence["data"][0]["page"] == 1
        assert "L-7781" in evidence["data"][0]["chunk"]

        with pytest.raises(urllib.error.HTTPError) as missing:
            urllib.request.urlopen(
                base + "/v1/documents/evidence?source=missing.pdf",
                timeout=10)
        assert missing.value.code == 404
    finally:
        httpd.shutdown()
        httpd.server_close()
        chat_server._state.clear()
        chat_server._state.update(old)
        docs.close()
