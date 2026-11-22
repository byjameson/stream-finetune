"""Extract text blocks with page/section provenance from local documents.

No network and no OCR. The extractor keeps the source structure that the
retrieval layer needs for citations:

* PDF: one block per page (1-based ``page``)
* DOCX/Markdown/HTML: blocks grouped under their current heading
  (``section``)

Binary formats are optional at import time so the rest of the package keeps
working when only the text path is installed.
"""
from __future__ import annotations

import io
import re
from html.parser import HTMLParser
from pathlib import Path

TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".html", ".htm"}
SUPPORTED_SUFFIXES = TEXT_SUFFIXES | {".pdf", ".docx"}


class DocumentExtractError(ValueError):
    """Raised when a document cannot be converted into usable text."""


def _clean(text: str) -> str:
    text = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _block(text: str, page: int | None = None,
           section: str = "") -> dict | None:
    text = _clean(text)
    if not text:
        return None
    return {
        "text": text,
        "page": page,
        "section": _clean(section).replace("\n", " "),
    }


def _markdown_blocks(text: str) -> list[dict]:
    blocks: list[dict] = []
    section = ""
    current: list[str] = []
    in_fence = False

    def flush() -> None:
        nonlocal current
        block = _block("\n".join(current), section=section)
        if block:
            blocks.append(block)
        current = []

    for line in _clean(text).splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            current.append(line)
            continue
        heading = (None if in_fence
                   else re.match(r"^(#{1,6})\s+(.+?)\s*#*\s*$", line))
        if heading:
            flush()
            section = heading.group(2).strip()
            continue
        current.append(line)
    flush()
    if not blocks:
        block = _block(text)
        if block:
            blocks.append(block)
    return blocks


class _HTMLBlocks(HTMLParser):
    """Small block-level HTML reader; no third-party parser required."""

    _BLOCKS = {"p", "li", "div", "section", "article", "tr", "td", "th"}
    _HEADINGS = {"h1", "h2", "h3", "h4", "h5", "h6"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title = ""
        self.blocks: list[dict] = []
        self._skip = 0
        self._in_title = False
        self._heading_tag = ""
        self._heading_parts: list[str] = []
        self._section = ""
        self._buffer: list[str] = []

    def _flush(self) -> None:
        text = " ".join(" ".join(self._buffer).split())
        if text:
            self.blocks.append({
                "text": text, "page": None, "section": self._section})
        self._buffer = []

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag in ("script", "style", "noscript", "svg"):
            self._skip += 1
        elif tag == "title":
            self._in_title = True
        elif tag in self._HEADINGS:
            self._flush()
            self._heading_tag = tag
            self._heading_parts = []
        elif tag in self._BLOCKS or tag == "br":
            self._flush()

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag in ("script", "style", "noscript", "svg"):
            if self._skip:
                self._skip -= 1
        elif tag == "title":
            self._in_title = False
        elif tag == self._heading_tag:
            heading = " ".join("".join(self._heading_parts).split())
            if heading:
                self._section = heading
            self._heading_tag = ""
            self._heading_parts = []
            self._flush()
        elif tag in self._BLOCKS:
            self._flush()

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif self._skip:
            return
        elif self._heading_tag:
            self._heading_parts.append(data)
        else:
            self._buffer.append(data)

    def finish(self) -> tuple[str, list[dict]]:
        self._flush()
        return self.title.strip(), self.blocks


def _html_blocks(html: str) -> tuple[str, list[dict]]:
    parser = _HTMLBlocks()
    parser.feed(html)
    parser.close()
    return parser.finish()


def _pdf_blocks(data: bytes) -> tuple[str, list[dict]]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - declared dependency
        raise DocumentExtractError(
            "pypdf is required for PDF ingestion") from exc
    try:
        reader = PdfReader(io.BytesIO(data))
    except Exception as exc:
        raise DocumentExtractError(f"cannot read PDF: {exc}") from exc
    if getattr(reader, "is_encrypted", False):
        try:
            reader.decrypt("")
        except Exception as exc:
            raise DocumentExtractError(
                "PDF is encrypted and cannot be read") from exc
    blocks: list[dict] = []
    for number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        block = _block(text, page=number)
        if block:
            blocks.append(block)
    if not blocks:
        raise DocumentExtractError(
            "PDF has no extractable text; scanned PDFs need OCR")
    title = ""
    try:
        metadata = reader.metadata
        if metadata is not None:
            title = str(metadata.title or "")
    except Exception:
        title = ""
    return title, blocks


def _docx_blocks(data: bytes) -> tuple[str, list[dict]]:
    try:
        from docx import Document
    except ImportError as exc:  # pragma: no cover - declared dependency
        raise DocumentExtractError(
            "python-docx is required for DOCX ingestion") from exc
    try:
        document = Document(io.BytesIO(data))
    except Exception as exc:
        raise DocumentExtractError(f"cannot read DOCX: {exc}") from exc
    from docx.oxml.ns import qn
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    blocks: list[dict] = []
    section = ""
    current: list[str] = []

    def flush() -> None:
        nonlocal current
        block = _block("\n\n".join(current), section=section)
        if block:
            blocks.append(block)
        current = []

    def iter_blocks():
        body = document.element.body
        for child in body.iterchildren():
            if child.tag == qn("w:p"):
                yield Paragraph(child, document)
            elif child.tag == qn("w:tbl"):
                yield Table(child, document)

    for item in iter_blocks():
        if isinstance(item, Paragraph):
            text = (item.text or "").strip()
            if not text:
                continue
            style = ""
            try:
                style = (item.style.name or "").lower()
            except Exception:
                style = ""
            if style.startswith("heading"):
                flush()
                section = text
                continue
            current.append(text)
            if sum(len(part) for part in current) > 1800:
                flush()
            continue

        # A table keeps its position relative to the paragraphs around it.
        flush()
        rows = []
        for row in item.rows:
            cells = [(cell.text or "").strip() for cell in row.cells]
            if any(cells):
                rows.append(" | ".join(cells))
        block = _block("\n".join(rows), section=section or "Table")
        if block:
            blocks.append(block)
    flush()

    if not blocks:
        raise DocumentExtractError("DOCX has no extractable text")
    title = ""
    try:
        title = str(document.core_properties.title or "")
    except Exception:
        title = ""
    return title, blocks


def extract_bytes(filename: str, data: bytes,
                  title: str = "") -> dict:
    """Convert file bytes into ``title``, ``format`` and provenance blocks."""
    name = Path(str(filename or "document")).name
    suffix = Path(name).suffix.lower()
    fallback = Path(name).stem or "document"
    if suffix == ".pdf":
        meta_title, blocks = _pdf_blocks(data)
    elif suffix == ".docx":
        meta_title, blocks = _docx_blocks(data)
    elif suffix in (".html", ".htm"):
        html = data.decode("utf-8-sig", errors="replace")
        meta_title, blocks = _html_blocks(html)
    elif suffix in (".md", ".markdown"):
        text = data.decode("utf-8-sig", errors="replace")
        meta_title, blocks = "", _markdown_blocks(text)
    elif not suffix or suffix == ".txt":
        # Extension-less uploads keep the old UTF-8 text path.
        text = data.decode("utf-8-sig", errors="replace")
        meta_title, blocks = "", [_block(text)]
    else:
        raise DocumentExtractError(
            f"unsupported document type: {suffix}; "
            "use .txt, .md, .html, .pdf or .docx")
    blocks = [block for block in blocks if block]
    if not blocks:
        raise DocumentExtractError(
            f"document has no extractable text: {name}")
    return {
        "title": str(title or "").strip() or meta_title.strip() or fallback,
        "blocks": blocks,
        "format": suffix.lstrip(".") or "text",
        "source": name,
    }


def extract_file(path: str | Path, title: str = "") -> dict:
    p = Path(path)
    try:
        data = p.read_bytes()
    except OSError as exc:
        raise DocumentExtractError(f"cannot read {p}: {exc}") from exc
    return extract_bytes(p.name, data, title=title)
