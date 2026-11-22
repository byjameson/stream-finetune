import json
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from atomzero import web_research
from atomzero.doc_memory import DocMemory
from atomzero.web_research import html_to_text, index_url, wikipedia_research


class _Page(BaseHTTPRequestHandler):
    def do_GET(self):
        body = (b"<html><head><title>Demo page</title></head>"
                b"<body><h1>Blue Lantern</h1>"
                b"<p>The project codename is Blue Lantern.</p></body></html>")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        return


def test_html_to_text_extracts_title_and_body():
    title, text = html_to_text(
        "<html><title>T</title><body><p>Hello <b>world</b></p></body></html>")
    assert title == "T"
    assert "Hello" in text and "world" in text


def test_index_url_stores_source_and_search(tmp_path):
    try:
        docs = DocMemory(tmp_path / "docs.db")
    except Exception as exc:
        pytest.skip(f"sqlite fts5 unavailable: {exc}")
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Page)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{httpd.server_address[1]}/"
    try:
        page = index_url(docs, url, allow_private=True)
        assert page["title"] == "Demo page"
        assert page["chunks"] >= 1
        hits = docs.search("Blue Lantern")
        assert hits and "Lantern" in hits[0]["chunk"]
        assert hits[0]["source"] == url
        with pytest.raises(ValueError):
            index_url(docs, url, allow_private=False)
    finally:
        httpd.shutdown()
        httpd.server_close()
        docs.close()


class _ApiResponse:
    def __init__(self, payload):
        self.payload = json.dumps(payload).encode("utf-8")

    def read(self):
        return self.payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_wikipedia_research_uses_intro_and_keeps_source(tmp_path, monkeypatch):
    try:
        docs = DocMemory(tmp_path / "docs.db")
    except Exception as exc:
        pytest.skip(f"sqlite fts5 unavailable: {exc}")
    calls = []

    def fake_urlopen(request, timeout=20):
        url = request.full_url if hasattr(request, "full_url") else str(request)
        calls.append(url)
        query = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        if query.get("list") == ["search"]:
            return _ApiResponse({"query": {"search": [
                {"pageid": 123, "title": "Ankara"}]}})
        return _ApiResponse({"query": {"pages": {"123": {
            "extract": "Ankara is the capital of Turkey. It is in Anatolia."}}}})

    monkeypatch.setattr(web_research.urllib.request, "urlopen", fake_urlopen)
    pages = wikipedia_research(docs, "Ankara", limit=1)
    assert pages[0]["source"] == "https://en.wikipedia.org/?curid=123"
    assert pages[0]["title"] == "Ankara"
    extract_url = [url for url in calls if "prop=extracts" in url][0]
    assert urllib.parse.parse_qs(
        urllib.parse.urlparse(extract_url).query)["exintro"] == ["1"]
    hits = docs.search("capital Turkey")
    assert hits and hits[0]["source"] == pages[0]["source"]
    docs.close()


def test_wikipedia_research_prefers_exact_title(tmp_path, monkeypatch):
    try:
        docs = DocMemory(tmp_path / "docs.db")
    except Exception as exc:
        pytest.skip(f"sqlite fts5 unavailable: {exc}")
    calls = []

    def fake_urlopen(request, timeout=20):
        url = request.full_url if hasattr(request, "full_url") else str(request)
        query = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        calls.append(query)
        if query.get("list") == ["search"]:
            return _ApiResponse({"query": {"search": [
                {"pageid": 2, "title": "New Ankara Stadium"},
                {"pageid": 1, "title": "Ankara"},
            ]}})
        pageid = query["pageids"][0]
        text = ("Ankara is the capital of Turkey. It is in Anatolia."
                if pageid == "1" else "A stadium in Ankara with a roof.")
        return _ApiResponse({"query": {"pages": {pageid: {"extract": text}}}})

    monkeypatch.setattr(web_research.urllib.request, "urlopen", fake_urlopen)
    pages = wikipedia_research(docs, "Ankara", limit=1)
    assert pages[0]["title"] == "Ankara"
    assert pages[0]["source"].endswith("curid=1")
    extract_calls = [call for call in calls if call.get("prop") == ["extracts"]]
    assert len(extract_calls) == 1
    assert extract_calls[0]["pageids"] == ["1"]
    docs.close()
