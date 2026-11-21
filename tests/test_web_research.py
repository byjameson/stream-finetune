import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from streamft.doc_memory import DocMemory
from streamft.web_research import html_to_text, index_url


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
