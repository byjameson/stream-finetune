import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from streamft import chat_server
from streamft.doc_memory import DocMemory


def _post(url, payload):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def test_doc_answer_returns_sources(tmp_path):
    try:
        docs = DocMemory(tmp_path / "docs.db")
    except Exception as exc:
        pytest.skip(f"sqlite fts5 unavailable: {exc}")
    docs.add_text("law.txt", "The capital of France is Paris.")
    answer = chat_server.doc_answer("capital of France", docs)
    assert answer is not None
    assert "Paris" in answer["text"]
    assert answer["sources"][0]["source"] == "law.txt"
    assert chat_server.doc_answer("nothing here", docs) is None
    docs.close()


def test_http_document_endpoints_and_grounded_chat(tmp_path):
    try:
        docs = DocMemory(tmp_path / "docs.db")
    except Exception as exc:
        pytest.skip(f"sqlite fts5 unavailable: {exc}")
    docs.add_text("law.txt", "The capital of France is Paris.",
                  title="Notes")
    old = dict(chat_server._state)
    chat_server._state.update({
        "model": None, "tok": None, "docs": docs, "adapter": "",
        "name": "tiny-test", "params": 1, "config": {}, "step": 0,
        "api_key": "", "metrics": {"requests": 0, "generated_tokens": 0},
        "user_id": 0, "assistant_id": 1,
    })
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), chat_server.Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        with urllib.request.urlopen(base + "/health", timeout=10) as r:
            health = json.loads(r.read().decode("utf-8"))
        assert health["documents"]["sources"] == 1
        with urllib.request.urlopen(base + "/v1/documents", timeout=10) as r:
            sources = json.loads(r.read().decode("utf-8"))
        assert sources["data"][0]["source"] == "law.txt"
        search = _post(base + "/v1/documents/search",
                       {"query": "capital France"})
        assert "Paris" in search["data"][0]["chunk"]
        chat = _post(base + "/v1/chat/completions", {
            "messages": [{"role": "user",
                          "content": "what is the capital of France?"}],
            "max_tokens": 20,
        })
        assert "Paris" in chat["choices"][0]["message"]["content"]
        assert chat["sources"][0]["source"] == "law.txt"
    finally:
        httpd.shutdown()
        httpd.server_close()
        chat_server._state.clear()
        chat_server._state.update(old)
        docs.close()
