import json
import threading
import urllib.error
import urllib.parse
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from streamft import chat_server
from streamft.doc_memory import DocMemory
from streamft.research_queue import ResearchQueue


def _post(url, payload):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def _post_raw(url, body):
    req = urllib.request.Request(
        url, data=body.encode("utf-8"),
        headers={"Content-Type": "text/plain"}, method="POST")
    with urllib.request.urlopen(req, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def test_demo_page_uses_daily_agent_endpoint():
    page = chat_server.PAGE.lower()
    assert "/v1/agent/chat" in page
    assert "/v1/documents" in page
    assert "add a document" in page
    assert "teach a fact" in page
    assert "short story" not in page
    assert "/generate" not in page


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


def test_doc_answer_dedupes_repeated_sources(tmp_path):
    try:
        docs = DocMemory(tmp_path / "docs.db")
    except Exception as exc:
        pytest.skip(f"sqlite fts5 unavailable: {exc}")
    docs.add_text(
        "ankara.txt",
        ("Ankara is a city. " * 30) + "\n\n" + ("Ankara is in Turkey. " * 30),
        title="Ankara")
    answer = chat_server.doc_answer("Ankara", docs)
    assert answer is not None
    assert len(answer["sources"]) == 1
    assert answer["sources"][0]["source"] == "ankara.txt"
    docs.close()


def test_ingest_document_validates_source_and_text(tmp_path):
    try:
        docs = DocMemory(tmp_path / "docs.db")
    except Exception as exc:
        pytest.skip(f"sqlite fts5 unavailable: {exc}")
    with pytest.raises(ValueError):
        chat_server.ingest_document(docs, "", "some text")
    with pytest.raises(ValueError):
        chat_server.ingest_document(docs, "bad\nsource", "some text")
    with pytest.raises(ValueError):
        chat_server.ingest_document(docs, "note.txt", "   ")
    docs.close()


def test_http_document_endpoints_and_grounded_chat(tmp_path, monkeypatch):
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
        "queue_path": str(tmp_path / "queue.jsonl"),
        "allow_url_fetch": True, "wiki_lang": "en",
        "session_dir": str(tmp_path / "sessions"), "sessions": {},
    })
    import streamft.web_research as wr
    monkeypatch.setattr(wr, "index_url", lambda docs, url: {
        "source": url, "title": "T", "chunks": 1, "preview": "p"})
    monkeypatch.setattr(wr, "wikipedia_research", lambda docs, q, lang, limit: [
        {"source": "https://wiki/x", "title": "X", "chunks": 1,
         "preview": "p"}])
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
        taught = _post(base + "/v1/teach",
                       {"fact": "The sky is blue."})
        assert taught["fact_chunks"] == 1
        assert "blue" in _post(
            base + "/v1/documents/search",
            {"query": "sky"})["data"][0]["chunk"]
        queued = _post(base + "/v1/teach",
                       {"user": "good morning",
                        "assistant": "Good morning!"})
        assert queued["skill_queued"] is True
        queue = (tmp_path / "queue.jsonl").read_text(encoding="utf-8")
        assert "good morning" in queue
        ingested = _post(base + "/v1/documents", {
            "source": "warranty.txt", "title": "Warranty",
            "text": "The warranty code is W-7781. Keep the receipt.",
        })
        assert ingested["chunks"] == 1
        assert ingested["title"] == "Warranty"
        assert len(ingested["sha256"]) == 64
        assert "W-7781" in _post(
            base + "/v1/documents/search",
            {"query": "warranty code"})["data"][0]["chunk"]
        warranty_chat = _post(base + "/v1/chat/completions", {
            "messages": [{"role": "user",
                          "content": "what is the warranty code?"}],
            "max_tokens": 20,
        })
        assert "W-7781" in warranty_chat["choices"][0]["message"]["content"]
        assert warranty_chat["sources"][0]["source"] == "warranty.txt"
        replaced = _post(base + "/v1/documents", {
            "source": "warranty.txt",
            "text": "The warranty code is W-9900.",
        })
        assert replaced["chunks"] == 1
        assert _post(base + "/v1/documents/search",
                     {"query": "7781"})["data"] == []
        assert "W-9900" in _post(
            base + "/v1/documents/search",
            {"query": "warranty"})["data"][0]["chunk"]
        raw = _post_raw(
            base + "/v1/documents/upload?" + urllib.parse.urlencode({
                "source": "raw.txt", "title": "Raw upload"}),
            "The raw upload code is R-42.")
        assert raw["source"] == "raw.txt"
        assert raw["chunks"] == 1
        assert "R-42" in _post(
            base + "/v1/documents/search",
            {"query": "raw upload"})["data"][0]["chunk"]
        old_max = chat_server.MAX_DOC_BYTES
        chat_server.MAX_DOC_BYTES = 5
        try:
            with pytest.raises(urllib.error.HTTPError) as too_big:
                _post(base + "/v1/documents",
                      {"source": "big.txt", "text": "123456"})
            assert too_big.value.code == 413
        finally:
            chat_server.MAX_DOC_BYTES = old_max
        with pytest.raises(urllib.error.HTTPError) as missing_source:
            _post(base + "/v1/documents", {"text": "no source"})
        assert missing_source.value.code == 400
        learned = _post(base + "/v1/learn_url",
                        {"url": "https://example.com/page"})
        assert learned["title"] == "T"
        research = _post(base + "/v1/research", {"query": "anything"})
        assert research["pages"][0]["title"] == "X"
        agent = _post(base + "/v1/agent/chat", {
            "session_id": "s1", "message": "add task buy milk"})
        assert "Added task 1" in agent["reply"]
        session_chat = _post(base + "/v1/chat/completions", {
            "session_id": "s1",
            "messages": [{"role": "user", "content": "list tasks"}],
            "max_tokens": 20,
        })
        content = session_chat["choices"][0]["message"]["content"]
        assert "buy milk" in content
    finally:
        httpd.shutdown()
        httpd.server_close()
        chat_server._state.clear()
        chat_server._state.update(old)
        docs.close()


def test_http_research_queue_requires_approval_and_cites(tmp_path, monkeypatch):
    try:
        docs = DocMemory(tmp_path / "docs.db")
    except Exception as exc:
        pytest.skip(f"sqlite fts5 unavailable: {exc}")
    queue = ResearchQueue(tmp_path / "research.json")
    old = dict(chat_server._state)
    chat_server._state.update({
        "model": None, "tok": None, "docs": docs, "adapter": "",
        "name": "tiny-test", "params": 1, "config": {}, "step": 0,
        "api_key": "", "metrics": {"requests": 0, "generated_tokens": 0},
        "user_id": 0, "assistant_id": 1,
        "queue_path": str(tmp_path / "queue.jsonl"),
        "allow_url_fetch": False, "wiki_lang": "en",
        "session_dir": str(tmp_path / "sessions"), "sessions": {},
        "research_queue": queue,
        "research_queue_path": str(tmp_path / "research.json"),
    })
    calls = []

    def fake_research(memory, query, lang, limit):
        calls.append(query)
        source = "https://en.wikipedia.org/?curid=9"
        chunks = memory.add_text(
            source, "Ankara is the capital of Turkey.", title="Ankara")
        return [{"source": source, "title": "Ankara", "chunks": chunks,
                 "preview": "Ankara is the capital of Turkey."}]

    import streamft.web_research as wr
    monkeypatch.setattr(wr, "wikipedia_research", fake_research)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), chat_server.Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        item = _post(base + "/v1/research/queue",
                     {"topic": "Ankara", "pages": 2})
        assert item["status"] == "pending"
        listed = json.loads(urllib.request.urlopen(
            base + "/v1/research/queue", timeout=10).read().decode("utf-8"))
        assert listed["data"][0]["id"] == item["id"]
        unapproved = _post(base + "/v1/research/queue/run", {"limit": 1})
        assert unapproved["ran"] == []
        assert calls == []
        approved = _post(base + "/v1/research/queue/approve",
                         {"id": item["id"]})
        assert approved["status"] == "approved"
        ran = _post(base + "/v1/research/queue/run", {"limit": 1})
        assert ran["ran"][0]["status"] == "done"
        assert calls == ["Ankara"]
        assert ran["ran"][0]["sources"][0]["source"].startswith("https://")
        assert "Ankara" in _post(
            base + "/v1/documents/search",
            {"query": "capital Turkey"})["data"][0]["chunk"]
        chat = _post(base + "/v1/chat/completions", {
            "messages": [{"role": "user",
                          "content": "what is the capital of Turkey?"}],
            "max_tokens": 20,
        })
        assert "Ankara" in chat["choices"][0]["message"]["content"]
        assert chat["sources"][0]["source"].startswith("https://")
        health = json.loads(urllib.request.urlopen(
            base + "/health", timeout=10).read().decode("utf-8"))
        assert health["research_queue"]["done"] == 1
        second = _post(base + "/v1/research/queue", {"topic": "Izmir"})
        rejected = _post(base + "/v1/research/queue/reject",
                         {"id": second["id"]})
        assert rejected["status"] == "rejected"
        ran_again = _post(base + "/v1/research/queue/run", {"limit": 1})
        assert ran_again["ran"] == []
        assert calls == ["Ankara"]
        with pytest.raises(urllib.error.HTTPError) as missing:
            _post(base + "/v1/research/queue/approve", {"id": "r9999"})
        assert missing.value.code == 404
    finally:
        httpd.shutdown()
        httpd.server_close()
        chat_server._state.clear()
        chat_server._state.update(old)
        docs.close()
