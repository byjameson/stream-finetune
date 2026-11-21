import pytest

from streamft import chat_server
from streamft.doc_memory import DocMemory
from streamft.research_queue import (
    ItemNotFound, ResearchQueue, ResearchQueueError)


def _docs(tmp_path):
    try:
        return DocMemory(tmp_path / "docs.db")
    except Exception as exc:
        pytest.skip(f"sqlite fts5 unavailable: {exc}")


def test_queue_requires_approval_then_cites_sources(tmp_path):
    docs = _docs(tmp_path)
    calls = []

    def fetcher(memory, topic, lang, pages):
        calls.append((topic, lang, pages))
        source = "https://en.wikipedia.org/?curid=1"
        chunks = memory.add_text(
            source, "Ankara is the capital of Turkey.", title="Ankara")
        return [{"source": source, "title": "Ankara", "chunks": chunks,
                 "preview": "Ankara is the capital of Turkey."}]

    queue = ResearchQueue(tmp_path / "queue.json")
    item = queue.add("Ankara", lang="en", pages=2)
    assert item["status"] == "pending"
    assert queue.run_approved(docs, fetcher=fetcher) == []
    assert calls == []
    queue.approve(item["id"])
    ran = queue.run_approved(docs, fetcher=fetcher)
    assert len(ran) == 1
    assert ran[0]["status"] == "done"
    assert ran[0]["sources"][0]["source"].startswith("https://")
    assert calls == [("Ankara", "en", 2)]
    hits = docs.search("capital Turkey")
    assert hits and hits[0]["source"].startswith("https://")
    answer = chat_server.doc_answer("capital of Turkey", docs)
    assert answer is not None
    assert "Ankara" in answer["text"]
    assert answer["sources"][0]["source"].startswith("https://")
    docs.close()


def test_queue_persists_dedupes_and_can_be_rejected(tmp_path):
    path = tmp_path / "queue.json"
    first = ResearchQueue(path)
    item = first.add("Ankara")
    second = ResearchQueue(path)
    assert second.list()[0]["id"] == item["id"]
    with pytest.raises(ResearchQueueError):
        second.add("  ankara  ")
    second.approve(item["id"])
    rejected = second.reject(item["id"])
    assert rejected["status"] == "rejected"
    again = second.add("Ankara")
    assert again["id"] == "r0002"
    assert second.stats()["total"] == 2
    assert second.stats()["rejected"] == 1
    assert second.stats()["pending"] == 1


def test_queue_fetch_error_is_recorded(tmp_path):
    docs = _docs(tmp_path)

    def failing(memory, topic, lang, pages):
        raise RuntimeError("network down")

    queue = ResearchQueue(tmp_path / "queue.json")
    item = queue.add("Ankara")
    queue.approve(item["id"])
    ran = queue.run_approved(docs, fetcher=failing)
    assert ran[0]["status"] == "error"
    assert "network down" in ran[0]["error"]
    assert ran[0]["sources"] == []
    docs.close()


def test_queue_validates_topics_and_ids(tmp_path):
    queue = ResearchQueue(tmp_path / "queue.json")
    with pytest.raises(ResearchQueueError):
        queue.add("")
    with pytest.raises(ResearchQueueError):
        queue.add("x" * 201)
    with pytest.raises(ResearchQueueError):
        queue.add("bad\ntopic")
    with pytest.raises(ResearchQueueError):
        queue.add("ok", lang="english!")
    with pytest.raises(ResearchQueueError):
        queue.add("ok", pages=0)
    with pytest.raises(ItemNotFound):
        queue.approve("r9999")
    with pytest.raises(ResearchQueueError):
        queue.run_approved(None)
