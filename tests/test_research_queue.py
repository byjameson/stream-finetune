import json
import time

import pytest

from atomzero import chat_server
from atomzero.doc_memory import DocMemory
from atomzero.research_queue import (
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


def test_queue_teacher_synthesis_is_unverified(tmp_path):
    docs = _docs(tmp_path)
    prompts = []

    def fetcher(memory, topic, lang, pages):
        source = "https://en.wikipedia.org/?curid=1"
        chunks = memory.add_text(
            source, "Ankara is the capital of Turkey.", title="Ankara")
        return [{"source": source, "title": "Ankara", "chunks": chunks,
                 "preview": "Ankara is the capital of Turkey.",
                 "context": "Ankara is the capital of Turkey. It is in Anatolia."}]

    class FakeTeacher:
        label = "fake"
        model = "fake-1"

        def ask(self, prompt, system="", max_tokens=512):
            prompts.append(prompt)
            return "Teacher synthesis: Ankara is the capital of Turkey."

    queue = ResearchQueue(tmp_path / "queue.json")
    item = queue.add("Ankara", pages=1)
    queue.approve(item["id"])
    ran = queue.run_approved(docs, fetcher=fetcher, teacher=FakeTeacher())
    assert ran[0]["status"] == "done"
    assert ran[0]["teacher"]["verified"] is False
    assert ran[0]["teacher"]["source"].startswith("teacher://fake/")
    assert prompts and "https://en.wikipedia.org/?curid=1" in prompts[0]
    assert "Ankara is the capital of Turkey" in prompts[0]
    teacher_hit = docs.search("teacher synthesis")[0]
    assert teacher_hit["kind"] == "teacher"
    assert teacher_hit["verified"] is False
    answer = chat_server.doc_answer("capital Turkey", docs)
    assert answer is not None
    assert any(source["kind"] == "teacher"
               for source in answer["sources"])
    docs.close()


def test_queue_teacher_error_keeps_wikipedia_pages(tmp_path):
    docs = _docs(tmp_path)

    def fetcher(memory, topic, lang, pages):
        source = "https://en.wikipedia.org/?curid=2"
        chunks = memory.add_text(
            source, "Ankara is in Turkey.", title="Ankara")
        return [{"source": source, "title": "Ankara", "chunks": chunks,
                 "preview": "Ankara is in Turkey.",
                 "context": "Ankara is in Turkey."}]

    class FailingTeacher:
        label = "fake"
        model = "fake-1"

        def ask(self, prompt, system="", max_tokens=512):
            raise RuntimeError("provider down")

    queue = ResearchQueue(tmp_path / "queue.json")
    item = queue.add("Ankara", pages=1)
    queue.approve(item["id"])
    ran = queue.run_approved(
        docs, fetcher=fetcher, teacher=FailingTeacher())
    assert ran[0]["status"] == "done"
    assert "provider down" in ran[0]["teacher"]["error"]
    assert ran[0]["sources"][0]["source"].startswith("https://")
    assert docs.search("Turkey")
    docs.close()


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


def test_queue_recovers_stale_running_items(tmp_path):
    path = tmp_path / "queue.json"
    queue = ResearchQueue(path)
    item = queue.add("Ankara")
    queue.approve(item["id"])
    data = json.loads(path.read_text(encoding="utf-8"))
    data[0]["status"] = "running"
    data[0]["started_at"] = time.strftime(
        "%Y-%m-%dT%H:%M:%S", time.localtime(time.time() - 7200))
    path.write_text(json.dumps(data), encoding="utf-8")
    assert queue.recover_stale_running(60) == [item["id"]]
    recovered = queue.list()[0]
    assert recovered["status"] == "approved"
    assert "recovered" in recovered["error"]

    data[0] = recovered
    data[0]["status"] = "running"
    data[0]["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    path.write_text(json.dumps(data), encoding="utf-8")
    assert queue.recover_stale_running(3600) == []


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
