import json
import time

from research import research_worker
from atomzero.research_queue import ResearchQueue


def test_worker_lock_blocks_overlap_and_releases(tmp_path):
    lock = tmp_path / "worker.lock"
    token = research_worker.acquire_lock(lock)
    assert token
    assert research_worker.acquire_lock(lock) is None
    research_worker.release_lock(lock, token)
    assert not lock.exists()
    second = research_worker.acquire_lock(lock)
    assert second
    research_worker.release_lock(lock, second)
    assert not lock.exists()


def test_worker_recovers_stale_lock(tmp_path):
    lock = tmp_path / "worker.lock"
    lock.write_text(json.dumps({
        "pid": 99999,
        "created": time.time() - 7200,
        "token": "old",
    }), encoding="utf-8")
    token = research_worker.acquire_lock(lock, stale_seconds=60)
    assert token
    assert json.loads(lock.read_text())["token"] == token
    research_worker.release_lock(lock, token)
    assert not lock.exists()


def test_worker_runs_only_approved_and_writes_log(tmp_path, monkeypatch):
    queue_path = tmp_path / "queue.json"
    queue = ResearchQueue(queue_path)
    queue.add("Pending topic")
    approved = queue.add("Ankara")
    queue.approve(approved["id"])
    calls = []

    def fake_fetcher(memory, topic, lang, pages):
        calls.append(topic)
        source = "https://en.wikipedia.org/?curid=1"
        chunks = memory.add_text(
            source, "Ankara is the capital of Turkey.", title="Ankara")
        return [{"source": source, "title": "Ankara", "chunks": chunks,
                 "preview": "Ankara is the capital of Turkey."}]

    monkeypatch.setattr(
        "atomzero.web_research.wikipedia_research", fake_fetcher)
    log = tmp_path / "worker.jsonl"
    lock = tmp_path / "worker.lock"
    code = research_worker.main([
        "--queue", str(queue_path), "--db", str(tmp_path / "docs.db"),
        "--limit", "2", "--log", str(log), "--lock", str(lock), "--once",
    ])
    assert code == 0
    assert calls == ["Ankara"]
    assert not lock.exists()
    statuses = {item["topic"]: item["status"]
                for item in queue.list()}
    assert statuses == {"Pending topic": "pending", "Ankara": "done"}
    records = [json.loads(line) for line in
               log.read_text(encoding="utf-8").splitlines()]
    assert records[0]["event"] == "research_run"
    assert records[0]["ran"][0]["topic"] == "Ankara"
    assert records[0]["ran"][0]["sources"] == 1
    assert records[0]["stats"]["done"] == 1
    assert records[0]["stats"]["pending"] == 1


def test_worker_logs_and_skips_when_locked(tmp_path):
    lock = tmp_path / "worker.lock"
    token = research_worker.acquire_lock(lock)
    log = tmp_path / "worker.jsonl"
    try:
        code = research_worker.main([
            "--queue", str(tmp_path / "queue.json"),
            "--db", str(tmp_path / "docs.db"),
            "--log", str(log), "--lock", str(lock), "--once",
        ])
    finally:
        research_worker.release_lock(lock, token)
    assert code == 0
    record = json.loads(log.read_text(encoding="utf-8").splitlines()[0])
    assert record["event"] == "research_locked"
