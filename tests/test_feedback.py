import json
from types import SimpleNamespace

import pytest

from research.learning_loop import main as learning_main, split_rows
from atomzero.feedback import (
    FeedbackError, FeedbackNotFound, FeedbackStore)


def test_feedback_is_reviewed_and_exported(tmp_path):
    store = FeedbackStore(tmp_path / "feedback.json")
    first = store.add("hello", "Hi! How can I help?", rating=1,
                      session_id="s1")
    second = store.add("what is 2+2?", "It is five.",
                       corrected="It is four.", rating=-1,
                       session_id="s1")
    third = store.add("bad", "bad answer", rating=-1)
    assert first["status"] == "pending"
    assert [item["id"] for item in store.list()] == [
        "f0001", "f0002", "f0003"]
    assert store.stats() == {"total": 3, "pending": 3}
    with pytest.raises(FeedbackError):
        store.approve(third["id"])
    store.approve(first["id"])
    store.approve(second["id"])
    assert store.stats()["approved"] == 2
    out = tmp_path / "training.jsonl"
    exported = store.export_training(out)
    assert exported["examples"] == 2
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert rows[0] == {"user": "hello", "assistant": "Hi! How can I help?"}
    assert rows[1] == {"user": "what is 2+2?", "assistant": "It is four."}
    store.reject(third["id"])
    assert store.stats()["rejected"] == 1


def test_feedback_rejects_bad_input_and_missing_ids(tmp_path):
    store = FeedbackStore(tmp_path / "feedback.json")
    with pytest.raises(FeedbackError):
        store.add("", "answer")
    with pytest.raises(FeedbackError):
        store.add("question", "answer", rating=2)
    with pytest.raises(FeedbackError):
        store.add("question", "answer", session_id="x" * 65)
    item = store.add("hello", "hi", rating=1)
    with pytest.raises(FeedbackError):
        store.add("hello", "hi", rating=1)
    with pytest.raises(FeedbackNotFound):
        store.approve("f9999")
    with pytest.raises(FeedbackNotFound):
        store.reject("f9999")


def test_learning_loop_cli_exports_approved(tmp_path, capsys):
    store = FeedbackStore(tmp_path / "feedback.json")
    item = store.add("hello", "Hi!", rating=1)
    assert learning_main(["--store", str(store.path), "approve",
                          item["id"]]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "approved"
    out = tmp_path / "training.jsonl"
    assert learning_main([
        "--store", str(store.path), "--out-queue", str(out), "export",
    ]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["examples"] == 1
    assert json.loads(out.read_text()) == {
        "user": "hello", "assistant": "Hi!"}


def test_learning_loop_records_accepted_adapter(tmp_path, monkeypatch, capsys):
    store = FeedbackStore(tmp_path / "feedback.json")
    for i in range(5):
        item = store.add(f"question {i}", f"answer {i}", rating=1)
        store.approve(item["id"])
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps({
        "decision": {"accepted": True, "target_gain": 0.1},
        "accepted_to": "checkpoints/adapters_feedback/v0007",
    }), encoding="utf-8")

    def fake_run(cmd, cwd=None):
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("research.learning_loop.subprocess.run", fake_run)
    active = tmp_path / "active.json"
    history = tmp_path / "history.jsonl"
    code = learning_main([
        "--store", str(store.path),
        "--out-queue", str(tmp_path / "training.jsonl"),
        "--history", str(history),
        "run", "--base", "checkpoints/base",
        "--out", str(tmp_path / "adapters"),
        "--result", str(result_path),
        "--active-file", str(active),
    ])
    assert code == 0
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert summary["accepted"] is True
    assert summary["version"] == "v0007"
    promoted = json.loads(active.read_text(encoding="utf-8"))
    assert promoted["path"] == "checkpoints/adapters_feedback/v0007"
    assert promoted["version"] == "v0007"
    assert promoted["examples"] == 5
    entries = [json.loads(line) for line in
               history.read_text(encoding="utf-8").splitlines()]
    assert entries[0]["version"] == "v0007"
    assert entries[0]["examples"] == 5
    assert entries[0]["decision"]["accepted"] is True
    assert len(entries[0]["queue_sha256"]) == 64
    assert learning_main(["--history", str(history), "history"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert listed["count"] == 1
    assert listed["entries"][0]["path"] == (
        "checkpoints/adapters_feedback/v0007")


def test_split_rows_keeps_holdout_out_of_training():
    rows = [{"user": str(i)} for i in range(10)]
    train, holdout = split_rows(rows, 0.2)
    assert len(train) == 8
    assert [row["user"] for row in holdout] == ["8", "9"]
    assert split_rows(rows, 0.0) == (rows, [])
    with pytest.raises(FeedbackError):
        split_rows(rows, 1.0)
    with pytest.raises(FeedbackError):
        split_rows([{"user": "one"}], 0.2)


def test_learning_loop_holdout_is_recorded(tmp_path, monkeypatch, capsys):
    store = FeedbackStore(tmp_path / "feedback.json")
    for i in range(8):
        item = store.add(f"question {i}", f"answer {i}", rating=1)
        store.approve(item["id"])
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps({
        "decision": {"accepted": True, "target_gain": 0.1},
        "accepted_to": "checkpoints/adapters_feedback/v0001",
    }), encoding="utf-8")
    monkeypatch.setattr(
        "research.learning_loop.subprocess.run",
        lambda cmd, cwd=None: SimpleNamespace(returncode=0))
    report = {
        "examples": 2,
        "models": [
            {"label": "base", "path": "base",
             "metrics": {"loss": 4.0, "tokens": 10, "ppl": 54.6}},
            {"label": "adapters_feedback/v0001", "path": "adapter",
             "metrics": {"loss": 3.0, "tokens": 10, "ppl": 20.1}},
        ],
    }
    monkeypatch.setattr(
        "research.learning_loop.holdout_compare",
        lambda base, data, adapter: report)
    train = tmp_path / "train.jsonl"
    holdout = tmp_path / "holdout.jsonl"
    active = tmp_path / "active.json"
    history = tmp_path / "history.jsonl"
    holdout_result = tmp_path / "holdout_result.json"
    code = learning_main([
        "--store", str(store.path),
        "--out-queue", str(train), "--history", str(history),
        "run", "--base", "checkpoints/base",
        "--out", str(tmp_path / "adapters"),
        "--result", str(result_path),
        "--active-file", str(active),
        "--holdout", "0.25",
        "--holdout-out", str(holdout),
        "--holdout-result", str(holdout_result),
    ])
    assert code == 0
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert summary["holdout"]["loss_relative"] == pytest.approx(-0.25)
    assert len(train.read_text().splitlines()) == 6
    assert len(holdout.read_text().splitlines()) == 2
    entry = json.loads(history.read_text(encoding="utf-8"))
    assert entry["examples"] == 6
    assert entry["holdout"]["examples"] == 2
    assert entry["holdout"]["result"] == str(holdout_result)
    assert json.loads(holdout_result.read_text(encoding="utf-8"))["examples"] == 2
    promoted = json.loads(active.read_text(encoding="utf-8"))
    assert promoted["holdout"] == entry["holdout"]


def test_learning_loop_activate_rolls_back(tmp_path, capsys):
    history = tmp_path / "history.jsonl"
    versions = []
    for number in (1, 2):
        adapter = tmp_path / "adapters" / f"v{number:04d}"
        adapter.mkdir(parents=True)
        (adapter / "adapter.pt").write_bytes(b"adapter")
        versions.append(adapter)
    history.write_text("".join(json.dumps({
        "version": f"v{number:04d}",
        "path": str(versions[number - 1]),
        "examples": 8,
        "result": f"results/v{number:04d}.json",
    }) + "\n" for number in (1, 2)), encoding="utf-8")
    active = tmp_path / "active.json"
    active.write_text(json.dumps({"path": str(versions[1])}),
                      encoding="utf-8")
    code = learning_main([
        "--history", str(history), "activate", "v0001",
        "--active-file", str(active),
    ])
    assert code == 0
    result = json.loads(capsys.readouterr().out)
    assert result["activated"] is True
    assert result["version"] == "v0001"
    assert result["previous"] == str(versions[1])
    promoted = json.loads(active.read_text(encoding="utf-8"))
    assert promoted["path"] == str(versions[0])
    assert promoted["activated"]
    assert learning_main([
        "--history", str(history), "activate", "v9999",
        "--active-file", str(active),
    ]) == 1
    assert "not found" in capsys.readouterr().out


def test_feedback_exports_only_approved_and_deduplicates(tmp_path):
    store = FeedbackStore(tmp_path / "feedback.json")
    store.add("same", "answer", rating=1)
    store.add("same", "answer", rating=1, corrected="better")
    store.add("drop", "nope", rating=-1)
    for item in store.list():
        if item["user"] != "drop":
            store.approve(item["id"])
    out = tmp_path / "training.jsonl"
    result = store.export_training(out)
    assert result["examples"] == 2
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert rows == [
        {"user": "same", "assistant": "answer"},
        {"user": "same", "assistant": "better"},
    ]
