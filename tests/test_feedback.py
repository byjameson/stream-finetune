import json

import pytest

from research.learning_loop import main as learning_main
from streamft.feedback import (
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
