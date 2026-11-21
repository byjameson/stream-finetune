import pytest

from research.chat_human import ChatHuman
from streamft.agent_runtime import AgentSession
from streamft.doc_memory import DocMemory, SourceNotFound


def _docs(tmp_path):
    try:
        return DocMemory(tmp_path / "docs.db")
    except Exception as exc:
        pytest.skip(f"sqlite fts5 unavailable: {exc}")


def test_doc_memory_forget_removes_only_that_source(tmp_path):
    docs = _docs(tmp_path)
    docs.add_text("pii.txt", "The patient email is alice@example.test.")
    docs.add_text("keep.txt", "The project code is BLUE-7.")
    assert len(docs.sources()) == 2
    result = docs.forget("pii.txt")
    assert result["chunks_deleted"] == 1
    assert result["meta_deleted"] == 1
    assert [source["source"] for source in docs.sources()] == ["keep.txt"]
    assert docs.search("alice@example.test") == []
    assert docs.search("BLUE-7")
    with pytest.raises(SourceNotFound):
        docs.forget("pii.txt")
    docs.close()


def test_doc_memory_forget_all_and_meta(tmp_path):
    docs = _docs(tmp_path)
    docs.add_text(
        "teacher://test/note", "Topic: PII\n\nThe code is X.",
        meta={"kind": "teacher", "source_url": "https://example.test",
              "teacher": "manual", "model": "manual", "verified": False})
    result = docs.forget_all()
    assert result["chunks_deleted"] == 1
    assert result["meta_deleted"] == 1
    assert docs.sources() == []
    docs.close()


def test_chat_human_forget_matches_and_clears(tmp_path):
    human = ChatHuman(tmp_path / "memory.json")
    human.respond("my name is Alice")
    human.respond("remember that I like green tea")
    human.respond("add task call Alice")
    human.respond("add task buy milk")
    removed = human.forget("alice")
    assert removed["facts_deleted"] == 1
    assert removed["episodes_deleted"] >= 1
    assert removed["tasks_deleted"] == 1
    assert not any("alice" in f"{f['key']} {f['value']}".lower()
                   for f in human.data["facts"])
    assert any("milk" in task["text"] for task in human.data["tasks"])
    cleared = human.forget()
    assert cleared["cleared_all"] is True
    assert human.data["facts"] == []
    assert human.data["episodes"] == []
    assert human.data["tasks"] == []


def test_agent_session_forget_delegates(tmp_path):
    docs = _docs(tmp_path)
    agent = AgentSession("pii", tmp_path / "memory.json", docs=docs)
    agent.chat("my name is Alice")
    result = agent.forget("Alice")
    assert result["facts_deleted"] == 1
    assert agent.chat("what is my name?")["reply"].startswith(
        "I do not know your name")
    docs.close()
