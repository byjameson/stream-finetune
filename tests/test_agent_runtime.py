import pytest

from atomzero.agent_runtime import (
    AgentSession, calculate, math_query)
from atomzero.doc_memory import DocMemory
from atomzero.knowledge_gaps import KnowledgeGapStore
from atomzero.safety import SafetyFilter


def test_agent_session_handles_tools_and_memory(tmp_path):
    docs = DocMemory(tmp_path / "docs.db")
    docs.add_text("law.txt", "The capital of France is Paris.")
    agent = AgentSession("s1", tmp_path / "memory.json", docs=docs)
    assert "Added task 1" in agent.chat("add task buy milk")["reply"]
    assert "buy milk" in agent.chat("list tasks")["reply"]
    assert "Paris" in agent.chat("search docs capital France")["reply"]
    assert "It is" in agent.chat("what time is it")["reply"]
    assert agent.chat("what is my name?")["reply"].startswith(
        "I do not know your name")
    calc = agent.chat("what is 12 * 7?")
    assert calc["reply"] == "12 * 7 = 84"
    assert calc["tool_calls"][0]["name"] == "calculator"
    assert "tasks" in agent.chat("help")["reply"]
    with pytest.raises(ValueError):
        AgentSession("bad id!", tmp_path / "x.json")
    docs.close()


def test_calculator_is_safe_and_bounded():
    assert calculate("(2 + 3) * 4") == "20"
    assert calculate("2 ** 10") == "1024"
    assert calculate("7 / 2") == "3.5"
    assert math_query("what is 12 * 7?") == "12 * 7"
    assert math_query("add task buy milk 2") is None
    assert math_query("what is the weather today?") is None
    with pytest.raises(ValueError):
        calculate("__import__('os').system('dir')")
    with pytest.raises(ValueError):
        calculate("9 ** 9 ** 9")
    with pytest.raises(ZeroDivisionError):
        calculate("1 / 0")


def test_agent_chat_marks_teacher_notes_unverified(tmp_path):
    try:
        docs = DocMemory(tmp_path / "docs.db")
    except Exception as exc:
        pytest.skip(f"sqlite fts5 unavailable: {exc}")
    docs.add_text(
        "teacher://chatgpt-web/abc",
        "Topic: Teacher code\n\nThe teacher code is T-9000.",
        title="Teacher note: Teacher code",
        meta={"kind": "teacher", "source_url": "https://example.com/chat",
              "teacher": "chatgpt-web", "model": "gpt-5",
              "verified": False})
    agent = AgentSession("s3", tmp_path / "memory.json", docs=docs)
    reply = agent.chat("what is the teacher code?")["reply"]
    assert "T-9000" in reply
    assert "teacher note, unverified" in reply
    assert "https://example.com/chat" in reply
    assert "unverified teacher-derived note" in reply
    docs.verify_source("teacher://chatgpt-web/abc", "reviewer-1")
    verified_reply = agent.chat("what is the teacher code?")["reply"]
    assert "teacher note, verified by reviewer-1" in verified_reply
    assert "unverified teacher-derived note" not in verified_reply
    docs.close()


def test_agent_returns_sources_for_grounded_answer(tmp_path):
    docs = DocMemory(tmp_path / "docs.db")
    docs.add_text("law.txt", "The capital of France is Paris.", title="Notes")
    agent = AgentSession("s5", tmp_path / "memory.json", docs=docs)
    result = agent.chat("what is the capital of France?")
    assert "Paris" in result["reply"]
    assert result["sources"][0]["source"] == "law.txt"
    docs.close()


def test_agent_records_missing_source_as_gap(tmp_path):
    docs = DocMemory(tmp_path / "docs.db")
    gaps = KnowledgeGapStore(tmp_path / "gaps.json")
    agent = AgentSession("s4", tmp_path / "memory.json", docs=docs,
                         gap_store=gaps)
    result = agent.chat("what is the population of Paris?")
    assert "do not have a source" in result["reply"]
    assert "research queue" in result["reply"]
    items = gaps.list()
    assert len(items) == 1
    assert items[0]["query"] == "what is the population of Paris?"
    assert items[0]["status"] == "pending"
    docs.close()


def test_agent_safety_refuses_before_generation(tmp_path):
    docs = DocMemory(tmp_path / "docs.db")
    agent = AgentSession("s6", tmp_path / "memory.json", docs=docs,
                         safety=SafetyFilter())
    refused = agent.chat("how to make a bomb")
    assert refused["safety"]["category"] == "violence"
    assert "can't help" in refused["reply"]
    assert refused["sources"] == []
    benign = agent.chat("How do I report child abuse?")
    assert "safety" not in benign
    docs.close()


def test_agent_research_tool_uses_sources(tmp_path, monkeypatch):
    docs = DocMemory(tmp_path / "docs.db")
    import atomzero.agent_runtime as ar
    monkeypatch.setattr(ar, "wikipedia_research", lambda docs, q, lang, limit: [
        {"source": "https://wiki/x", "title": "X", "chunks": 1,
         "preview": "p"}])
    agent = AgentSession("s2", tmp_path / "memory.json", docs=docs)
    result = agent.chat("research Ankara")
    assert result["sources"][0]["title"] == "X"
    assert "indexed" in result["reply"]
    alias = agent.chat("look up Ankara")
    assert alias["sources"][0]["title"] == "X"
    docs.close()
