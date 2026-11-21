import pytest

from streamft.agent_runtime import (
    AgentSession, calculate, math_query)
from streamft.doc_memory import DocMemory


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


def test_agent_research_tool_uses_sources(tmp_path, monkeypatch):
    docs = DocMemory(tmp_path / "docs.db")
    import streamft.agent_runtime as ar
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
