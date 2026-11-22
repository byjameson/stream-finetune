import pytest

from research.chat_human import ChatHuman
from atomzero.doc_memory import DocMemory


def test_chat_human_answers_from_local_documents(tmp_path):
    try:
        docs = DocMemory(tmp_path / "docs.db")
    except Exception as exc:
        pytest.skip(f"sqlite fts5 unavailable: {exc}")
    docs.add_text("law.txt", "The capital of France is Paris.",
                  title="Notes")
    human = ChatHuman(tmp_path / "memory.json", docs=docs)
    reply = human.respond("what is the capital of France?")
    assert "Paris" in reply
    assert "law.txt" in reply
    assert "not professional advice" in reply
    docs.close()
