import pytest

from streamft.doc_memory import DocMemory


def test_doc_memory_indexes_and_searches_with_source(tmp_path):
    try:
        mem = DocMemory(tmp_path / "docs.db")
    except Exception as exc:
        pytest.skip(f"sqlite fts5 unavailable: {exc}")
    added = mem.add_text(
        "law.txt", "The capital of France is Paris. Rent contracts need a date.",
        title="Notes")
    assert added >= 1
    hits = mem.search("capital France", k=3)
    assert hits
    assert "Paris" in hits[0]["chunk"]
    assert hits[0]["source"] == "law.txt"
    assert mem.stats()["sources"] == 1
    # Re-ingesting the same source replaces its chunks.
    mem.add_text("law.txt", "Only one fact.", title="Notes")
    assert len(mem.search("fact", k=3)) == 1
    mem.close()
