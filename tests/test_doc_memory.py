import json

import pytest

from streamft.doc_memory import DocMemory, _chunks, main


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


def test_chunks_split_long_text_on_word_boundaries():
    chunks = _chunks(("alpha " * 300).strip())
    assert len(chunks) > 1
    assert all(len(chunk) <= 900 for chunk in chunks)
    assert all(chunk.startswith("alpha") for chunk in chunks)


def test_search_ignores_common_question_words(tmp_path):
    try:
        mem = DocMemory(tmp_path / "docs.db")
    except Exception as exc:
        pytest.skip(f"sqlite fts5 unavailable: {exc}")
    mem.add_text("noise.txt", "is " * 300)
    mem.add_text("ankara.txt", "Ankara is the capital of Turkey.",
                 title="Ankara")
    hits = mem.search("What is Ankara?", k=2)
    assert hits
    assert "capital" in hits[0]["chunk"]
    mem.close()


def test_cli_accepts_db_after_subcommand(tmp_path, capsys):
    note = tmp_path / "note.txt"
    note.write_text("The project codename is Blue Lantern.", encoding="utf-8")
    db = tmp_path / "docs.db"
    main(["add", str(note), "--db", str(db), "--title", "Demo"])
    added = json.loads(capsys.readouterr().out)
    assert added["added"] == 1
    main(["--db", str(db), "search", "codename"])
    hits = json.loads(capsys.readouterr().out)
    assert "Blue Lantern" in hits[0]["chunk"]
