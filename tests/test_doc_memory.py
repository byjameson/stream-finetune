import json

import pytest

from atomzero.doc_memory import (
    DocMemory, SourceNotFound, _chunks, ann_topk, best_snippet, embed_text,
    main)


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


def test_best_snippet_selects_query_sentence_and_keeps_teacher_value():
    text = (
        "Berlin was first documented in the 13th century. "
        "The city has a population of 3.7 million. "
        "It is the capital of Germany.")
    snippet = best_snippet("what is the population of Berlin?", text)
    assert "population of 3.7 million" in snippet
    teacher = ("Topic: Teacher code\n\nThe teacher code is T-9000.")
    assert "T-9000" in best_snippet("what is the teacher code?", teacher)


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


def test_hybrid_rejects_unrelated_query(tmp_path):
    mem = DocMemory(tmp_path / "docs.db")
    mem.add_text("law.txt", "The capital of France is Paris.",
                 title="Notes")
    assert mem.search("nothing here", k=2) == []
    mem.close()


def test_hashed_hybrid_stays_silent_without_an_fts_match(tmp_path):
    mem = DocMemory(tmp_path / "docs.db")
    mem.add_text("launch.txt", "The launch code is ORANGE-42.",
                 title="Launch")
    mem.add_text("support.txt", "The support email is help@example.test.",
                 title="Support")
    assert mem.search("what is the population of Paris?", k=2) == []
    mem.close()


def test_hybrid_vector_cache_is_pruned_on_replace(tmp_path):
    mem = DocMemory(tmp_path / "docs.db")
    mem.add_text("warranty.txt", "The warranty code is W-7781.")
    mem.search("warranty code", k=2)
    assert mem.con.execute(
        "SELECT COUNT(*) FROM doc_vectors").fetchone()[0] == 1
    mem.add_text("warranty.txt", "The warranty code is W-9900.")
    assert mem.search("7781", k=2) == []
    assert mem.con.execute(
        "SELECT COUNT(*) FROM doc_vectors").fetchone()[0] == 1
    assert mem.search("warranty code", k=2)[0]["source"] == "warranty.txt"
    mem.close()


def test_prewarmed_index_matches_linear_scan(tmp_path):
    mem = DocMemory(tmp_path / "docs.db")
    mem.add_text("printer.txt",
                 "Replace the cyan cartridge when print quality fades.",
                 title="Printer")
    mem.add_text("law.txt", "The capital of France is Paris.",
                 title="Notes")
    mem.add_text("weather.txt", "Tomorrow will be sunny with light wind.",
                 title="Weather")
    stats = mem.prewarm_vectors()
    assert stats["built"] is True
    assert stats["chunks"] == 3
    assert stats["dim"] == 512
    assert stats["backend"] == "numpy"
    indexed, indexed_scores = mem._embedding_candidates(
        "Why are my documents printing pale?", None, 3)
    scanned, scan_scores = mem._embedding_candidates_scan(
        "Why are my documents printing pale?", None, 3)
    assert [hit["source"] for hit in indexed] == [
        hit["source"] for hit in scanned]
    for hit in indexed:
        assert indexed_scores[hit["_rowid"]] == pytest.approx(
            scan_scores[hit["_rowid"]], abs=1e-6)
    mem.close()


def test_ann_index_matches_exact_when_probing_every_bucket(tmp_path):
    mem = DocMemory(tmp_path / "docs.db")
    for index in range(20):
        mem.add_text(f"policy-{index}.txt",
                     f"Policy {index}: approval code CODE-{index}.",
                     title=f"Policy {index}", replace=False)
    mem.vector_index_mode = "ann"
    mem.ann_min_chunks = 4
    stats = mem.prewarm_vectors()
    assert stats["mode"] == "ann"
    assert stats["ann"] is True
    assert stats["exact_fallback"] is True
    assert stats["nlist"] >= 2
    index = mem._vector_index
    query = embed_text("Policy 3 approval code")
    exact_positions, _ = mem._exact_candidates(index, query, 5)
    ann_positions, _ = ann_topk(
        index["matrix"], index["ann"], query, 5,
        probe=index["ann"]["nlist"])
    assert [int(value) for value in ann_positions] == [
        int(value) for value in exact_positions]
    mem.close()


def test_ann_index_falls_back_to_exact_on_low_confidence(tmp_path):
    mem = DocMemory(tmp_path / "docs.db")
    for index in range(20):
        mem.add_text(f"policy-{index}.txt",
                     f"Policy {index}: approval code CODE-{index}.",
                     title=f"Policy {index}", replace=False)
    mem.vector_index_mode = "ann"
    mem.ann_min_chunks = 4
    mem.ann_probe = 1
    mem.prewarm_vectors()
    mem._embedding_candidates("zzzz qqqq", None, 3)
    stats = mem.vector_index_stats()
    assert stats["mode"] == "ann"
    assert stats["fallbacks"] >= 1
    mem.close()


def test_vector_index_invalidates_on_add_and_forget(tmp_path):
    mem = DocMemory(tmp_path / "docs.db")
    mem.add_text("launch.txt", "The launch code is ORANGE-42.")
    mem.prewarm_vectors()
    assert mem.vector_index_stats()["built"] is True
    mem.add_text("support.txt", "The support email is help@example.test.")
    assert mem.vector_index_stats()["built"] is False
    mem.search("support email", k=2)
    assert mem.vector_index_stats()["built"] is True
    mem.forget("support.txt")
    assert mem.vector_index_stats()["built"] is False
    mem.close()


def test_teacher_metadata_is_returned(tmp_path):
    try:
        mem = DocMemory(tmp_path / "docs.db")
    except Exception as exc:
        pytest.skip(f"sqlite fts5 unavailable: {exc}")
    mem.add_text(
        "teacher://chatgpt-web/abc", "Topic: Ankara\n\nAnkara is in Turkey.",
        title="Teacher note: Ankara",
        meta={"kind": "teacher", "source_url": "https://example.com/chat",
              "teacher": "chatgpt-web", "model": "gpt-5", "verified": False})
    hit = mem.search("Ankara")[0]
    assert hit["kind"] == "teacher"
    assert hit["source_url"] == "https://example.com/chat"
    assert hit["teacher"] == "chatgpt-web"
    assert hit["model"] == "gpt-5"
    assert hit["verified"] is False
    source = mem.sources()[0]
    assert source["kind"] == "teacher"
    assert source["verified"] is False
    mem.close()


def test_verify_teacher_note_records_reviewer_and_timestamp(tmp_path):
    try:
        mem = DocMemory(tmp_path / "docs.db")
    except Exception as exc:
        pytest.skip(f"sqlite fts5 unavailable: {exc}")
    source = "teacher://chatgpt-web/verify"
    mem.add_text(
        source, "Topic: Ankara\n\nAnkara is in Turkey.",
        title="Teacher note: Ankara",
        meta={"kind": "teacher", "source_url": "https://example.com/chat",
              "teacher": "chatgpt-web", "model": "gpt-5", "verified": False})
    with pytest.raises(SourceNotFound):
        mem.verify_source("teacher://missing", "reviewer")
    with pytest.raises(ValueError):
        mem.verify_source(source, "")
    mem.add_text("notes.txt", "A normal document.", title="Notes")
    with pytest.raises(ValueError):
        mem.verify_source("notes.txt", "reviewer")
    note = mem.verify_source(source, "reviewer-1")
    assert note["verified"] is True
    assert note["reviewer"] == "reviewer-1"
    assert note["verified_at"]
    hit = mem.search("Ankara")[0]
    assert hit["verified"] is True
    assert hit["reviewer"] == "reviewer-1"
    listed = next(item for item in mem.sources()
                  if item["source"] == source)
    assert listed["verified"] is True
    assert listed["verified_at"] == note["verified_at"]
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
