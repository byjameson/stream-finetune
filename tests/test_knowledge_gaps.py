import pytest

from atomzero.knowledge_gaps import (
    GapNotFound, KnowledgeGapError, KnowledgeGapStore)


def test_gap_store_dedupes_and_tracks_status(tmp_path):
    store = KnowledgeGapStore(tmp_path / "gaps.json")
    first = store.add("What is the launch code?", session_id="s1")
    second = store.add("  what is the launch code?  ", session_id="s2")
    assert first["id"] == second["id"]
    assert second["count"] == 2
    assert store.stats() == {"total": 1, "pending": 1}
    proposed = store.mark_proposed(first["id"], "r0001")
    assert proposed["status"] == "proposed"
    assert proposed["research_id"] == "r0001"
    assert store.stats() == {"total": 1, "proposed": 1}
    with pytest.raises(KnowledgeGapError):
        store.mark_proposed(first["id"], "r0002")
    dismissed = store.dismiss(first["id"])
    assert dismissed["status"] == "dismissed"
    assert store.stats() == {"total": 1, "dismissed": 1}


def test_gap_store_validates_input_and_missing_ids(tmp_path):
    store = KnowledgeGapStore(tmp_path / "gaps.json")
    with pytest.raises(KnowledgeGapError):
        store.add("")
    with pytest.raises(KnowledgeGapError):
        store.add("x" * 501)
    with pytest.raises(GapNotFound):
        store.get("g9999")
    with pytest.raises(GapNotFound):
        store.dismiss("g9999")
