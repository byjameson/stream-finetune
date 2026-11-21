import pytest

from streamft.doc_memory import DocMemory, rerank_hits


def test_rerank_prefers_query_coverage_over_raw_bm25():
    hits = [
        {"chunk": "France is a country in Europe.", "source": "a.txt",
         "title": "France", "score": -10.0},
        {"chunk": "The capital of France is Paris.", "source": "b.txt",
         "title": "France capital", "score": -1.0},
    ]
    ranked = rerank_hits("what is the capital of France?", hits, k=2)
    assert ranked[0]["source"] == "b.txt"
    assert "rerank_score" in ranked[0]


def test_rerank_keeps_original_order_when_no_query_terms():
    hits = [
        {"chunk": "a", "source": "a", "title": "", "score": -2.0},
        {"chunk": "b", "source": "b", "title": "", "score": -1.0},
    ]
    assert rerank_hits("", hits, k=2) == hits


def test_search_reranker_returns_best_document(tmp_path):
    try:
        docs = DocMemory(tmp_path / "docs.db")
    except Exception as exc:
        pytest.skip(f"sqlite fts5 unavailable: {exc}")
    docs.add_text("country.txt", "France is a country in Europe.",
                  title="France")
    docs.add_text("capital.txt", "The capital of France is Paris.",
                  title="France capital")
    hits = docs.search("what is the capital of France?", k=1)
    assert hits[0]["source"] == "capital.txt"
    assert "Paris" in hits[0]["chunk"]
    docs.close()
