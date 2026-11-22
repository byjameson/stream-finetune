import pytest
import torch

from atomzero.doc_memory import DocMemory
from atomzero.neural_embedder import NeuralEmbedderError, NeuralReranker


def fake_encoder(texts):
    rows = []
    for text in texts:
        low = text.lower()
        if "password" in low:
            rows.append([1.0, 0.0])
        elif "weather" in low:
            rows.append([0.0, 1.0])
        else:
            rows.append([0.0, 0.0])
    return torch.tensor(rows, dtype=torch.float32)


def test_neural_reranker_orders_by_embedding_similarity():
    reranker = NeuralReranker(encoder=fake_encoder)
    hits = [
        {"chunk": "Weather forecast for tomorrow.", "source": "weather",
         "title": "Weather", "score": -2.0},
        {"chunk": "Reset your password here.", "source": "security",
         "title": "Security", "score": -1.0},
    ]
    ranked = reranker.rerank("how do I reset my password?", hits, k=2)
    assert ranked[0]["source"] == "security"
    assert ranked[0]["neural_score"] > ranked[1]["neural_score"]


def test_neural_reranker_normalizes_vectors():
    reranker = NeuralReranker(encoder=fake_encoder)
    vectors = reranker.embed(["password", "weather"])
    assert torch.allclose(vectors.norm(dim=1), torch.ones(2))


def test_missing_neural_model_is_a_clear_error(tmp_path):
    reranker = NeuralReranker(tmp_path / "missing-model")
    with pytest.raises(NeuralEmbedderError):
        reranker.embed(["hello"])


def test_hybrid_neural_candidates_recover_an_fts_miss(tmp_path):
    def synonym_encoder(texts):
        rows = []
        for text in texts:
            low = text.lower()
            if "login" in low or "password" in low:
                rows.append([1.0, 0.0])
            elif "weather" in low:
                rows.append([0.0, 1.0])
            else:
                rows.append([0.0, 0.0])
        return torch.tensor(rows, dtype=torch.float32)

    docs = DocMemory(tmp_path / "docs.db")
    docs.add_text("security.txt", "Reset your password from the account page.",
                  title="Security")
    docs.add_text("weather.txt", "Tomorrow will be sunny.", title="Weather")
    docs.reranker = NeuralReranker(encoder=synonym_encoder)
    hits = docs.search("How do I regain login access?", k=2)
    assert hits[0]["source"] == "security.txt"
    assert hits[0]["neural_score"] == 1.0
    docs.close()


def test_doc_memory_uses_attached_reranker(tmp_path):
    class Reverse:
        def rerank(self, query, hits, k):
            return list(reversed(hits))[:k]

    docs = DocMemory(tmp_path / "docs.db")
    docs.add_text("a.txt", "alpha beta", title="A")
    docs.add_text("b.txt", "alpha gamma", title="B")
    raw = docs.search("alpha", k=2, rerank=False)
    docs.reranker = Reverse()
    hits = docs.search("alpha", k=2)
    assert [hit["source"] for hit in hits] == [
        hit["source"] for hit in reversed(raw)]
    docs.close()
