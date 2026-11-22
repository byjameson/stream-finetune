from atomzero.doc_memory import DocMemory
from atomzero.knowledge_gaps import KnowledgeGapStore
from atomzero.research_queue import ResearchQueue
from atomzero.curiosity import propose_from_gaps


def _docs(tmp_path):
    return DocMemory(tmp_path / "docs.db")


def test_curiosity_proposes_pending_gaps_only(tmp_path):
    gaps = KnowledgeGapStore(tmp_path / "gaps.json")
    queue = ResearchQueue(tmp_path / "queue.json")
    gaps.add("What is the capital of New Zealand?")
    gaps.add("What is the capital of New Zealand?")
    result = propose_from_gaps(gaps, queue, limit=3)
    assert len(result["proposed"]) == 1
    item = result["proposed"][0]["research"]
    assert item["status"] == "pending"
    assert item["kind"] == "curiosity"
    assert item["origin_id"].startswith("g")
    assert result["proposed"][0]["gap"]["status"] == "proposed"
    assert gaps.list()[0]["count"] == 2


def test_curiosity_requires_approval_before_teacher(tmp_path):
    docs = _docs(tmp_path)
    gaps = KnowledgeGapStore(tmp_path / "gaps.json")
    queue = ResearchQueue(tmp_path / "queue.json")
    gaps.add("What is the capital of New Zealand?")
    calls = []

    class Teacher:
        label = "fake"
        model = "fake-1"

        def ask(self, prompt, system="", max_tokens=512):
            calls.append(prompt)
            return "The capital of New Zealand is Wellington."

    def fetcher(memory, topic, lang, pages):
        source = "https://en.wikipedia.org/?curid=99"
        chunks = memory.add_text(
            source, "Wellington is the capital of New Zealand.",
            title="Wellington")
        return [{"source": source, "title": "Wellington", "chunks": chunks,
                 "preview": "Wellington is the capital of New Zealand.",
                 "context": "Wellington is the capital of New Zealand."}]

    proposed = propose_from_gaps(gaps, queue, limit=1)["proposed"][0]
    assert queue.run_approved(docs, fetcher=fetcher, teacher=Teacher()) == []
    assert calls == []
    queue.approve(proposed["research"]["id"])
    ran = queue.run_approved(docs, fetcher=fetcher, teacher=Teacher())
    assert ran[0]["status"] == "done"
    assert calls
    assert ran[0]["teacher"]["verified"] is False
    assert docs.search("Wellington")
    docs.close()


def test_curiosity_limit_and_deduped_queue(tmp_path):
    gaps = KnowledgeGapStore(tmp_path / "gaps.json")
    queue = ResearchQueue(tmp_path / "queue.json")
    for query in ("Question one?", "Question two?", "Question three?"):
        gaps.add(query)
    first = propose_from_gaps(gaps, queue, limit=2)
    assert len(first["proposed"]) == 2
    second = propose_from_gaps(gaps, queue, limit=2)
    assert len(second["proposed"]) == 1
    assert len(queue.list()) == 3
    assert all(item["kind"] == "curiosity" for item in queue.list())
