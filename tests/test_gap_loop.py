import json

from research.gap_loop import main as gap_main
from atomzero.knowledge_gaps import KnowledgeGapStore
from atomzero.research_queue import ResearchQueue


def test_gap_loop_proposes_pending_research(tmp_path, capsys):
    store = KnowledgeGapStore(tmp_path / "gaps.json")
    gap = store.add("What is the launch code?")
    queue_path = tmp_path / "research.json"
    code = gap_main([
        "--store", str(store.path), "--queue", str(queue_path),
        "propose", gap["id"], "--pages", "1",
    ])
    assert code == 0
    result = json.loads(capsys.readouterr().out)
    assert result["research"]["status"] == "pending"
    assert result["gap"]["status"] == "proposed"
    queue = ResearchQueue(queue_path)
    assert queue.list()[0]["topic"] == "What is the launch code?"
    assert queue.list()[0]["status"] == "pending"
    assert store.get(gap["id"])["research_id"] == result["research"]["id"]


def test_gap_loop_dismisses(tmp_path, capsys):
    store = KnowledgeGapStore(tmp_path / "gaps.json")
    gap = store.add("What is the population of Paris?")
    code = gap_main(["--store", str(store.path), "dismiss", gap["id"]])
    assert code == 0
    assert json.loads(capsys.readouterr().out)["status"] == "dismissed"
