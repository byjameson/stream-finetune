import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "research/results/spiking_semantic.json"


def test_narrative_coherence_by_model_size():
    payload = json.loads(DATA.read_text(encoding="utf-8"))
    rows = {Path(r["model"]).name: r for r in payload["results"]}
    small = rows["spiking_4m_10k"]
    mid = rows["spiking_21m"]
    large = rows["spiking_50m_run"]
    assert small["accuracy"] < 0.35
    assert mid["accuracy"] > 0.35
    assert abs(large["accuracy"] - mid["accuracy"]) < 0.05
    for r in (small, mid, large):
        assert r["chance"] == 0.25
        assert r["n"] == 100
