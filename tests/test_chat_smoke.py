import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_chat_smoke_pipeline_is_wired():
    data = json.loads(
        (ROOT / "research/results/spiking_chat_tr_smoke.json")
        .read_text(encoding="utf-8"))
    assert data["step"] == 500
    assert data["config"]["vocab"] == 8192
    assert data["config"]["dim"] == 1024
    # Sanity only: a 500-step from-scratch model is expected to be poor.
    for shard in ("chat", "wiki", "stories"):
        assert 0 < data["bpb"][shard]["bpb"] < 4.0
    assert data["samples"]
    assert any(s["turkish_chars"] > 0 for s in data["samples"])
