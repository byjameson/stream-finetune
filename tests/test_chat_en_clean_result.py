import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_clean_21m_chat_is_still_blocked():
    data = json.loads(
        (ROOT / "research/results/spiking_chat_en_clean_21m.json")
        .read_text(encoding="utf-8"))
    assert data["step"] == 12000
    assert data["lang"] == "en"
    assert data["bpb"]["chat"]["bpb"] < 2.5
    assert data["gate"]["alpha_at_least_0_5"]
    assert not data["gate"]["no_tag_leak"]
    assert data["verdict"] == "BLOCKED"
    assert min(s["alpha"] for s in data["samples"]) >= 0.5
    assert any(s["tag_leak"] for s in data["samples"])
