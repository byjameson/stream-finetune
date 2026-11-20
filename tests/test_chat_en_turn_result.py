import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_turn_aligned_21m_is_still_blocked():
    data = json.loads(
        (ROOT / "research/results/spiking_chat_en_clean_21m_turn.json")
        .read_text(encoding="utf-8"))
    assert data["step"] == 12000
    assert data["bpb"]["chat"]["bpb"] < 2.5
    assert data["verdict"] == "BLOCKED"
    assert not data["gate"]["samples_nonempty"]
    assert not data["gate"]["alpha_at_least_0_5"]
    assert not data["gate"]["no_tag_leak"]
    assert sum(1 for s in data["samples"] if s["chars"] == 0) >= 1
    assert sum(s["tag_leak"] for s in data["samples"]) >= 5
