import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_sft_chat_is_blocked_but_shows_story_structure():
    data = json.loads(
        (ROOT / "research/results/spiking_chat_en_clean_21m_sft.json")
        .read_text(encoding="utf-8"))
    assert data["step"] == 12000
    chat = data["bpb"]["chat"]
    assert chat["bpb"] < 2.5
    assert chat["assistant"]["loss"] > 0
    assert data["verdict"] == "BLOCKED"
    assert not data["gate"]["alpha_at_least_0_5"]
    assert not data["gate"]["no_tag_leak"]
    assert data["gate"]["samples_nonempty"]
    assert sum(s["tag_leak"] for s in data["samples"]) >= 1
    assert min(s["alpha"] for s in data["samples"]) < 0.5
    # The qualitative gain: no empty sample and at least one longer response.
    assert all(s["chars"] > 0 for s in data["samples"])
    assert max(s["chars"] for s in data["samples"]) > 100
