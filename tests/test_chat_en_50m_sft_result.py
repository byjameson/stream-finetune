import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_50m_sft_does_not_beat_the_21m_sft_shape():
    data = json.loads(
        (ROOT / "research/results/spiking_chat_en_clean_50m_sft.json")
        .read_text(encoding="utf-8"))
    assert data["step"] == 12000
    chat = data["bpb"]["chat"]
    assert chat["bpb"] < 2.5
    assert chat["assistant"]["loss"] > 0
    assert data["verdict"] == "BLOCKED"
    assert not data["gate"]["samples_nonempty"]
    assert not data["gate"]["alpha_at_least_0_5"]
    assert not data["gate"]["no_tag_leak"]
    assert min(s["alpha"] for s in data["samples"]) < 0.5
    assert sum(1 for s in data["samples"] if s["chars"] == 0) >= 1
    assert sum(s["tag_leak"] for s in data["samples"]) >= 2
