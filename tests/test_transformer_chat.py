import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_transformer_chat_passes_the_smoke_gate_with_constrained_decoding():
    data = json.loads(
        (ROOT / "research/results/transformer_chat_en_clean_21m_banned.json")
        .read_text(encoding="utf-8"))
    assert data["verdict"] == "PASS_CHAT_SMOKE"
    assert all(data["gate"].values())
    assert data["bpb"]["chat"]["bpb"] < 1.5
    assert data["bpb"]["chat"]["assistant"]["loss"] < 4.2
    assert len(data["samples"]) == 12
    assert min(s["alpha"] for s in data["samples"]) >= 0.5
    assert all(s["chars"] > 0 for s in data["samples"])
    assert all(s["tag_leak"] == 0 for s in data["samples"])
    # Constrained decoding is explicit, not hidden.
    assert "banned" in data["decoding"]
