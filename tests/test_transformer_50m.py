import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_50m_transformer_passes_but_20m_is_the_better_product():
    big = json.loads(
        (ROOT / "research/results/transformer_chat_en_clean_50m_banned.json")
        .read_text(encoding="utf-8"))
    small = json.loads(
        (ROOT / "research/results/transformer_chat_en_clean_21m_banned.json")
        .read_text(encoding="utf-8"))
    assert big["verdict"] == "PASS_CHAT_SMOKE"
    assert all(big["gate"].values())
    assert big["bpb"]["chat"]["bpb"] < 1.4
    assert big["bpb"]["chat"]["assistant"]["loss"] < 4.0
    assert min(s["alpha"] for s in big["samples"]) >= 0.5
    assert all(s["tag_leak"] == 0 for s in big["samples"])
    # The 50M model is only marginally better on bpb and has a lower alpha
    # floor; the 20M checkpoint remains the product choice.
    assert big["bpb"]["chat"]["bpb"] < small["bpb"]["chat"]["bpb"]
    assert min(s["alpha"] for s in big["samples"]) < min(
        s["alpha"] for s in small["samples"])
