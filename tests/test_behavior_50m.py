import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_v0001_passes_corrected_clean_smoke_and_50m_does_not():
    small = json.loads(
        (ROOT / "research/results/accepted_v0001_banned_fixed.json")
        .read_text(encoding="utf-8"))
    big = json.loads(
        (ROOT / "research/results/transformer_behavior_50m_banned.json")
        .read_text(encoding="utf-8"))
    assert small["verdict"] == "PASS_CHAT_SMOKE"
    assert all(small["gate"].values())
    assert small["bpb"]["chat"]["bpb"] < 1.5
    assert small["bpb"]["chat"]["assistant"]["loss"] < 4.4
    # The short 400-step 50M run has better loss but one empty raw sample.
    # The longer 50M run is the product checkpoint; see
    # test_behavior_50m_long.py.
    assert big["bpb"]["chat"]["bpb"] < small["bpb"]["chat"]["bpb"]
    assert big["verdict"] == "BLOCKED"
    assert not big["gate"]["samples_nonempty"]
    assert any(not s["chars"] for s in big["samples"])
