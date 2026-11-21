import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_50m_long_behavior_is_the_new_product():
    small = json.loads(
        (ROOT / "research/results/accepted_v0001_banned_fixed.json")
        .read_text(encoding="utf-8"))
    big = json.loads(
        (ROOT / "research/results/transformer_behavior_50m_long_banned.json")
        .read_text(encoding="utf-8"))
    behavior = json.loads(
        (ROOT / "research/results/behavior_50m_long.json")
        .read_text(encoding="utf-8"))
    e2e = json.loads(
        (ROOT / "research/results/chat_human_eval_50m_long.json")
        .read_text(encoding="utf-8"))
    assert big["verdict"] == "PASS_CHAT_SMOKE"
    assert all(big["gate"].values())
    assert big["bpb"]["chat"]["bpb"] < small["bpb"]["chat"]["bpb"]
    assert (big["bpb"]["chat"]["assistant"]["loss"]
            < small["bpb"]["chat"]["assistant"]["loss"])
    assert behavior["passed"]
    assert e2e["verdict"] == "PASS_VIRTUAL_HUMAN_SMOKE"
    assert all(e2e["checks"].values())
