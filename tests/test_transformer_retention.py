import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_retention_skill_is_the_product_checkpoint():
    clean = json.loads(
        (ROOT / "research/results/transformer_chat_skill_ret_21m_banned.json")
        .read_text(encoding="utf-8"))
    skill = json.loads(
        (ROOT / "research/results/transformer_chat_skill_ret_21m_skill.json")
        .read_text(encoding="utf-8"))
    e2e = json.loads(
        (ROOT / "research/results/chat_human_eval_ret.json")
        .read_text(encoding="utf-8"))
    assert clean["verdict"] == "PASS_CHAT_SMOKE"
    assert clean["bpb"]["chat"]["bpb"] < 1.5
    assert clean["bpb"]["chat"]["assistant"]["loss"] < 4.4
    assert skill["verdict"] == "PASS_CHAT_SMOKE"
    assert skill["bpb"]["chat"]["assistant"]["loss"] < 0.5
    assert e2e["verdict"] == "PASS_VIRTUAL_HUMAN_SMOKE"
    assert all(e2e["checks"].values())
    assert e2e["open_ended_ok"]
