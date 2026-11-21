import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_gated_lora_50m_is_accepted_and_passes_all_checks():
    gate = json.loads(
        (ROOT / "research/results/gated_lora_50m.json")
        .read_text(encoding="utf-8"))
    clean = json.loads(
        (ROOT / "research/results/accepted_lora_50m_banned.json")
        .read_text(encoding="utf-8"))
    behavior = json.loads(
        (ROOT / "research/results/behavior_lora_50m.json")
        .read_text(encoding="utf-8"))
    e2e = json.loads(
        (ROOT / "research/results/chat_human_eval_lora_50m.json")
        .read_text(encoding="utf-8"))
    assert gate["decision"]["accepted"] is True
    assert gate["decision"]["target_gain"] >= 0.05
    assert gate["decision"]["clean_bpb_reg"] <= 0.03
    assert gate["accepted_to"] == "checkpoints\\adapters_50m\\v0001"
    assert clean["verdict"] == "PASS_CHAT_SMOKE"
    assert clean["bpb"]["chat"]["bpb"] < 1.4
    assert behavior["passed"]
    assert e2e["verdict"] == "PASS_VIRTUAL_HUMAN_SMOKE"
    assert all(e2e["checks"].values())
