import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_accepted_v0001_passes_clean_and_e2e_checks():
    clean = json.loads(
        (ROOT / "research/results/accepted_v0001_banned.json")
        .read_text(encoding="utf-8"))
    e2e = json.loads(
        (ROOT / "research/results/chat_human_eval_v0001.json")
        .read_text(encoding="utf-8"))
    assert clean["verdict"] == "PASS_CHAT_SMOKE"
    assert all(clean["gate"].values())
    assert clean["bpb"]["chat"]["bpb"] < 1.5
    assert clean["bpb"]["chat"]["assistant"]["loss"] < 4.4
    assert e2e["verdict"] == "PASS_VIRTUAL_HUMAN_SMOKE"
    assert all(e2e["checks"].values())
    assert e2e["open_ended_ok"]
