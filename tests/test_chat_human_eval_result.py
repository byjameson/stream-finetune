import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_virtual_human_end_to_end_smoke_passes():
    data = json.loads(
        (ROOT / "research/results/chat_human_eval.json")
        .read_text(encoding="utf-8"))
    assert data["verdict"] == "PASS_VIRTUAL_HUMAN_SMOKE"
    assert data["open_ended_ok"]
    assert all(data["checks"].values())
    assert len(data["turns"]) == 13
    assert data["turns"][0]["assistant"].lower().startswith(("hello", "hi"))
    assert "well" in data["turns"][11]["assistant"].lower()
    assert len(data["turns"][12]["assistant"]) > 40
