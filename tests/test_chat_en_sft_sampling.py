import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_sft_sampling_sweep_still_needs_the_turn_prompt():
    data = json.loads(
        (ROOT / "research/results/spiking_chat_en_sft_sampling.json")
        .read_text(encoding="utf-8"))
    assert data["verdict"] == "DECODING_DOES_NOT_FIX_IT"
    assert len(data["rows"]) == 9
    assert {row["temperature"] for row in data["rows"]} == {0.3, 0.5, 0.7}
    for row in data["rows"]:
        assert row["chars"] > 0
        assert row["tag_leak"] == 0
