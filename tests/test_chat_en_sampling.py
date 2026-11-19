import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_chat_sampling_sweep_does_not_rescue_the_21m_model():
    data = json.loads(
        (ROOT / "research/results/spiking_chat_en_sampling.json")
        .read_text(encoding="utf-8"))
    assert data["verdict"] == "DECODING_DOES_NOT_FIX_IT"
    assert len(data["rows"]) == 9
    assert {row["temperature"] for row in data["rows"]} == {0.3, 0.5, 0.7}
    for row in data["rows"]:
        assert row["chars"] > 0
        assert row["tag_leak"] == 0
    # The low-temperature samples are repetitive function-word loops, not
    # answers; this file documents that decoding is not the fix.
    assert max(row["repeat3"] for row in data["rows"]) > 0.1
