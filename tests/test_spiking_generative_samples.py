import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_generative_samples_are_not_degenerate_but_remain_weak():
    payload = json.loads(
        (ROOT / "research/results/spiking_generative_samples.json")
        .read_text(encoding="utf-8"))
    rows = payload["rows"]
    assert len(rows) == 12
    for row in rows:
        # Not collapsed to one token or a tight loop.
        assert row["adjacent_repeat_rate"] < 0.2
        assert row["top_word_frac"] < 0.25
        assert row["distinct_2"] > 0.8
        # But short samples are not evidence of general language quality.
        assert row["words"] >= 50
