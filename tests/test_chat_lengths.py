import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_chat_conversations_are_longer_than_the_128_token_window():
    data = json.loads(
        (ROOT / "research/results/chat_en_clean_lengths.json")
        .read_text(encoding="utf-8"))
    assert data["conversations"] == 50924
    assert data["percentiles"]["p50"] == 355
    assert data["caps"]["128"]["fraction_of_conversations"] < 0.20
    assert data["caps"]["128"]["fraction_of_tokens_covered"] < 0.05
    assert data["caps"]["512"]["fraction_of_conversations"] > 0.50
    assert data["caps"]["1024"]["fraction_of_conversations"] > 0.80
