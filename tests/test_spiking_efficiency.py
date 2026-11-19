import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_tiny_story_models_are_small_and_fast():
    data = json.loads(
        (ROOT / "research/results/spiking_efficiency.json")
        .read_text(encoding="utf-8"))
    rows = {(r["model"], r["device"]): r for r in data["rows"]}
    mix = rows[("mix4m", "cuda")]
    big = rows[("ts21m", "cuda")]
    assert mix["tokens_per_s"] > big["tokens_per_s"]
    assert mix["peak_vram_mb"] < 200
    assert big["peak_vram_mb"] < 200
    assert rows[("mix4m", "cpu")]["tokens_per_s"] > 100
