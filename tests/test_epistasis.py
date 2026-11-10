import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "research/results/epistasis.json"


def test_epistasis_matrix_pins_the_watermark_result():
    payload = json.loads(DATA.read_text(encoding="utf-8"))
    assert payload["rows"] == {"sst2": 400, "agnews": 400, "watermark_triggers": 8}
    by_name = {row["name"]: row for row in payload["results"]}
    assert set(by_name) == {
        "base", "sst2", "agnews", "watermark",
        "sst2+agnews", "sst2+watermark", "agnews+watermark", "all_three",
    }
    # The watermark is perfect alone and disappears in every merge.
    assert by_name["watermark"]["watermark"] == 1.0
    for name in ("sst2+watermark", "agnews+watermark", "all_three"):
        assert by_name[name]["watermark"] == 0.0
    # The task adapters still work in the pair merges.
    assert by_name["sst2"]["sst2"] > 0.9
    assert by_name["sst2+agnews"]["sst2"] > 0.9
    assert by_name["agnews"]["agnews"] < 0.5
