import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_story_recipe_replicates_across_corpora_and_seeds():
    data = json.loads(
        (ROOT / "research/results/spiking_story_seedcheck.json")
        .read_text(encoding="utf-8"))
    runs = data["runs"]
    assert len(runs) == 4
    assert {r["corpus"] for r in runs} == {"space", "nova"}
    assert {r["seed"] for r in runs} == {0, 1}
    for r in runs:
        assert r["target_improvement"] > 0.60
        assert r["ts_regression"] < 0.05
        assert r["wt_regression"] <= 0.05
    assert all(data["gate"].values())
