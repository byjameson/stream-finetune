import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_safe_space_recipe_passes_narrow_story_gate():
    data = json.loads(
        (ROOT / "research/results/spiking_space_safe2.json")
        .read_text(encoding="utf-8"))
    full = data["full_eval"]
    gen = data["generation"]
    assert full["target_improvement"] > 0.50
    assert full["old_regression"] < 0.05
    assert full["general_regression"] <= 0.05
    assert all(a >= 0.60 for a in gen["after_alpha_ratio"])
    assert all(e <= 0.05 for e in gen["after_equals_ratio"])
    assert all(r == 0 for r in gen["after_replacement_count"])
    assert data["release_gate"]["verdict"] == "PASS_NARROW_STORY_LAB"
