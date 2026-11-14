import json
import random
from pathlib import Path

from research.spiking_binding import LEN, NAMES, make_split, sample_story

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "research/results/spiking_binding.json"


def test_binding_split_is_disjoint_and_fixed_length():
    train, test = make_split(0)
    assert len(train) == 48 and len(test) == 16
    assert not set(train) & set(test)
    story = sample_story(train, random.Random(0))
    assert len(story) == LEN


def test_binding_results_are_near_chance():
    payload = json.loads(DATA.read_text(encoding="utf-8"))
    rows = {r["mode"]: r for r in payload["results"]}
    assert rows["spike"]["test_acc"] < 0.25
    assert rows["rate"]["test_acc"] > 0.25
    assert max(r["test_acc"] for r in rows.values()) < 0.4
