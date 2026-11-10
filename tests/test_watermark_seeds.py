import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FILES = {
    "seed1": ROOT / "research/results/watermark_threshold.json",
    "seed2": ROOT / "research/results/watermark_threshold_seed2.json",
    "seed3": ROOT / "research/results/watermark_threshold_seed3.json",
}


def load(seed):
    payload = json.loads(FILES[seed].read_text(encoding="utf-8"))
    return {(r["host"], r["wm_fraction"]): r for r in payload["results"]}


def test_watermark_threshold_is_seed_dependent():
    data = {seed: load(seed) for seed in FILES}
    for host in ("sst2", "agnews"):
        for seed in FILES:
            assert data[seed][(host, 0.50)]["hits"] == 0
            assert data[seed][(host, 1.00)]["hits"] == 8
        # 0.85 is 8/8 for seed 1 only; 0.90 is 8/8 for seed 1 and seed 3; only 1.00 is 8/8
        # across all three seeds.
        assert data["seed1"][(host, 0.85)]["hits"] == 8
        assert data["seed2"][(host, 0.85)]["hits"] == 5
        assert data["seed3"][(host, 0.85)]["hits"] == 6
        assert data["seed2"][(host, 0.90)]["hits"] == 7
