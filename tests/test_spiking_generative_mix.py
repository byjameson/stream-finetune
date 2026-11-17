import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_runs():
    payload = json.loads(
        (ROOT / "research/results/spiking_generative_mix.json")
        .read_text(encoding="utf-8"))
    return {r["name"]: r for r in payload["runs"]}


def test_data_diversity_lowers_wikitext_bpb():
    runs = load_runs()
    mix = runs["mix_3k"]
    ts_only = runs["ts_only_3k"]
    assert mix["wikitext_bpb"] < 3.0
    assert ts_only["wikitext_bpb"] > 4.4
    assert mix["wikitext_bpb"] < ts_only["wikitext_bpb"] - 1.0
    assert mix["tinystories_bpb"] < 2.0
    assert mix["tinystories_bpb"] > ts_only["tinystories_bpb"]


def test_parameter_scaling_does_not_move_wikitext():
    runs = load_runs()
    assert runs["spiking_50m"]["wikitext_bpb"] > 4.0
    assert runs["spiking_21m"]["wikitext_bpb"] > 4.0
    assert abs(runs["spiking_50m"]["wikitext_bpb"]
               - runs["spiking_21m"]["wikitext_bpb"]) < 0.01
