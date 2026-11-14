import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "research/results/spiking_long_replay_50m.json"


def test_50m_replay_keeps_new_gain_and_semantic_signal():
    payload = json.loads(DATA.read_text(encoding="utf-8"))
    a = payload["after_a"]
    arms = {x["arm"]: x for x in payload["arms"]}
    naive = arms["naive"]
    replay = arms["replay_20"]
    ewc = arms["replay_20_ewc"]

    naive_gain = a["new"]["bpb"] - naive["after"]["new"]["bpb"]
    replay_gain = a["new"]["bpb"] - replay["after"]["new"]["bpb"]
    ewc_gain = a["new"]["bpb"] - ewc["after"]["new"]["bpb"]
    assert payload["after_a"]["old"]["bpb"] < 1.4
    assert replay_gain / naive_gain > 0.95
    assert ewc_gain / naive_gain > 0.90
    assert ewc["after"]["old"]["bpb"] < replay["after"]["old"]["bpb"]
    assert replay["quality"]["semantic"]["content_f1"] > payload["phase_a_quality"]["semantic"]["content_f1"]
    assert naive["quality"]["semantic"]["content_f1"] == 0.0
