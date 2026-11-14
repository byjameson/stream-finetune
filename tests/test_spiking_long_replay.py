import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "research/results/spiking_long_replay.json"


def test_long_phase_b_replay_first_and_quality():
    payload = json.loads(DATA.read_text(encoding="utf-8"))
    a = payload["after_a"]
    arms = {x["arm"]: x for x in payload["arms"]}
    naive = arms["naive"]
    replay = arms["replay_20"]
    ewc = arms["replay_20_ewc"]

    naive_gain = a["new"]["bpb"] - naive["after"]["new"]["bpb"]
    replay_gain = a["new"]["bpb"] - replay["after"]["new"]["bpb"]
    ewc_gain = a["new"]["bpb"] - ewc["after"]["new"]["bpb"]

    assert replay_gain / naive_gain > 0.99
    assert ewc_gain / naive_gain > 0.98
    assert replay["after"]["old"]["bpb"] < a["old"]["bpb"]
    assert ewc["after"]["old"]["bpb"] < a["old"]["bpb"]

    # Generation collapse in naive is visible in length and repetition.
    assert naive["quality"]["chars_mean"] < replay["quality"]["chars_mean"] * 0.7
    assert naive["quality"]["repeat_4gram"] > replay["quality"]["repeat_4gram"] * 3
    assert ewc["quality"]["chars_mean"] >= replay["quality"]["chars_mean"]
