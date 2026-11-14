import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "research/results/spiking_replay_rollback.json"


def test_replay_beats_rollback_and_the_combination():
    payload = json.loads(DATA.read_text(encoding="utf-8"))
    arms = {a["arm"]: a for a in payload["arms"]}
    a = payload["after_a"]
    naive = arms["naive"]
    replay = arms["replay_20"]
    rollback = arms["rollback_10"]
    both = arms["replay_20_rollback_10"]

    # Replay keeps the old task and almost all of the new gain.
    assert abs(replay["after"]["old"]["bpb"] - a["old"]["bpb"]) < 0.05
    naive_gain = a["new"]["bpb"] - naive["after"]["new"]["bpb"]
    replay_gain = a["new"]["bpb"] - replay["after"]["new"]["bpb"]
    assert replay_gain / naive_gain > 0.95

    # Rollback recovers the old task but spends most of the new gain.
    assert rollback["after"]["old"]["bpb"] < 2.1
    rollback_gain = a["new"]["bpb"] - rollback["after"]["new"]["bpb"]
    assert rollback_gain / naive_gain < 0.25

    # Combining them does not beat replay-only on either axis.
    assert both["after"]["new"]["bpb"] > replay["after"]["new"]["bpb"]
    assert both["after"]["old"]["bpb"] >= replay["after"]["old"]["bpb"] - 0.02
