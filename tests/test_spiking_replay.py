import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "research/results/spiking_replay.json"


def test_replay_protects_old_task_at_low_new_cost():
    payload = json.loads(DATA.read_text(encoding="utf-8"))
    arms = {float(a["replay"]): a for a in payload["arms"]}
    base = payload["after_a"]
    naive = arms[0.0]
    replay = arms[0.2]
    # Naive sequential training forgets; 20% replay does not.
    assert naive["after"]["old"]["bpb"] > base["old"]["bpb"] + 0.4
    assert abs(replay["after"]["old"]["bpb"] - base["old"]["bpb"]) < 0.02
    # Replay keeps almost all of the new-task gain.
    naive_gain = base["new"]["bpb"] - naive["after"]["new"]["bpb"]
    replay_gain = base["new"]["bpb"] - replay["after"]["new"]["bpb"]
    assert replay_gain / naive_gain > 0.95
