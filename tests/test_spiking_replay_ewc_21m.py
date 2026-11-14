import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "research/results/spiking_replay_ewc_21m.json"


def test_replay_first_holds_at_21m():
    payload = json.loads(DATA.read_text(encoding="utf-8"))
    assert len(payload["seeds"]) == 2
    for run in payload["seeds"]:
        a = run["after_a"]
        arms = {x["arm"]: x for x in run["arms"]}
        naive = arms["naive"]
        replay = arms["replay_20"]
        rollback = arms["replay_20_rollback_10"]
        ewc = arms["replay_20_ewc"]
        naive_gain = a["new"]["bpb"] - naive["after"]["new"]["bpb"]
        replay_gain = a["new"]["bpb"] - replay["after"]["new"]["bpb"]
        rollback_gain = a["new"]["bpb"] - rollback["after"]["new"]["bpb"]
        ewc_gain = a["new"]["bpb"] - ewc["after"]["new"]["bpb"]
        # Replay protects old and retains >98% of new learning.
        assert abs(replay["after"]["old"]["bpb"] - a["old"]["bpb"]) < 0.06
        assert replay_gain / naive_gain > 0.98
        # Hard rollback after replay spends the new learning.
        assert rollback_gain / naive_gain < 0.15
        # EWC is close to replay, not a large improvement.
        assert ewc_gain / naive_gain > 0.98
        assert abs(ewc["after"]["new"]["bpb"] - replay["after"]["new"]["bpb"]) < 0.05
