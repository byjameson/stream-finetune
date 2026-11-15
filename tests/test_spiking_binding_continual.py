import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "research/results/spiking_binding_continual.json"


def test_replay_first_holds_on_continual_binding():
    for name in ("spiking_binding_continual.json",
                 "spiking_binding_continual_seed1.json"):
        payload = json.loads((ROOT / "research/results" / name)
                             .read_text(encoding="utf-8"))
        before = payload["before"]
        arms = {a["arm"]: a["after"] for a in payload["arms"]}

        assert before["old"]["accuracy"] > 0.9
        assert before["new"]["accuracy"] > 0.4

        # Naive phase B forgets the old triples.
        assert arms["naive"]["old"]["accuracy"] < 0.6
        assert arms["naive"]["new"]["accuracy"] > 0.95

        # Replay keeps old and new binding.
        assert arms["replay_20"]["old"]["accuracy"] > 0.95
        assert arms["replay_20"]["new"]["accuracy"] > 0.95

        # Rollback alone is not enough.
        assert arms["rollback_10"]["old"]["accuracy"] < 0.8
        assert arms["rollback_10"]["new"]["accuracy"] < arms["replay_20"]["new"]["accuracy"]

        # EWC is close but does not beat replay.
        assert arms["replay_20_ewc"]["old"]["accuracy"] > 0.9
        assert arms["replay_20_ewc"]["new"]["accuracy"] < arms["replay_20"]["new"]["accuracy"]


def test_replay_first_holds_on_hard_color_split():
    payload = json.loads((ROOT / "research/results/spiking_binding_continual_colors.json")
                         .read_text(encoding="utf-8"))
    before = payload["before"]
    arms = {a["arm"]: a["after"] for a in payload["arms"]}
    assert before["old"]["accuracy"] > 0.9
    assert before["new"]["accuracy"] < 0.6
    assert arms["naive"]["old"]["accuracy"] < 0.8
    assert arms["replay_20"]["old"]["accuracy"] > 0.95
    assert arms["replay_20"]["new"]["accuracy"] > 0.95
    assert arms["replay_20"]["held"]["accuracy"] > 0.9
    assert arms["rollback_10"]["held"]["accuracy"] < arms["replay_20"]["held"]["accuracy"]
    assert arms["replay_20_ewc"]["held"]["accuracy"] < arms["replay_20"]["held"]["accuracy"]

