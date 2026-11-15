import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "research/results/spiking_binding_continual.json"


def test_replay_first_holds_on_continual_binding():
    payload = json.loads(DATA.read_text(encoding="utf-8"))
    before = payload["before"]
    arms = {a["arm"]: a["after"] for a in payload["arms"]}

    assert before["old"]["accuracy"] > 0.9
    assert before["new"]["accuracy"] > 0.7

    # Naive phase B forgets the old triples.
    assert arms["naive"]["old"]["accuracy"] < 0.4
    assert arms["naive"]["new"]["accuracy"] > 0.95

    # Replay keeps old and new binding.
    assert arms["replay_20"]["old"]["accuracy"] > 0.95
    assert arms["replay_20"]["new"]["accuracy"] > 0.95
    assert arms["replay_20"]["held"]["accuracy"] > before["held"]["accuracy"]

    # Rollback alone is not enough.
    assert arms["rollback_10"]["old"]["accuracy"] < 0.5
    assert arms["rollback_10"]["new"]["accuracy"] < arms["replay_20"]["new"]["accuracy"]

    # The combination is not additive.
    assert arms["replay_20_rollback_10"]["new"]["accuracy"] < arms["replay_20"]["new"]["accuracy"]

    # EWC is close but does not beat replay.
    assert arms["replay_20_ewc"]["old"]["accuracy"] > 0.95
    assert arms["replay_20_ewc"]["new"]["accuracy"] < arms["replay_20"]["new"]["accuracy"]
