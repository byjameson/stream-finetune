import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "research/results/spiking_feedback.json"


def test_top_down_feedback_did_not_help():
    payload = json.loads(DATA.read_text(encoding="utf-8"))
    arms = {a["arm"]: a for a in payload["arms"]}
    base = arms["baseline"]
    fb = arms["feedback"]
    assert base["semantic"]["accuracy"] > 0.42
    assert fb["semantic"]["accuracy"] < base["semantic"]["accuracy"]
    assert fb["old"]["bpb"] > base["old"]["bpb"]
