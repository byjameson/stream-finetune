import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "research/results/spiking_ewc_sweep_50m.json"


def test_ewc_lambda_does_not_buy_semantics():
    payload = json.loads(DATA.read_text(encoding="utf-8"))
    a = payload["before"]
    arms = {float(x["lambda"]): x for x in payload["arms"]}
    assert set(arms) == {0.5, 2.0}
    for lam, arm in arms.items():
        assert arm["after"]["old"]["bpb"] < 1.32
        assert arm["after"]["new"]["bpb"] > 3.2
        assert 0.35 < arm["semantic"]["accuracy"] < 0.45
    assert payload["semantic_a"]["accuracy"] >= 0.35
    assert a["old"]["bpb"] < 1.32
