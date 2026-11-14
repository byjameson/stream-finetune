import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_arms(fname):
    payload = json.loads((ROOT / "research/results" / fname).read_text(encoding="utf-8"))
    return {r["name"]: r for r in payload["arms"]}


def test_shift_causes_forgetting_and_learning():
    for fname in ("spiking_shift.json", "spiking_shift_seed1.json"):
        arms = load_arms(fname)
        a = arms["phase_a"]
        b = arms["phase_b_no_rollback"]
        assert b["old"]["bpb"] > a["old"]["bpb"]
        assert b["new"]["bpb"] < a["new"]["bpb"]


def test_global_rollback_is_phase_a():
    for fname in ("spiking_shift.json", "spiking_shift_seed1.json"):
        arms = load_arms(fname)
        a = arms["phase_a"]
        g = arms["global_rollback"]
        assert abs(g["old"]["bpb"] - a["old"]["bpb"]) < 1e-9
        assert abs(g["new"]["bpb"] - a["new"]["bpb"]) < 1e-9


def test_ten_percent_rollback_keeps_both_tasks():
    for fname in ("spiking_shift.json", "spiking_shift_seed1.json"):
        arms = load_arms(fname)
        a = arms["phase_a"]
        b = arms["phase_b_no_rollback"]
        r = arms["importance_10"]
        old_recovery = ((b["old"]["bpb"] - r["old"]["bpb"])
                        / (b["old"]["bpb"] - a["old"]["bpb"]))
        new_gain = ((a["new"]["bpb"] - r["new"]["bpb"])
                    / (a["new"]["bpb"] - b["new"]["bpb"]))
        assert r["fraction"] == 0.1
        assert old_recovery > 0.90
        assert new_gain > 0.10
