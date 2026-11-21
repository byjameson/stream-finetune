from research.gated_update import decide


def test_gate_accepts_improvement_without_regression():
    base = {"clean": {"bpb": 1.44, "assistant_loss": 4.27},
            "queue": {"assistant_loss": 3.0}}
    cand = {"clean": {"bpb": 1.45, "assistant_loss": 4.30},
            "queue": {"assistant_loss": 2.7}}
    result = decide(base, cand)
    assert result["accepted"]
    assert result["target_gain"] > 0.05


def test_gate_rejects_clean_regression():
    base = {"clean": {"bpb": 1.44, "assistant_loss": 4.27},
            "queue": {"assistant_loss": 3.0}}
    cand = {"clean": {"bpb": 1.60, "assistant_loss": 4.60},
            "queue": {"assistant_loss": 2.0}}
    result = decide(base, cand)
    assert not result["accepted"]
    assert not result["checks"]["clean_bpb"]


def test_gate_rejects_behavior_failure_even_with_good_loss():
    base = {"clean": {"bpb": 1.44, "assistant_loss": 4.27},
            "queue": {"assistant_loss": 3.0}}
    cand = {"clean": {"bpb": 1.45, "assistant_loss": 4.30},
            "queue": {"assistant_loss": 2.0}}
    result = decide(base, cand, behavior_ok=False)
    assert not result["accepted"]
    assert not result["checks"]["behavior"]


def test_gate_rejects_empty_generation_smoke():
    base = {"clean": {"bpb": 1.44, "assistant_loss": 4.27},
            "queue": {"assistant_loss": 3.0}}
    cand = {"clean": {"bpb": 1.45, "assistant_loss": 4.30},
            "queue": {"assistant_loss": 2.0}}
    result = decide(base, cand, clean_smoke_ok=False)
    assert not result["accepted"]
    assert not result["checks"]["clean_smoke"]
