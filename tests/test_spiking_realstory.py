import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_real_public_domain_cross_book_gate_is_blocked():
    data = json.loads(
        (ROOT / "research/results/spiking_realstory_seedcheck.json")
        .read_text(encoding="utf-8"))
    assert data["verdict"] == "BLOCKED"
    assert len(data["runs"]) == 2
    assert {r["seed"] for r in data["runs"]} == {0, 1}
    for r in data["runs"]:
        # bpb adapts and retains, but generation is the blocker.
        assert r["target_improvement"] >= 0.10
        assert r["ts_regression"] < 0.05
        assert r["wt_regression"] <= 0.05
        assert min(r["generation"]["alpha_ratio"]) < 0.60
        assert r["worst_samples"]
    checks = data["gate"]
    assert checks["target_improves_10pct"]
    assert checks["old_ts_within_5pct"]
    assert checks["general_wt_within_5pct"]
    assert not checks["generation_alpha"]
    assert checks["generation_no_formatting"]
    assert checks["generation_no_replacement"]
    # The base model is healthy on the same cross-book prompt, so the collapse
    # comes from the adaptation, not the prompt.
    assert min(data["baseline"]["generation_alpha_ratio"]) >= 0.60
