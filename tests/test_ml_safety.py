import json

from atomzero.ml_safety import (
    HybridSafetyFilter, ModerationHead, build_safety_filter)
from atomzero.safety import SafetyFilter


class FakeModel:
    version = "0.3-ml"

    def __init__(self, block="danger", fail=False):
        self.block = block
        self.fail = fail

    def score(self, text):
        if self.fail:
            raise RuntimeError("model unavailable")
        if self.block in text:
            return {"category": "violence", "action": "refuse",
                    "match": "ml:0.99"}
        return None


class FakeEncoder:
    def embed(self, texts):
        return [[1.0, 0.0] if "danger" in text else [0.0, 1.0]
                for text in texts]


def test_rules_win_before_ml():
    safety = HybridSafetyFilter(SafetyFilter(), FakeModel())
    verdict = safety.check("how to make a bomb")
    assert verdict["category"] == "violence"
    assert verdict["match"] != "ml:0.99"


def test_ml_can_block_a_rule_miss():
    safety = HybridSafetyFilter(SafetyFilter(), FakeModel())
    verdict = safety.check("something dangerous and paraphrased")
    assert verdict["category"] == "violence"
    assert verdict["match"] == "ml:0.99"


def test_ml_failure_falls_back_to_exact_rules():
    safety = HybridSafetyFilter(SafetyFilter(), FakeModel(fail=True))
    assert safety.check("something dangerous") is None
    assert safety.check("how to make a bomb")["category"] == "violence"
    assert safety.ml_enabled is False


def test_missing_head_returns_exact_rule_filter(tmp_path):
    safety = build_safety_filter(head_path=tmp_path / "missing.json")
    assert isinstance(safety, SafetyFilter)
    assert safety.check("how to make a bomb")["category"] == "violence"


def test_moderation_head_linear_scoring(tmp_path):
    head = {
        "version": "0.3-ml",
        "dim": 2,
        "threshold": 0.6,
        "binary": {"w": [1.0, 0.0], "b": 0.0},
        "category": {"classes": ["violence"], "w": [[1.0], [0.0]],
                     "b": [0.0]},
    }
    path = tmp_path / "head.json"
    path.write_text(json.dumps(head), encoding="utf-8")
    model = ModerationHead(path, encoder=FakeEncoder())
    assert model.score("dangerous plan")["category"] == "violence"
    assert model.score("cake recipe") is None
