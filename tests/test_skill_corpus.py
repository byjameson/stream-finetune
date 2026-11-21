import json
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent

pytestmark = pytest.mark.skipif(
    not (ROOT / "data/skill_meta.json").exists(),
    reason="local skill corpus artifact is not committed",
)


def test_skill_corpus_is_built_and_mixed():
    meta = json.loads(
        (ROOT / "data/skill_meta.json").read_text(encoding="utf-8"))
    assert meta["train_dialogues"] == 58800
    assert meta["valid_dialogues"] == 1200
    assert meta["skill_train_tokens"] > 2_000_000
    assert meta["mix_train_tokens"] > 5_000_000
    train = torch.load(ROOT / "data/skill_mix_train.pt", weights_only=True)
    assert train.numel() == meta["mix_train_tokens"]
    sample = (ROOT / "data/skill_sample.txt").read_text(encoding="utf-8")
    assert "<|user|>" in sample
    assert "<|assistant|>" in sample
