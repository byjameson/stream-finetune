import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from research.compare_adapters import (assistant_label_mask, label_for,
                                       score_rows)


class DummyTokenizer:
    ids = {
        "<|user|>": 0,
        "<|assistant|>": 1,
        "<|end|>": 2,
        "hello": 3,
        "hi": 4,
        "bye": 5,
    }

    def token_to_id(self, token):
        return self.ids.get(token)

    def encode(self, text):
        return SimpleNamespace(
            ids=[self.ids.get(word, 6) for word in text.split()])


class DummyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.vocab = 7
        self.max_seq = 32
        self.bias = nn.Parameter(torch.zeros(self.vocab))

    def forward(self, x):
        return self.bias.view(1, 1, -1).expand(x.size(0), x.size(1), -1)


def test_assistant_mask_covers_tag_answer_and_end():
    # y starts after <|user|>: hello <|assistant|> hi <|end|>
    y = torch.tensor([[3, 4, 1, 5, 6, 2, 7]])
    mask = assistant_label_mask(y, assistant_id=1, end_id=2)
    assert mask.tolist() == [[False, False, True, True, True, True, False]]


def test_score_rows_reports_assistant_only_loss():
    rows = [
        {"user": "hello", "assistant": "hi"},
        {"user": "bye", "assistant": "hello"},
    ]
    result = score_rows(DummyLM(), DummyTokenizer(), rows)
    assert result["examples"] == 2
    assert result["tokens"] == 6
    assert result["loss"] == pytest.approx(math.log(7.0), abs=1e-6)
    assert len(result["rows"]) == 2
    assert all(row["tokens"] == 3 for row in result["rows"])
    assert all(not row["truncated"] for row in result["rows"])


def test_label_for_adapter_path():
    assert label_for("checkpoints/adapters_feedback/v0002") == (
        "adapters_feedback/v0002")
    assert label_for("checkpoints/my_lora") == "my_lora"
