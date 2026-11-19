import json
from pathlib import Path

import pytest
import torch

pytest.importorskip("tokenizers")
from tokenizers import Tokenizer  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

pytestmark = pytest.mark.skipif(
    not (ROOT / "data/chat_en_clean_meta.json").exists(),
    reason="local clean chat corpus artifact is not committed",
)


def test_clean_english_chat_corpus():
    meta = json.loads(
        (ROOT / "data/chat_en_clean_meta.json").read_text(encoding="utf-8"))
    assert meta["language"] == "en"
    assert meta["tokens"]["data/chat_en_clean_train.pt"] > 20_000_000
    train = torch.load(ROOT / "data/chat_en_clean_train.pt",
                       weights_only=True)
    assert train.numel() == meta["tokens"]["data/chat_en_clean_train.pt"]
    tok = Tokenizer.from_file(str(ROOT / "data/chat_en_bpe8k.json"))
    assert tok.token_to_id("<|end|>") == 2
