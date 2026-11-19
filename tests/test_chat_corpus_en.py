import json
from pathlib import Path

import pytest
import torch

pytest.importorskip("tokenizers")
from tokenizers import Tokenizer  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

pytestmark = pytest.mark.skipif(
    not (ROOT / "data/chat_en_meta.json").exists(),
    reason="local English corpus artifact is not committed",
)


def test_english_chat_corpus_and_tokenizer_exist():
    meta = json.loads(
        (ROOT / "data/chat_en_meta.json").read_text(encoding="utf-8"))
    assert meta["language"] == "en"
    assert meta["tokens"]["data/chat_en_train.pt"] > 20_000_000
    assert meta["tokens"]["data/chat_en_valid.pt"] > 100_000
    train = torch.load(ROOT / "data/chat_en_train.pt", weights_only=True)
    assert train.numel() == meta["tokens"]["data/chat_en_train.pt"]
    assert train.dtype == torch.int64


def test_english_chat_tokenizer_keeps_special_tokens():
    tok = Tokenizer.from_file(str(ROOT / "data/chat_en_bpe8k.json"))
    assert tok.get_vocab_size() == 8192
    assert tok.token_to_id("<|user|>") == 0
    assert tok.token_to_id("<|assistant|>") == 1
    assert tok.token_to_id("<|end|>") == 2
    text = "<|user|> hello<|end|><|assistant|> hi there<|end|>"
    decoded = tok.decode(tok.encode(text).ids, skip_special_tokens=False)
    assert decoded == text
