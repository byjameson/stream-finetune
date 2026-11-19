import json
from pathlib import Path

import torch
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent.parent


def test_turkish_chat_corpus_and_tokenizer_exist():
    meta = json.loads(
        (ROOT / "data/chat_tr_meta.json").read_text(encoding="utf-8"))
    assert meta["tokens"]["data/chat_tr_train.pt"] > 20_000_000
    assert meta["tokens"]["data/chat_tr_valid.pt"] > 100_000
    train = torch.load(ROOT / "data/chat_tr_train.pt", weights_only=True)
    assert train.numel() == meta["tokens"]["data/chat_tr_train.pt"]
    assert train.dtype == torch.int64


def test_turkish_chat_tokenizer_roundtrips_and_keeps_special_tokens():
    tok = Tokenizer.from_file(str(ROOT / "data/chat_tr_bpe8k.json"))
    assert tok.get_vocab_size() == 8192
    assert tok.token_to_id("<|user|>") == 0
    assert tok.token_to_id("<|assistant|>") == 1
    assert tok.token_to_id("<|end|>") == 2
    text = "merhaba, nas" + chr(0x131) + "ls" + chr(0x131) + "n?"
    ids = tok.encode(text).ids
    decoded = tok.decode(ids, skip_special_tokens=False)
    assert chr(0x131) in decoded
    assert tok.decode(tok.encode("<|end|>").ids,
                      skip_special_tokens=False) == "<|end|>"
