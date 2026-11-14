import json

import torch

from streamft.spiking import (
    SpikingConfig,
    SpikingLM,
    count_params,
    generate,
    load_checkpoint,
    save_checkpoint,
    selective_rollback,
)


class FakeTokenizer:
    def encode(self, text):
        return type("Enc", (), {"ids": [1, 2, 3]})()

    def decode(self, ids):
        return " ".join(str(i) for i in ids)


def test_checkpoint_roundtrip_without_tokenizer(tmp_path):
    model = SpikingLM(SpikingConfig(vocab=32, dim=16, layers=1))
    save_checkpoint(model, tmp_path, tokenizer_path=None, meta={"step": 3})
    loaded, tok, meta = load_checkpoint(tmp_path)
    assert tok is None
    assert meta["step"] == 3
    assert count_params(loaded) == count_params(model)
    x = torch.randint(0, 32, (2, 8))
    with torch.no_grad():
        assert torch.allclose(model(x), loaded(x))


def test_incremental_step_matches_forward():
    model = SpikingLM(SpikingConfig(vocab=32, dim=16, layers=1))
    x = torch.randint(0, 32, (2, 8))
    with torch.no_grad():
        full = model(x)
        state = model.init_state(2, x.device)
        parts = []
        for t in range(x.shape[1]):
            out, state = model.step(x[:, t], state)
            parts.append(out)
        inc = torch.stack(parts, dim=1)
    assert torch.allclose(full, inc, atol=1e-6)


def test_generate_greedy_and_sampled():
    model = SpikingLM(SpikingConfig(vocab=32, dim=16, layers=1))
    tok = FakeTokenizer()
    greedy = generate(model, tok, "x", 4)
    sampled = generate(model, tok, "x", 4, temperature=0.8, top_k=8)
    assert isinstance(greedy, str) and greedy
    assert isinstance(sampled, str) and sampled


def test_checkpoint_writes_config_and_card(tmp_path):
    model = SpikingLM(SpikingConfig(vocab=32, dim=16, layers=1))
    save_checkpoint(model, tmp_path, tokenizer_path=None)
    cfg = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    card = json.loads((tmp_path / "model_card.json").read_text(encoding="utf-8"))
    assert cfg["vocab"] == 32
    assert card["params"] == count_params(model)


def test_selective_rollback_restores_a_fraction():
    model = SpikingLM(SpikingConfig(vocab=32, dim=16, layers=1))
    state_a = {k: torch.zeros_like(v) for k, v in model.state_dict().items()}
    state_b = {k: torch.ones_like(v) for k, v in model.state_dict().items()}
    imp = {k: torch.ones_like(v, dtype=torch.float32)
           for k, v in model.state_dict().items()}
    out = selective_rollback(state_a, state_b, imp, model, 0.10)
    flat = torch.cat([v.flatten() for v in out.values()])
    assert flat.numel() == count_params(model)
    assert float(flat.min()) == 0.0
    assert float(flat.max()) == 1.0
