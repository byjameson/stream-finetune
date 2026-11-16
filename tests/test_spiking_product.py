import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import torch

from streamft.spiking import (
    SpikingConfig,
    SpikingLM,
    count_params,
    evaluate_bpb,
    generate,
    load_checkpoint,
    save_checkpoint,
    selective_rollback,
)
from streamft.spiking_server import Handler, _state


class FakeTokenizer:
    def encode(self, text):
        return type("Enc", (), {"ids": [1, 2, 3]})()

    def decode(self, ids):
        return " ".join(str(i) for i in ids)


def test_checkpoint_roundtrip_without_tokenizer(tmp_path):
    model = SpikingLM(SpikingConfig(vocab=32, dim=16, layers=1))
    save_checkpoint(model, tmp_path, tokenizer_path=None, meta={"step": 3})
    assert (tmp_path / "model.safetensors").exists()
    assert not (tmp_path / "model.pt").exists()
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


def test_feedback_starts_as_identity():
    torch.manual_seed(0)
    base = SpikingLM(SpikingConfig(vocab=32, dim=16, layers=2))
    fb = SpikingLM(SpikingConfig(vocab=32, dim=16, layers=2, feedback=True))
    fb.load_state_dict({k: v for k, v in base.state_dict().items()}, strict=False)
    x = torch.randint(0, 32, (2, 8))
    with torch.no_grad():
        assert torch.allclose(base(x), fb(x), atol=1e-6)
    assert fb.w_fb is not None


def test_attention_readout_matches_incremental_steps():
    torch.manual_seed(0)
    model = SpikingLM(SpikingConfig(vocab=32, dim=16, layers=2, attention=True))
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
    assert model.attn_q is not None
    assert len(state["hist_k"]) == x.shape[1]


def test_pointer_readout_matches_incremental_steps():
    torch.manual_seed(0)
    model = SpikingLM(SpikingConfig(vocab=32, dim=16, layers=2,
                                    pointer_offset=4))
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
    assert model.copy_scale is not None
    assert len(state["emb_hist"]) == x.shape[1]


def test_dropout_is_deterministic_in_eval():
    torch.manual_seed(0)
    model = SpikingLM(SpikingConfig(vocab=32, dim=16, layers=1, dropout=0.2))
    x = torch.randint(0, 32, (2, 8))
    model.train()
    model(x)
    model.eval()
    with torch.no_grad():
        a = model(x)
        b = model(x)
    assert torch.allclose(a, b)


def test_statement_memory_matches_incremental_steps():
    torch.manual_seed(0)
    model = SpikingLM(SpikingConfig(vocab=32, dim=16, layers=1, memory=True))
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
    assert model.last_copy_logits.shape == (2, 8, 32)
    assert len(state["mem_k"]) == x.shape[1]


def test_slot_memory_matches_incremental_steps():
    torch.manual_seed(0)
    model = SpikingLM(SpikingConfig(vocab=32, dim=16, layers=1, slots=4,
                                    mem_read_pos=5))
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
    assert model.last_copy_logits.shape == (2, 32)
    assert len(state["slot_k"]) == 1


def test_multihead_slot_memory_matches_incremental_steps():
    torch.manual_seed(0)
    model = SpikingLM(SpikingConfig(vocab=32, dim=16, layers=1, slots=4,
                                    slot_heads=4, mem_read_pos=5))
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
    assert model.last_copy_logits.shape == (2, 32)
    assert model.last_mem_attn.shape == (2, 1)


def test_attr_key_slot_memory_matches_incremental_steps():
    torch.manual_seed(0)
    model = SpikingLM(SpikingConfig(vocab=32, dim=16, layers=1, slots=4,
                                    slot_heads=4, slot_attr_keys=True,
                                    mem_read_pos=4))
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
    assert model.last_copy_logits.shape == (2, 32)
    assert model.last_mem_attn.shape == (2, 1)


def test_token_triggered_slot_memory_matches_incremental_steps():
    torch.manual_seed(0)
    model = SpikingLM(SpikingConfig(vocab=32, dim=16, layers=1, slots=3,
                                    slot_heads=2, slot_attr_keys=True,
                                    write_token=4, read_token=9,
                                    mem_read_pos=-1))
    x = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 4, 0, 1, 2, 3, 4,
                       6, 7, 8, 3, 9, 1, 0],
                      [1, 2, 3, 0, 4, 6, 7, 8, 5, 4, 1, 2, 3, 0, 4,
                       7, 8, 6, 3, 9, 2, 1]])
    with torch.no_grad():
        full = model(x)
        state = model.init_state(2, x.device)
        parts = []
        for t in range(x.shape[1]):
            out, state = model.step(x[:, t], state)
            parts.append(out)
        inc = torch.stack(parts, dim=1)
    assert torch.allclose(full, inc, atol=1e-6)
    assert model.last_copy_logits.shape == (2, 32)
    assert model.last_mem_attn.shape == (2, 3)


def test_factorized_slot_memory_matches_incremental_steps():
    torch.manual_seed(0)
    model = SpikingLM(SpikingConfig(vocab=32, dim=16, layers=1, slots=3,
                                    slot_attr_keys=True, slot_factorized=True,
                                    write_token=4, read_token=9,
                                    mem_read_pos=-1))
    x = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 4, 0, 1, 2, 3, 4,
                       6, 7, 8, 3, 9, 1, 0],
                      [1, 2, 3, 0, 4, 6, 7, 8, 5, 4, 1, 2, 3, 0, 4,
                       7, 8, 6, 3, 9, 2, 1]])
    with torch.no_grad():
        full = model(x)
        state = model.init_state(2, x.device)
        parts = []
        for t in range(x.shape[1]):
            out, state = model.step(x[:, t], state)
            parts.append(out)
        inc = torch.stack(parts, dim=1)
    assert torch.allclose(full, inc, atol=1e-6)
    assert model.last_copy_logits.shape == (2, 32)
    assert model.last_mem_attn.shape == (2, 3)


def test_checkpoint_writes_config_and_card(tmp_path):
    model = SpikingLM(SpikingConfig(vocab=32, dim=16, layers=1))
    save_checkpoint(model, tmp_path, tokenizer_path=None)
    cfg = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    card = json.loads((tmp_path / "model_card.json").read_text(encoding="utf-8"))
    assert cfg["vocab"] == 32
    assert card["params"] == count_params(model)


def test_evaluate_bpb_smoke():
    model = SpikingLM(SpikingConfig(vocab=32, dim=16, layers=1))
    x = torch.randint(0, 32, (1, 8))
    y = torch.randint(0, 32, (1, 8))
    out = evaluate_bpb(model, FakeTokenizer(), [(x, y)])
    assert out["tokens"] == 8
    assert out["bytes"] > 0
    assert out["bpb"] > 0


def test_http_health_auth_and_metrics():
    model = SpikingLM(SpikingConfig(vocab=32, dim=16, layers=1))
    _state.update({
        "model": model,
        "tok": FakeTokenizer(),
        "device": torch.device("cpu"),
        "name": "test-spiking",
        "params": count_params(model),
        "config": model.config.__dict__,
        "api_key": "secret",
        "metrics": {"requests": 0, "generated_tokens": 0},
    })
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"

    def call(path, payload=None, headers=None):
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(base + path, data=data,
                                     headers=headers or {})
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read())

    try:
        assert call("/health")[0] == 200
        assert call("/metrics")[1] == {"requests": 0, "generated_tokens": 0}
        assert call("/generate", {"prompt": "x", "max_new_tokens": 2})[0] == 401
        code, body = call("/generate", {"prompt": "x", "max_new_tokens": 2},
                          {"Authorization": "Bearer secret"})
        assert code == 200
        assert body["text"]
        assert call("/metrics")[1] == {"requests": 1, "generated_tokens": 2}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


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
