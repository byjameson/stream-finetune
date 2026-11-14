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
