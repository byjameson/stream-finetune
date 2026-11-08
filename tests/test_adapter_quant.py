import torch
from torch import nn

from streamft.adapter import load_adapter, save_quantized, tensor_error
from streamft.lora import LoRALinear


def _state():
    torch.manual_seed(0)
    return {
        "model.layers.0.self_attn.q_proj.A": torch.randn(8, 2048) * 0.02,
        "model.layers.0.self_attn.q_proj.B": torch.randn(2048, 8) * 0.02,
    }


def test_roundtrip_8bit(tmp_path):
    state = _state()
    path = tmp_path / "a8.pt"
    save_quantized(state, path, 8, group=64)
    back, meta = load_adapter(path)
    assert set(back) == set(state)
    for k in state:
        assert back[k].shape == state[k].shape
        assert tensor_error(state[k], back[k]) < 0.02
    assert path.stat().st_size < sum(v.numel() for v in state.values()) * 4


def test_roundtrip_4bit(tmp_path):
    state = _state()
    path = tmp_path / "a4.pt"
    save_quantized(state, path, 4, group=64)
    back, _ = load_adapter(path)
    for k in state:
        assert back[k].shape == state[k].shape
        assert tensor_error(state[k], back[k]) < 0.15
    assert path.stat().st_size < sum(v.numel() for v in state.values()) * 2


def test_plain_adapter_load(tmp_path):
    state = _state()
    path = tmp_path / "plain.pt"
    torch.save({"lora": state, "meta": {"tag": "x"}}, path)
    back, meta = load_adapter(path)
    assert set(back) == set(state)
    assert meta["tag"] == "x"


def test_lora_linear_shapes():
    m = LoRALinear(nn.Linear(8, 4), r=2)
    assert m.A.shape == (2, 8)
    assert m.B.shape == (4, 2)
    assert m(torch.randn(3, 8)).shape == (3, 4)


def test_lora_fast_path():
    m = LoRALinear(nn.Linear(8, 4).to(torch.bfloat16), r=2)
    m.set_fast(torch.bfloat16)
    out = m(torch.randn(3, 8, dtype=torch.bfloat16))
    assert out.shape == (3, 4)
    assert out.dtype == torch.bfloat16
