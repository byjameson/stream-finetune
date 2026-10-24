"""Smoke tests for the streamft package (CPU only, no model download).

They pin the things that broke silently during development:
  * a LoRA layer must be a *no-op* at init (B = 0), otherwise "before/after" is a lie;
  * chunked cross-entropy must produce exactly d(mean CE)/dh, not a scaled version;
  * only the adapter gets gradients (the NF4 base stays frozen).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from streamft import cli, stream_train
from streamft.lora import LoRALinear, apply_lora, count_lora_params, lora_parameters


# ---------------------------------------------------------------- lora

def test_lora_is_noop_at_init():
    torch.manual_seed(0)
    base = nn.Linear(16, 8)
    lora = LoRALinear(base, r=4, alpha=16)
    x = torch.randn(3, 16)
    assert torch.allclose(lora(x), base(x), atol=1e-6)
    assert lora(x).shape == (3, 8)
    assert sum(p.numel() for p in lora.parameters() if p.requires_grad) == 4 * (16 + 8)


def test_gradients_reach_only_the_adapter():
    torch.manual_seed(0)
    base = nn.Linear(16, 8)
    lora = LoRALinear(base, r=4)
    lora(torch.randn(3, 16)).sum().backward()
    assert base.weight.grad is None
    assert lora.A.grad is not None
    assert lora.B.grad is not None


class _Attn(nn.Module):
    def __init__(self, d):
        super().__init__()
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(self, name, nn.Linear(d, d, bias=False))


class _Mlp(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.gate_proj = nn.Linear(d, 2 * d, bias=False)
        self.up_proj = nn.Linear(d, 2 * d, bias=False)
        self.down_proj = nn.Linear(2 * d, d, bias=False)


class _Layer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.self_attn = _Attn(d)
        self.mlp = _Mlp(d)


class _FakeQwen(nn.Module):
    """Just enough structure for apply_lora: model.layers[i].{self_attn,mlp}."""

    def __init__(self, d=16, n=2):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_Layer(d) for _ in range(n)])


def test_apply_lora_counts_and_freezes():
    model = _FakeQwen()
    apply_lora(model, r_attn=4, r_mlp=4)

    r, d = 4, 16
    per_proj_16 = r * (d + d)          # q/k/v/o and down_proj
    per_proj_32 = r * (d + 2 * d)      # gate/up_proj
    expected = 2 * (4 * per_proj_16 + 3 * per_proj_32)
    assert count_lora_params(model) == expected

    q = model.model.layers[0].self_attn.q_proj
    assert isinstance(q, LoRALinear)
    assert all(p.requires_grad is False for p in q.base.parameters())

    trainable = [p for p in model.parameters() if p.requires_grad]
    assert trainable and len(list(lora_parameters(model))) == len(trainable)


# ---------------------------------------------------- chunked loss

def test_chunked_ce_gradient_matches_full_mean():
    torch.manual_seed(0)
    d, T, V = 12, 9, 7
    head = nn.Linear(d, V, bias=False)
    ys = torch.randint(0, V, (1, T))

    h1 = torch.randn(1, T, d, requires_grad=True)
    full = F.cross_entropy(head(h1).reshape(-1, V), ys.reshape(-1), reduction="mean")
    full.backward()

    h2 = torch.randn(1, T, d, requires_grad=True)
    with torch.no_grad():
        h2.copy_(h1.detach())
    chunked_last, chunked_mean = stream_train.chunked_lm_loss_and_backward(head, h2, ys, chunk=4)

    assert math.isclose(float(chunked_mean.detach()), float(full.detach()), rel_tol=1e-5)
    assert torch.allclose(h1.grad, h2.grad, atol=1e-6)


# ---------------------------------------------------------------- cli

def test_cli_is_importable():
    assert callable(cli.main)
