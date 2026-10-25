"""Smoke tests for the streamft package (CPU only, no model download).

They pin the things that broke silently during development:
  * a LoRA layer must be a *no-op* at init (B = 0), otherwise "before/after" is a lie;
  * chunked cross-entropy must produce exactly d(mean CE)/dh, not a scaled version;
  * only the adapter gets gradients (the NF4 base stays frozen).
"""

import math

import pytest
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


# ------------------------------------------- masked (SFT) chunked loss

def test_masked_chunked_gradient_matches_full_with_uneven_masks():
    """The 25.10 fix: chunks with different valid counts must not be re-weighted.

    With chunk=3 and a prompt mask over the first 4 positions, chunk 0 has zero valid
    targets, chunk 1 has two, later chunks have three. The old implementation divided
    each chunk by its own count, which inflated the sparse chunks' gradient.
    """
    from streamft.stream_quant_lora import chunked_lm_loss_and_backward_masked

    torch.manual_seed(0)
    d, T, V = 12, 10, 7
    head = nn.Linear(d, V, bias=False)
    ys = torch.randint(0, V, (1, T))
    ys[0, :4] = -100     # prompt
    ys[0, 7] = -100      # a masked middle position

    h1 = torch.randn(1, T, d, requires_grad=True)
    full = F.cross_entropy(head(h1).reshape(-1, V), ys.reshape(-1),
                           ignore_index=-100, reduction="mean")
    full.backward()

    h2 = torch.randn(1, T, d, requires_grad=True)
    with torch.no_grad():
        h2.copy_(h1.detach())
    _, mean = chunked_lm_loss_and_backward_masked(head, h2, ys, chunk=3)

    assert math.isclose(float(mean.detach()), float(full.detach()), rel_tol=1e-6)
    assert torch.allclose(h1.grad, h2.grad, atol=1e-6)


def test_masked_chunked_handles_fully_masked_chunks():
    """A chunk with no valid targets must contribute nothing at all."""
    from streamft.stream_quant_lora import chunked_lm_loss_and_backward_masked

    torch.manual_seed(0)
    d, T, V = 8, 6, 5
    head = nn.Linear(d, V, bias=False)
    ys = torch.randint(0, V, (1, T))
    ys[0, :3] = -100
    h = torch.randn(1, T, d, requires_grad=True)
    _, mean = chunked_lm_loss_and_backward_masked(head, h, ys, chunk=2)
    logits = head(h).reshape(-1, V)
    tgt = ys.reshape(-1)
    mask = tgt != -100
    ref = F.cross_entropy(logits[mask], tgt[mask], reduction="mean")
    assert math.isclose(float(mean.detach()), float(ref.detach()), rel_tol=1e-6)


# ------------------------------------------------ checkpoint file layouts

def test_disk_store_supports_single_safetensors_file(tmp_path):
    """save_pretrained writes one file for small NF4 models; the index is optional."""
    from safetensors.torch import save_file

    from streamft.disk_quant_store import DiskQuantStore

    save_file({"a": torch.zeros(4, dtype=torch.uint8),
               "b.c": torch.ones(3)}, str(tmp_path / "model.safetensors"))
    store = DiskQuantStore(str(tmp_path))
    assert store.has("a") and store.has("b.c")
    assert store.get("b.c").tolist() == [1.0, 1.0, 1.0]


# ------------------------------------------------ SFT masking failures are loud

class _StubTokenizer:
    """apply_chat_template returns FULL for both messages, PROMPT for the prefix call."""

    def __init__(self, full, prompt):
        self.full, self.prompt = full, prompt

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        return "FULL" if len(messages) == 2 else "PROMPT"

    def __call__(self, text, add_special_tokens=False):
        from types import SimpleNamespace

        ids = self.full if text == "FULL" else self.prompt
        return SimpleNamespace(input_ids=list(ids))


def test_encode_chat_drops_rows_whose_prompt_is_not_a_prefix():
    from streamft.cli import _encode_chat

    row = {"instruction": "q", "response": "a"}
    # prompt is not a prefix of the templated row -> the example must be dropped
    assert _encode_chat(_StubTokenizer([1, 2, 3, 4], [9, 9]), row, 32) is None
    # prompt longer than the row -> dropped
    assert _encode_chat(_StubTokenizer([1, 2], [1, 2, 3]), row, 32) is None
    # valid prefix -> every prompt token masked, the rest kept
    ids, labels = _encode_chat(_StubTokenizer([1, 2, 3, 4], [1, 2]), row, 32)
    assert ids.tolist() == [[1, 2, 3, 4]]
    assert labels.tolist() == [[-100, -100, 3, 4]]


# ------------------------------------------------ SFT step budget & empty eval

def test_sft_schedule_keeps_the_step_budget_with_mixed_rows():
    from streamft.cli import sft_schedule

    rows = list(range(10))

    def encode(r):
        return None if r % 3 == 0 else [r]      # 4 of 10 are rejected

    valid, dropped = sft_schedule(rows, encode)
    assert dropped == 4 and len(valid) == 6

    steps = 25
    schedule = [valid[i % len(valid)] for i in range(steps)]
    assert len(schedule) == steps               # N updates even though 4 rows were dropped
    assert all(v % 3 != 0 for v in schedule)


def test_masked_eval_raises_when_every_row_is_skipped(monkeypatch):
    import types

    from streamft import cli

    class FakeTrainer:
        def __init__(self):
            self.model = types.SimpleNamespace(eval=lambda: None, train=lambda: None)
            self.norm = nn.Identity()
            self.lm_head = nn.Linear(4, 7, bias=False)

        def _forward_cached(self, ids):
            return torch.zeros(1, ids.shape[1], 4), None, None

    monkeypatch.setattr(cli, "_encode_chat", lambda *a, **k: None)
    with pytest.raises(RuntimeError, match="every row was skipped"):
        cli._masked_eval(FakeTrainer(), [{"instruction": "x", "response": "y"}], object(), 8)


# ------------------------------------------------ synth / planner helpers

def test_parse_pair_rejects_template_junk():
    from streamft.easy import parse_pair

    good = parse_pair('{"instruction": "Give me a tip", "response": "Sharpen your knives."}')
    assert good and good["instruction"] == "Give me a tip"
    assert parse_pair('{"instruction": "<instruction>", "response": "<response>"}') is None
    assert parse_pair('{"instruction": "x", "response": "short"}') is None
    assert parse_pair("not json") is None
    assert parse_pair('{"instruction": "a" * 300, "response": "b" * 20}') is None


def test_kv_cache_matches_hand_calculation():
    from streamft.fit import kv_cache_gb

    cfg = {"hidden_size": 2048, "num_hidden_layers": 24, "num_attention_heads": 16,
           "num_key_value_heads": 2}
    # 2 * 24 layers * 2 kv heads * 128 head_dim * 1024 seq * 2 bytes
    expect = 2 * 24 * 2 * 128 * 1024 * 2 / 1e9
    assert abs(kv_cache_gb(cfg, 1024) - expect) < 1e-12
