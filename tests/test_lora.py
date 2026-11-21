import torch
import torch.nn as nn

from streamft.lora import (apply_adapter, inject_lora, lora_parameters,
                           lora_state_dict, save_adapter)


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                                 nn.Linear(8, 4))

    def forward(self, x):
        return self.mlp(x)


def test_lora_starts_as_identity_and_only_adapters_train():
    torch.manual_seed(0)
    base = Tiny()
    clone = Tiny()
    clone.load_state_dict(base.state_dict())
    inject_lora(base, rank=2, alpha=4)
    x = torch.randn(3, 4)
    assert torch.allclose(base(x), clone(x), atol=1e-6)
    trainable = lora_parameters(base)
    assert trainable
    assert all(name.endswith((".A", ".B"))
               for name, p in base.named_parameters() if p.requires_grad)
    assert sum(p.numel() for p in trainable) < sum(
        p.numel() for p in base.parameters())


def test_adapter_save_and_load(tmp_path):
    torch.manual_seed(1)
    base = Tiny()
    original = Tiny()
    original.load_state_dict(base.state_dict())
    inject_lora(base, rank=2, alpha=4)
    with torch.no_grad():
        for name, p in base.named_parameters():
            if name.endswith(".B"):
                p.normal_()
    adapter = save_adapter(base, tmp_path / "adapter", rank=2, alpha=4,
                           base="tiny")
    assert (adapter / "adapter.pt").exists()
    fresh = Tiny()
    fresh.load_state_dict(original.state_dict())
    apply_adapter(fresh, adapter)
    x = torch.randn(3, 4)
    assert torch.allclose(base(x), fresh(x), atol=1e-6)
    assert lora_state_dict(base).keys() == lora_state_dict(fresh).keys()
