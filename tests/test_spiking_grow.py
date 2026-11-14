import torch

from streamft.spiking import SpikingConfig, SpikingLM, count_params, save_checkpoint
from streamft.spiking_grow import grow_checkpoint, grow_width


def test_width_growth_is_function_preserving():
    torch.manual_seed(0)
    model = SpikingLM(SpikingConfig(vocab=32, dim=16, layers=1))
    x = torch.randint(0, 32, (2, 8))
    with torch.no_grad():
        before = model(x)
        grown = grow_width(model, 24)
        after = grown(x)
    assert grown.config.dim == 24
    assert count_params(grown) > count_params(model)
    assert torch.allclose(before, after, atol=1e-6)


def test_grow_checkpoint_roundtrip(tmp_path):
    model = SpikingLM(SpikingConfig(vocab=32, dim=16, layers=1))
    save_checkpoint(model, tmp_path / "old", meta={"name": "tiny"})
    result = grow_checkpoint(str(tmp_path / "old"), str(tmp_path / "new"), 24)
    assert result["new_params"] > result["old_params"]
    assert (tmp_path / "new" / "model.safetensors").exists()
    from streamft.spiking import load_checkpoint
    grown, tok, meta = load_checkpoint(tmp_path / "new")
    assert tok is None
    assert grown.config.dim == 24
    assert meta["grown_from"].endswith("old")


def test_shrink_is_rejected():
    model = SpikingLM(SpikingConfig(vocab=32, dim=16, layers=1))
    try:
        grow_width(model, 8)
    except ValueError:
        return
    raise AssertionError("shrink should fail")
