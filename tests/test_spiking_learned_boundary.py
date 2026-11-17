import torch

from streamft.spiking import SpikingConfig, SpikingLM


def test_learned_boundary_gates_and_color_copy_scale():
    model = SpikingLM(SpikingConfig(
        vocab=32, dim=16, layers=1, memory=False, slots=2,
        slot_attr_keys=True, slot_factorized=True, query_color_copy=True,
        color_copy_init=3.0, learned_boundaries=True,
        write_token=-1, read_token=-1, mem_read_pos=-1))
    assert model.boundary_write is not None
    assert model.boundary_read is not None
    assert abs(model.color_copy_scale.item() - 3.0) < 1e-6

    x = torch.randint(0, 32, (2, 8))
    out = model(x)
    assert out.shape == (2, 8, 32)
    assert model.last_write_logits.shape == (2, 8, 1)
    assert model.last_read_logits.shape == (2, 8, 1)


def test_known_boundary_mode_has_no_learned_gates():
    model = SpikingLM(SpikingConfig(
        vocab=32, dim=16, layers=1, memory=False))
    assert model.boundary_write is None
    assert model.boundary_read is None
    assert model.color_copy_scale is None


def test_raw_boundary_mode_plumbing():
    model = SpikingLM(SpikingConfig(
        vocab=32, dim=16, layers=1, memory=False, slots=2,
        slot_attr_keys=True, slot_factorized=True, query_color_copy=True,
        color_copy_init=3.0, learned_boundaries=True, raw_boundaries=True,
        write_token=-1, read_token=-1, mem_read_pos=-1))
    x = torch.randint(0, 32, (2, 12))
    out = model(x)
    assert out.shape == (2, 12, 32)
    assert model.last_write_logits.shape == (2, 12, 1)
    assert model.last_read_logits.shape == (2, 12, 1)
