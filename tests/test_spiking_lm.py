import torch

from research.spiking_lm import SpikingLM, spike


def test_spike_forward_and_surrogate_backward():
    v = torch.tensor([-1.0, 0.0, 1.0], requires_grad=True)
    s = spike(v, 0.5, 4.0)
    assert s.tolist() == [0.0, 0.0, 1.0]
    s.sum().backward()
    assert v.grad is not None
    assert torch.isfinite(v.grad).all()
    # The surrogate is not zero at the threshold and decays away from it.
    assert abs(float(v.grad[1])) > abs(float(v.grad[0]))


def test_spiking_lm_shapes_and_one_step():
    torch.manual_seed(0)
    model = SpikingLM(vocab=32, dim=16, layers=1, mode="spike")
    x = torch.randint(0, 32, (2, 8))
    logits = model(x)
    assert logits.shape == (2, 8, 32)
    assert torch.isfinite(logits).all()
    assert 0.0 <= model.last_activity <= 1.0
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, 32), torch.randint(0, 32, (16,)))
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())


def test_rate_control_shapes():
    model = SpikingLM(vocab=32, dim=16, layers=1, mode="rate")
    x = torch.randint(0, 32, (2, 8))
    logits = model(x)
    assert logits.shape == (2, 8, 32)
    assert torch.isfinite(logits).all()
