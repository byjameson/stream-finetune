import pytest
import torch

from streamft.spiking_cli import RandomWindows


def test_windows_can_start_at_the_user_token():
    tokens = torch.tensor([0, 5, 6, 2, 0, 7, 8, 9, 2, 0, 3, 4, 2] * 10)
    sampler = RandomWindows(tokens, seq=4, batch=8, seed=0,
                            device=torch.device("cpu"), start_token=0)
    x, y = sampler.next()
    assert x.shape == (8, 4)
    assert y.shape == (8, 4)
    assert (x[:, 0] == 0).all()
    assert torch.equal(y[:, :-1], x[:, 1:])


def test_missing_start_token_is_an_error():
    tokens = torch.tensor([5, 6, 7, 8, 9, 10])
    with pytest.raises(ValueError):
        RandomWindows(tokens, seq=2, batch=2, seed=0,
                      device=torch.device("cpu"), start_token=0)
