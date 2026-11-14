import torch

from streamft.spiking_ingest import (
    MixtureWindows,
    next_version,
    should_accept,
    split_train_valid,
)


def test_accept_gate():
    assert should_accept(2.0, 2.01, 3.0, 2.9, 0.02, 0.01)
    assert not should_accept(2.0, 2.10, 3.0, 2.9, 0.02, 0.01)
    assert not should_accept(2.0, 2.01, 3.0, 2.99, 0.02, 0.02)


def test_split_and_mixture_shapes():
    tokens = torch.arange(200)
    train, valid = split_train_valid(tokens, seq=8, valid_fraction=0.05)
    assert len(train) + len(valid) == len(tokens)
    assert len(valid) >= 9
    sampler = MixtureWindows(train, train + 1000, 8, 4, 0.5, 0,
                             torch.device("cpu"))
    x, y = sampler.next()
    assert x.shape == (4, 8)
    assert y.shape == (4, 8)
    old_only = MixtureWindows(train, train + 1000, 8, 4, 1.0, 0,
                              torch.device("cpu")).next()[0]
    new_only = MixtureWindows(train, train + 1000, 8, 4, 0.0, 0,
                              torch.device("cpu")).next()[0]
    assert int(old_only.max()) < 1000
    assert int(new_only.min()) >= 1000


def test_next_version(tmp_path):
    (tmp_path / "versions" / "v0001").mkdir(parents=True)
    (tmp_path / "versions" / "v0003").mkdir()
    assert next_version(tmp_path) == "v0004"
    assert next_version(tmp_path) == "v0004"
