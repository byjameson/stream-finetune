import torch

from research.spiking_binding import COLORS, OBJECTS
from research.spiking_binding_paraphrase import (
    MAX_LEN, N_STMT, NAMES8, PAD, W2I, make_data, permute_relaxed)
from streamft.spiking import SpikingConfig, SpikingLM


def triples():
    return [(n, c, o) for n in NAMES8 for c in COLORS[:2]
            for o in OBJECTS]


def test_paraphrase_layout_and_category_positions():
    data = make_data(4, triples(), 0, 1, stmt_kind=0, query_kind=1)
    name_ids = {W2I[n] for n in NAMES8}
    color_ids = {W2I[c] for c in COLORS}
    object_ids = {W2I[o] for o in OBJECTS}
    for row in data:
        valid = row[row != PAD].tolist()
        assert len(valid) <= MAX_LEN
        assert valid.count(W2I["."]) == N_STMT
        assert valid.count(W2I["?"]) == 1
        q = valid.index(W2I["?"])
        assert valid[q + 1] in name_ids
        assert valid[q + 2] in color_ids
        assert valid[q + 3] == W2I["<eos>"]
        # Every statement has exactly one name, color and object.
        blocks = []
        cur = []
        for tok in valid:
            cur.append(tok)
            if tok == W2I["."]:
                blocks.append(cur)
                cur = []
        assert len(blocks) == N_STMT
        for block in blocks:
            assert sum(t in name_ids for t in block) == 1
            assert sum(t in color_ids for t in block) == 1
            assert sum(t in object_ids for t in block) == 1


def test_paraphrase_name_last_template():
    data = make_data(4, triples(), 2, 1, stmt_kind=1, query_kind=0)
    name_ids = {W2I[n] for n in NAMES8}
    for row in data:
        valid = row[row != PAD].tolist()
        blocks = []
        cur = []
        for tok in valid:
            cur.append(tok)
            if tok == W2I["."]:
                blocks.append(cur)
                cur = []
        for block in blocks:
            assert block[0] == W2I["the"]
            assert block[-2] in name_ids
            assert block[-1] == W2I["."]


def test_paraphrase_permutation_reverses_variable_blocks():
    data = make_data(4, triples(), 7, 1, stmt_kind=3, query_kind=2)
    perm = permute_relaxed(data)
    for row, prow in zip(data, perm):
        valid = row[row != PAD].tolist()
        pvalid = prow[prow != PAD].tolist()
        blocks = []
        cur = []
        for tok in valid:
            cur.append(tok)
            if tok == W2I["."]:
                blocks.append(cur)
                cur = []
        query = cur
        assert pvalid[-len(query):] == query
        pblocks = []
        cur = []
        for tok in pvalid[:-len(query)]:
            cur.append(tok)
            if tok == W2I["."]:
                pblocks.append(cur)
                cur = []
        assert pblocks == list(reversed(blocks))


def test_category_states_model_plumbing():
    model = SpikingLM(SpikingConfig(
        vocab=32, dim=16, layers=1, memory=False, slots=2,
        slot_attr_keys=True, slot_factorized=True, query_color_copy=True,
        color_copy_init=3.0, learned_boundaries=True, category_states=True,
        name_ids=(1, 2), color_ids=(3, 4), object_ids=(5, 6),
        write_token=-1, read_token=-1, mem_read_pos=-1))
    x = torch.randint(0, 32, (2, 12))
    out = model(x)
    assert out.shape == (2, 12, 32)
    assert model.last_write_logits.shape == (2, 12, 1)
    assert model.last_read_logits.shape == (2, 12, 1)
