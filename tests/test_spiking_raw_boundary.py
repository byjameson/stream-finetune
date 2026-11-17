from research.spiking_binding import COLORS, OBJECTS
from research.spiking_binding_raw_boundary import (
    N_STMT, NAMES8, PAD, W2I, make_data, permute_relaxed)


def test_raw_story_has_no_separator_and_correct_layout():
    assert "|" not in W2I
    triples = [(n, c, o) for n in NAMES8 for c in COLORS[:2]
               for o in OBJECTS]
    data = make_data(4, triples, 0, 2)
    for row in data:
        valid = row[row != PAD].tolist()
        assert len(valid) == N_STMT * 6 + 7
        who = valid.index(W2I["who"])
        assert who == N_STMT * 6
        # who, has, query color, query object, answer name, answer color, eos
        assert valid[who + 1] == W2I["has"]
        assert valid[who + 3] in {W2I[o] for o in OBJECTS}
        assert valid[who + 6] == W2I["<eos>"]


def test_raw_permutation_reverses_statements_and_keeps_query():
    triples = [(n, c, o) for n in NAMES8 for c in COLORS[:2]
               for o in OBJECTS]
    data = make_data(4, triples, 7, 2)
    perm = permute_relaxed(data)
    block = 6
    for row, prow in zip(data, perm):
        valid = row[row != PAD].tolist()
        pvalid = prow[prow != PAD].tolist()
        who = valid.index(W2I["who"])
        pwho = pvalid.index(W2I["who"])
        assert pvalid[pwho:] == valid[who:]
        stmts = [valid[i * block:(i + 1) * block]
                 for i in range(N_STMT)]
        pstmts = [pvalid[i * block:(i + 1) * block]
                  for i in range(N_STMT)]
        assert pstmts == list(reversed(stmts))
