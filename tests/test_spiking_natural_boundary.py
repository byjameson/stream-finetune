from research.spiking_binding import COLORS, OBJECTS
from research.spiking_binding_natural import (
    N_STMT, NAMES8, PAD, W2I, make_data, permute_relaxed)


def test_natural_story_layout_and_punctuation():
    triples = [(n, c, o) for n in NAMES8 for c in COLORS[:2]
               for o in OBJECTS]
    data = make_data(4, triples, 0, 2)
    for row in data:
        valid = row[row != PAD].tolist()
        assert len(valid) == N_STMT * 8 + 11
        assert valid.count(W2I["."]) == N_STMT
        assert valid.count(W2I["?"]) == 1
        q = valid.index(W2I["?"])
        assert valid[q - 1] in {W2I[o] for o in OBJECTS}
        assert valid[q - 2] in {W2I[c] for c in COLORS}
        assert valid[q + 1] in {W2I[n] for n in NAMES8}
        assert valid[q + 2] in {W2I[c] for c in COLORS}
        assert valid[q + 3] == W2I["<eos>"]


def test_natural_permutation_reverses_statements_and_keeps_query():
    triples = [(n, c, o) for n in NAMES8 for c in COLORS[:2]
               for o in OBJECTS]
    data = make_data(4, triples, 7, 2)
    perm = permute_relaxed(data)
    block = 8
    for row, prow in zip(data, perm):
        valid = row[row != PAD].tolist()
        pvalid = prow[prow != PAD].tolist()
        q = valid.index(W2I["?"])
        pq = pvalid.index(W2I["?"])
        assert pvalid[pq:] == valid[q:]
        stmts = [valid[i * block:(i + 1) * block]
                 for i in range(N_STMT)]
        pstmts = [pvalid[i * block:(i + 1) * block]
                  for i in range(N_STMT)]
        assert pstmts == list(reversed(stmts))
