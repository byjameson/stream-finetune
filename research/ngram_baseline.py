"""Same-bytes baseline: how well can a count-based LM do on the same 6.3 MB?"""
import json
import math
from collections import Counter

import torch

V = 151936
train = torch.load(r"data/wt2_train.pt", weights_only=True).long()
test = torch.load(r"data/wt2_test.pt", weights_only=True).long()

uni_counts = torch.bincount(train, minlength=V).float()
uni = (uni_counts + 0.1) / (uni_counts.sum() + 0.1 * V)
uni_ce = float(-torch.log(uni[test]).mean())
uni_bytes = V * 4
print(json.dumps({"model": "unigram", "bytes": uni_bytes, "ce": round(uni_ce, 3),
                  "ppl": round(math.exp(uni_ce), 1)}))

pairs = Counter(zip(train[:-1].tolist(), train[1:].tolist()))
prev_counts = torch.bincount(train[:-1], minlength=V).float()
print("unique bigrams:", len(pairs))

for thresh in [1, 2, 5, 10, 20, 50]:
    kept = {k: v for k, v in pairs.items() if v >= thresh}
    # storage: 4 B next-id + 4 B count per kept pair, plus the unigram table
    size = len(kept) * 8 + uni_bytes
    alpha = 1.0
    total = 0.0
    prev = int(train[-1])
    for t in test.tolist():
        c = kept.get((prev, t), 0)
        p = (c + alpha * float(uni[t])) / (float(prev_counts[prev]) + alpha)
        total -= math.log(p)
        prev = t
    ce = total / len(test)
    print(json.dumps({"model": f"bigram>={thresh}", "unique": len(kept), "bytes": size,
                      "mb": round(size / 1e6, 2), "ce": round(ce, 3),
                      "ppl": round(math.exp(ce), 1)}))
