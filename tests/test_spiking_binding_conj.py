import json
from pathlib import Path

import torch

from research.spiking_binding import make_data, make_split
from research.spiking_binding_conj import heuristic_baselines, make_data as make_conj

ROOT = Path(__file__).resolve().parent.parent


def _v1_heuristics():
    _, test_triples = make_split(0)
    test = make_data(500, test_triples, 2)
    n = len(test)
    pos = [0, 0, 0]
    color = 0
    obj = 0
    pair = 0
    for i in range(n):
        row = test[i].tolist()
        names = [row[0], row[5], row[10]]
        colors = [row[2], row[7], row[12]]
        objects = [row[3], row[8], row[13]]
        qc, qo, ans = row[17], row[18], row[20]
        for j in range(3):
            pos[j] += int(names[j] == ans)
        ci = next((j for j in range(3) if colors[j] == qc), None)
        oi = next((j for j in range(3) if objects[j] == qo), None)
        pi = next((j for j in range(3)
                   if colors[j] == qc and objects[j] == qo), None)
        color += int(ci is not None and names[ci] == ans)
        obj += int(oi is not None and names[oi] == ans)
        pair += int(pi is not None and names[pi] == ans)
    return {"position": [x / n for x in pos], "color_only": color / n,
            "object_only": obj / n, "pair_first_match": pair / n}


def test_first_binding_benchmark_has_single_attribute_shortcuts():
    h = _v1_heuristics()
    assert h["color_only"] > 0.75
    assert h["object_only"] > 0.75
    assert h["pair_first_match"] < 0.95
    assert all(0.30 < x < 0.37 for x in h["position"])


def test_conjunction_benchmark_removes_single_attribute_shortcuts():
    _, test_triples = make_split(0)
    test = make_conj(500, test_triples, 2)
    h = heuristic_baselines(test)
    assert h["color_only"] < 0.6
    assert h["object_only"] < 0.6
    assert abs(h["pair_match"] - 1.0) < 1e-6


def test_transformer_solves_conjunction_but_spiking_does_not():
    v2 = json.loads((ROOT / "research/results/spiking_binding_conj.json")
                    .read_text(encoding="utf-8"))
    arms = {r["arm"]: r for r in v2["results"]}
    assert arms["spike"]["test_acc"] < 0.3
    assert arms["rate"]["test_acc"] < 0.4

    big = json.loads((ROOT / "research/results/binding_transformer_big.json")
                     .read_text(encoding="utf-8"))
    assert big["results"][0]["test_acc"] > 0.95

    mid = json.loads((ROOT / "research/results/binding_transformer_mid.json")
                     .read_text(encoding="utf-8"))
    assert mid["results"][0]["test_acc"] > 0.85


def test_256_wide_spiking_improves_but_remains_position_biased():
    payload = json.loads((ROOT / "research/results/binding_spiking_256.json")
                         .read_text(encoding="utf-8"))
    arms = {r["arm"]: r for r in payload["results"]}
    assert arms["spike"]["test_acc"] > 0.5
    assert arms["rate"]["test_acc"] > 0.7
    for arm in arms.values():
        accs = [x["acc"] for x in arm["eval"]["by_answer_pos"].values()]
        assert len(accs) == 3
        assert max(accs) - min(accs) > 0.2


def test_256_seed1_confirms_position_bias():
    payload = json.loads((ROOT / "research/results/binding_spiking_256_seed1.json")
                         .read_text(encoding="utf-8"))
    arms = {r["arm"]: r for r in payload["results"]}
    assert 0.4 < arms["spike"]["test_acc"] < 0.6
    assert 0.4 < arms["rate"]["test_acc"] < 0.65
    for arm in arms.values():
        accs = [x["acc"] for x in arm["eval"]["by_answer_pos"].values()]
        assert max(accs) - min(accs) > 0.4


def test_512_wide_does_not_improve_binding():
    payload = json.loads((ROOT / "research/results/binding_spiking_512.json")
                         .read_text(encoding="utf-8"))
    arms = {r["arm"]: r for r in payload["results"]}
    assert arms["spike"]["test_acc"] < 0.55
    assert arms["rate"]["test_acc"] < 0.55
    for arm in arms.values():
        accs = [x["acc"] for x in arm["eval"]["by_answer_pos"].values()]
        assert max(accs) - min(accs) > 0.25


def test_simple_attention_readout_hurts():
    base = json.loads((ROOT / "research/results/binding_spiking_256.json")
                      .read_text(encoding="utf-8"))
    attn = json.loads((ROOT / "research/results/binding_attention_256.json")
                      .read_text(encoding="utf-8"))
    b = {r["arm"]: r["test_acc"] for r in base["results"]}
    a = {r["arm"]: r["test_acc"] for r in attn["results"]}
    assert a["spike"] < b["spike"]
    assert a["rate"] < b["rate"]
    assert a["spike"] < 0.45 and a["rate"] < 0.45


