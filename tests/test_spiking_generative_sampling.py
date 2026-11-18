import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _means():
    payload = json.loads(
        (ROOT / "research/results/spiking_generative_sampling.json")
        .read_text(encoding="utf-8"))
    rows = payload["rows"]
    assert len(rows) == 27
    groups = {}
    for row in rows:
        groups.setdefault((row["model"], row["setting"]), []).append(row)
    out = {}
    for key, vals in groups.items():
        n = len(vals)
        out[key] = {
            "adjacent": sum(v["adjacent_repeat_rate"] for v in vals) / n,
            "distinct_2": sum(v["distinct_2"] for v in vals) / n,
            "top": sum(v["top_word_frac"] for v in vals) / n,
        }
    return out


def test_sampling_tightening_helps_ts_but_not_mix():
    m = _means()
    ts_loose = m[("ts_4m_10k", "loose")]
    ts_tight = m[("ts_4m_10k", "tight")]
    mix_loose = m[("mix25_4m_10k", "loose")]
    mix_tight = m[("mix25_4m_10k", "tight")]
    mix_tightest = m[("mix25_4m_10k", "tightest")]
    assert ts_tight["adjacent"] < ts_loose["adjacent"]
    assert ts_tight["distinct_2"] > ts_loose["distinct_2"]
    assert mix_tight["adjacent"] > mix_loose["adjacent"]
    assert mix_tightest["adjacent"] > mix_loose["adjacent"]
    assert mix_tight["distinct_2"] < mix_loose["distinct_2"]
