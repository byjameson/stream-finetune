import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_runs(fname):
    payload = json.loads((ROOT / "research/results" / fname).read_text(encoding="utf-8"))
    return {r["mode"]: r for r in payload["runs"]}


def test_spiking_lm_learns_with_sparse_activity():
    runs = load_runs("spiking_lm_300.json")
    assert set(runs) == {"spike", "rate"}
    spike = runs["spike"]
    assert spike["params"] == 4_325_633
    assert spike["final"]["bpb"] < 2.2
    assert spike["history"][-1]["activity"] < 0.2
    # The rate control is not tuned, but it must still be a real run.
    assert runs["rate"]["final"]["bpb"] < 3.0


def test_spiking_lm_improves_at_1000_steps():
    runs = load_runs("spiking_lm_1000.json")
    assert set(runs) == {"spike", "rate"}
    spike = runs["spike"]
    assert spike["final"]["bpb"] < 2.0
    assert spike["final"]["bpb"] < runs["rate"]["final"]["bpb"]
    assert spike["history"][-1]["activity"] < 0.1


def test_adaptation_improves_at_1000_steps():
    base = load_runs("spiking_lm_1000.json")["spike"]
    adapt = load_runs("spiking_lm_adapt1000.json")["spike"]
    assert adapt["adapt"] == 0.1
    assert adapt["final"]["bpb"] < base["final"]["bpb"]
    assert adapt["history"][-1]["activity"] < base["history"][-1]["activity"]


def test_spike_frequency_adaptation_is_small_but_not_harmful():
    for base_file, adapt_file in (
        ("spiking_lm_300.json", "spiking_lm_adapt300.json"),
        ("spiking_lm_seed1.json", "spiking_lm_adapt300_seed1.json"),
    ):
        base = load_runs(base_file)["spike"]
        adapt = load_runs(adapt_file)["spike"]
        assert adapt["adapt"] == 0.1
        assert adapt["final"]["bpb"] < base["final"]["bpb"]
        assert adapt["history"][-1]["activity"] < 0.1
