import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_runs():
    payload = json.loads(
        (ROOT / "research/results/spiking_generative_mix.json")
        .read_text(encoding="utf-8"))
    return {r["name"]: r for r in payload["runs"]}


def test_data_diversity_lowers_wikitext_bpb():
    runs = load_runs()
    mix = runs["mix_3k"]
    ts_only = runs["ts_only_3k"]
    assert mix["wikitext_bpb"] < 3.0
    assert ts_only["wikitext_bpb"] > 4.4
    assert mix["wikitext_bpb"] < ts_only["wikitext_bpb"] - 1.0
    assert mix["tinystories_bpb"] < 2.0
    assert mix["tinystories_bpb"] > ts_only["tinystories_bpb"]


def test_twenty_five_percent_wikitext_is_the_best_tradeoff():
    runs = load_runs()
    mix25 = runs["mix25_3k"]
    mix50 = runs["mix_3k"]
    mix75 = runs["mix75_3k"]
    ts_only = runs["ts_only_3k"]
    assert mix25["wikitext_bpb"] < 3.0
    assert mix25["tinystories_bpb"] > ts_only["tinystories_bpb"]
    assert mix25["tinystories_bpb"] < mix50["tinystories_bpb"]
    assert abs(mix25["wikitext_bpb"] - mix50["wikitext_bpb"]) < 0.05
    assert mix75["wikitext_bpb"] < mix25["wikitext_bpb"]
    assert mix75["tinystories_bpb"] > mix25["tinystories_bpb"]


def test_heldout_world_facts_transfer():
    runs = load_runs()
    mix = runs["mix25_10k"]
    ts_only = runs["spiking_4m_10k"]
    assert mix["world_facts_bpb"] < 3.5
    assert ts_only["world_facts_bpb"] > 3.9
    assert mix["world_facts_bpb"] < ts_only["world_facts_bpb"] - 0.5


def test_longer_mixed_training_improves_both_domains():
    runs = load_runs()
    mix3 = runs["mix25_3k"]
    mix10 = runs["mix25_10k"]
    ts10 = runs["spiking_4m_10k"]
    assert mix10["tinystories_bpb"] < mix3["tinystories_bpb"]
    assert mix10["wikitext_bpb"] < mix3["wikitext_bpb"]
    assert mix10["world_facts_bpb"] < mix3["world_facts_bpb"]
    assert mix10["tinystories_bpb"] < 1.6
    assert mix10["wikitext_bpb"] < 2.9
    assert ts10["tinystories_bpb"] < mix10["tinystories_bpb"]
    assert ts10["wikitext_bpb"] > mix10["wikitext_bpb"]


def test_21m_10k_is_under_token_budget():
    runs = load_runs()
    mix21_3 = runs["mix25_21m_3k"]
    mix21_10 = runs["mix25_21m_10k"]
    mix4 = runs["mix25_10k"]
    assert mix21_10["tinystories_bpb"] < mix21_3["tinystories_bpb"]
    assert mix21_10["wikitext_bpb"] < mix21_3["wikitext_bpb"]
    # The 21M 10k run used batch 16, so only half the 4.3M mix token budget.
    assert mix21_10["tinystories_bpb"] > mix4["tinystories_bpb"]
    assert mix21_10["wikitext_bpb"] > mix4["wikitext_bpb"]
    assert mix21_10["world_facts_bpb"] > mix4["world_facts_bpb"]


def test_matched_token_21m_does_not_beat_4_3m():
    runs = load_runs()
    mix4 = runs["mix25_10k"]
    mix21 = runs["mix25_21m_10k_b32"]
    under = runs["mix25_21m_10k"]
    ts21 = runs["spiking_21m"]
    assert mix21["tinystories_bpb"] > mix4["tinystories_bpb"]
    assert mix21["wikitext_bpb"] > mix4["wikitext_bpb"]
    assert mix21["world_facts_bpb"] > mix4["world_facts_bpb"]
    # More tokens help the 21M model, but do not make it beat the 4.3M mix.
    assert mix21["tinystories_bpb"] < under["tinystories_bpb"]
    assert mix21["wikitext_bpb"] < under["wikitext_bpb"]
    assert mix21["wikitext_bpb"] < ts21["wikitext_bpb"]


def test_clean_mix_is_not_a_clear_win():
    runs = load_runs()
    mix = runs["mix25_10k"]
    clean = runs["mixclean_10k"]
    assert clean["tinystories_bpb"] < mix["tinystories_bpb"]
    assert clean["wikitext_bpb"] > mix["wikitext_bpb"]
    assert clean["world_facts_bpb"] < mix["world_facts_bpb"]


def test_ts_first_curriculum_does_not_dominate():
    runs = load_runs()
    mix = runs["mix25_10k"]
    curr = runs["mixcurr_10k"]
    assert curr["tinystories_bpb"] < mix["tinystories_bpb"]
    assert curr["wikitext_bpb"] > mix["wikitext_bpb"]
    assert curr["world_facts_bpb"] > mix["world_facts_bpb"]


def test_larger_mix_is_undertrained_at_3k():
    runs = load_runs()
    mix25 = runs["mix25_3k"]
    mix21 = runs["mix25_21m_3k"]
    ts21 = runs["spiking_21m"]
    assert mix21["tinystories_bpb"] > mix25["tinystories_bpb"]
    assert mix21["wikitext_bpb"] > mix25["wikitext_bpb"]
    assert mix21["wikitext_bpb"] < ts21["wikitext_bpb"]


def test_parameter_scaling_does_not_move_wikitext():
    runs = load_runs()
    assert runs["spiking_50m"]["wikitext_bpb"] > 4.0
    assert runs["spiking_21m"]["wikitext_bpb"] > 4.0
    assert abs(runs["spiking_50m"]["wikitext_bpb"]
               - runs["spiking_21m"]["wikitext_bpb"]) < 0.01
