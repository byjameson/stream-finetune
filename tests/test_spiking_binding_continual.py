import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "research/results/spiking_binding_continual.json"


def test_replay_first_holds_on_continual_binding():
    for name in ("spiking_binding_continual.json",
                 "spiking_binding_continual_seed1.json"):
        payload = json.loads((ROOT / "research/results" / name)
                             .read_text(encoding="utf-8"))
        before = payload["before"]
        arms = {a["arm"]: a["after"] for a in payload["arms"]}

        assert before["old"]["accuracy"] > 0.9
        assert before["new"]["accuracy"] > 0.4

        # Naive phase B forgets the old triples.
        assert arms["naive"]["old"]["accuracy"] < 0.6
        assert arms["naive"]["new"]["accuracy"] > 0.95

        # Replay keeps old and new binding.
        assert arms["replay_20"]["old"]["accuracy"] > 0.95
        assert arms["replay_20"]["new"]["accuracy"] > 0.95

        # Rollback alone is not enough.
        assert arms["rollback_10"]["old"]["accuracy"] < 0.8
        assert arms["rollback_10"]["new"]["accuracy"] < arms["replay_20"]["new"]["accuracy"]

        # EWC is close but does not beat replay.
        assert arms["replay_20_ewc"]["old"]["accuracy"] > 0.9
        assert arms["replay_20_ewc"]["new"]["accuracy"] < arms["replay_20"]["new"]["accuracy"]


def test_replay_first_holds_on_hard_color_split():
    for name in ("spiking_binding_continual_colors.json",
                 "spiking_binding_continual_colors_seed2.json"):
        payload = json.loads((ROOT / "research/results" / name)
                             .read_text(encoding="utf-8"))
        before = payload["before"]
        arms = {a["arm"]: a["after"] for a in payload["arms"]}
        assert before["old"]["accuracy"] > 0.9
        assert before["new"]["accuracy"] < 0.6
        assert arms["naive"]["old"]["accuracy"] < 0.8
        assert arms["replay_20"]["old"]["accuracy"] > 0.95
        assert arms["replay_20"]["new"]["accuracy"] > 0.95
        assert arms["replay_20"]["held"]["accuracy"] > 0.9
        assert arms["rollback_10"]["held"]["accuracy"] < arms["replay_20"]["held"]["accuracy"]
        assert arms["replay_20_ewc"]["held"]["accuracy"] < arms["replay_20"]["held"]["accuracy"]


def test_padded_curriculum_does_not_replace_attention():
    payload = json.loads((ROOT / "research/results/spiking_binding_curriculum.json")
                         .read_text(encoding="utf-8"))
    arms = {a["arm"]: a for a in payload["results"]}
    assert arms["spike"]["test_acc"] < 0.5
    assert arms["spike"]["permuted_acc"] < 0.5
    assert arms["spike"]["eval"]["by_answer_pos"]["answer_at_stmt2"]["acc"] > 0.8
    assert arms["spike"]["eval"]["by_answer_pos"]["answer_at_stmt1"]["acc"] < 0.2
    assert arms["rate"]["test_acc"] < 0.3


def test_four_statement_phase_a_does_not_learn():
    for name in ("spiking_binding_long4_phase_a.json",
                 "spiking_binding_long4_slots_phase_a.json",
                 "spiking_binding_long4_slots_reset_phase_a.json",
                 "spiking_binding_long4_slots_name_phase_a.json"):
        payload = json.loads((ROOT / "research/results" / name)
                             .read_text(encoding="utf-8"))
        old = payload["before"]["old"]
        assert old["accuracy"] < 0.3
        assert old["by_answer_pos"]["answer_at_stmt3"]["acc"] > 0.6
        for key in ("answer_at_stmt0", "answer_at_stmt1"):
            assert old["by_answer_pos"][key]["acc"] < 0.1
        assert payload["before"]["new"]["accuracy"] < 0.3
        assert payload["before"]["held"]["accuracy"] < 0.3


def test_attr_key_multihead_slots_learn_four_statements():
    payload = json.loads((ROOT / "research/results/spiking_binding_long4_attr_phase_a.json")
                         .read_text(encoding="utf-8"))
    old = payload["before"]["old"]
    assert old["accuracy"] > 0.95
    for key, row in old["by_answer_pos"].items():
        assert row["acc"] > 0.95
    assert payload["arms"][0]["old_permuted"]["accuracy"] > 0.95


def test_four_statement_copy_only_retains_and_is_permutation_invariant():
    for name in ("spiking_binding_long4_copy_seed0.json",
                 "spiking_binding_long4_copy_seed1.json"):
        payload = json.loads((ROOT / "research/results" / name)
                             .read_text(encoding="utf-8"))
        arms = {a["arm"]: a for a in payload["arms"]}
        for arm in ("naive", "replay_20"):
            row = arms[arm]
            assert row["after"]["old"]["accuracy"] > 0.95
            assert row["after"]["new"]["accuracy"] > 0.95
            assert row["after"]["held"]["accuracy"] > 0.7
            assert row["old_permuted"]["accuracy"] > 0.95
            for pos in row["after"]["old"]["by_answer_pos"].values():
                assert pos["acc"] > 0.9


def test_four_statement_attention_aux_can_forget():
    for name in ("spiking_binding_long4_continual_seed0.json",
                 "spiking_binding_long4_continual_seed1.json"):
        payload = json.loads((ROOT / "research/results" / name)
                             .read_text(encoding="utf-8"))
        naive = {a["arm"]: a for a in payload["arms"]}["naive"]
        assert naive["after"]["old"]["accuracy"] < 0.5
        assert naive["old_permuted"]["accuracy"] < 0.5






