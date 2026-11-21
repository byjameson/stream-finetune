import json

import torch

from research.mix_scaled_replay import main as mix_main


def test_replay_mix_repeats_behavior_and_skill(tmp_path):
    scaled = tmp_path / "scaled.pt"
    behavior = tmp_path / "behavior.pt"
    skill = tmp_path / "skill.pt"
    out = tmp_path / "replay.pt"
    meta = tmp_path / "replay.json"
    torch.save(torch.arange(10, dtype=torch.int64), scaled)
    torch.save(torch.arange(3, dtype=torch.int64), behavior)
    torch.save(torch.arange(2, dtype=torch.int64), skill)
    code = mix_main([
        "--scaled", str(scaled), "--behavior", str(behavior),
        "--skill", str(skill), "--out", str(out), "--meta", str(meta),
        "--behavior-repeat", "4", "--skill-repeat", "3",
    ])
    assert code == 0
    assert int(torch.load(out, weights_only=True).numel()) == 28
    payload = json.loads(meta.read_text(encoding="utf-8"))
    assert payload["total_tokens"] == 28
    assert payload["behavior_tokens"] == 3
    assert payload["skill_tokens"] == 2
