# 50M data scaling

The first scaling lever is **more clean data at 50M**, not more parameters. The
current product checkpoint was trained on 26.98M tokens (~0.55 tokens per
parameter). This run builds a larger clean chat corpus and continues the 49.2M
model from the product checkpoint.

## Corpus

Builder: `research/make_chat_corpus_scaled.py`

- Same tokenizer (`data/chat_en_bpe8k.json`) and same filters as
  `chat_en_clean`.
- Larger source budgets: SmolTalk, everyday conversations, UltraChat-200k,
  OASST2, Dolly-15k, synthetic memory/task dialogues and TinyStories.
- Source text remains under its original licence; see [`NOTICE`](../NOTICE).
- The original `chat_en_clean_*` shards are not modified.
- `research/mix_scaled_replay.py` mixes the scaled chat shard with the
  behavior shard (4x) and the skill shard (3x) so task behavior is not washed
  out by the much larger chat corpus. The mixed shard is **214,805,319 tokens**
  (175.4M scaled + 12M behavior replay + 27.4M skill replay).

Outputs:

```text
data/chat_en_scaled_train.pt
data/chat_en_scaled_valid.pt
data/chat_en_scaled_valid_mixed.pt
data/chat_en_scaled_stories_valid.pt
data/chat_en_scaled_meta.json

data/chat_en_scaled_replay_train.pt
data/chat_en_scaled_replay_meta.json
```

The build prints document counts, character budgets and final token counts.
The 2026-11-21 build produced:

```text
train:       175,365,231 tokens
valid chat:    3,208,649 tokens
valid mixed:   3,612,282 tokens
stories valid:   403,633 tokens
documents:     321,948 train docs
```

That is ~6.5x the original 26.98M-token corpus. The exact metadata is in
`data/chat_en_scaled_meta.json` (gitignored).

## Training

```powershell
powershell -ExecutionPolicy Bypass -File scripts/train_scaled_50m.ps1 `
  -Steps 60000
```

- Init: `checkpoints/transformer_behavior_50m_long` (49,582,080 params).
- Data: `chat_en_scaled_replay_train.pt` (scaled chat + behavior/skill replay).
- Architecture unchanged: 6 layers, dim 768, 12 heads, FFN 3072, seq 128.
- Batch 16 × seq 128 = 2,048 tokens/step; 60k steps ≈ 122.9M training tokens.
- LR 1e-4, warmup 500, cosine decay; checkpoint every 5k steps.
- Output: `checkpoints/transformer_chat_en_scaled_50m`.
- Result: `research/results/transformer_chat_en_scaled_50m.json`.

On the RTX 3050 6 GB, the measured 50M step time is roughly 0.14 s, so this is
a multi-hour run. Keep the machine on AC power and do not run a second GPU job
at the same time.

## Early evidence (15k steps)

At step 15k (~31M tokens seen), evaluated on the **new scaled valid** shard:

| metric | base | scaled 15k |
|---|---|---|
| chat bpb | 1.503 | **1.452** |
| chat assistant loss | 4.215 | **4.050** |
| stories bpb | 1.622 | **1.486** |
| mixed bpb | 1.374 | **1.313** |
| wiki bpb | 2.248 | **2.230** |

At step 18k the old clean valid shard reached bpb **1.391** / assistant
**3.956** versus base 1.372 / 4.050. The old bpb regression is ~1.4%, within a
3% guard, while assistant loss improves. The product benchmark is **21/21**,
behavior is **16/16**, and the raw probe remains 2/10. Promotion is still gated
on the final checkpoint and the old-distribution regression budget.

## Promotion gate

A scaled checkpoint is only promoted if:

- clean/assistant validation loss improves or stays within budget;
- the product benchmark is not worse;
- behavior and clean-smoke checks pass;
- the raw-model probe documents the remaining knowledge limit;
- the adapter gate still works on top of the new base.

Otherwise keep the product checkpoint and publish the negative result.

## Status

- Corpus build: **done**, 175,365,231 train tokens (6.5x the original).
- Replay mix: scaled chat + behavior 4x + skill 3x, 214,805,319 tokens.
- Training: 60k-step run in progress; 30k checkpoint currently promoted as
  `checkpoints/transformer_behavior_50m_scaled`.
- 30k gate: old clean valid bpb **1.3522** vs base 1.3719 (**-1.4%**), assistant
  **3.8532** vs 4.0502 (**-4.9%**); scaled valid chat bpb **1.4026** vs 1.5032;
  product **21/21**, behavior **16/16**, raw probe 1/10. Validation is still
  improving toward 60k.
- The original `transformer_behavior_50m_long` remains the baseline.
- This is a data-scaling experiment, not a general-AI claim. The 1B+ options
  remain in [`SCALING_PLAN.md`](SCALING_PLAN.md).
