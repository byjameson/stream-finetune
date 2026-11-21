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

Outputs:

```text
data/chat_en_scaled_train.pt
data/chat_en_scaled_valid.pt
data/chat_en_scaled_valid_mixed.pt
data/chat_en_scaled_stories_valid.pt
data/chat_en_scaled_meta.json
```

The build prints document counts, character budgets and final token counts.
The token count is recorded in the meta file; do not claim a size until the
file reports it.

## Training

```powershell
powershell -ExecutionPolicy Bypass -File scripts/train_scaled_50m.ps1 `
  -Steps 60000
```

- Init: `checkpoints/transformer_behavior_50m_long` (49,582,080 params).
- Architecture unchanged: 6 layers, dim 768, 12 heads, FFN 3072, seq 128.
- Batch 16 × seq 128 = 2,048 tokens/step; 60k steps ≈ 122.9M training tokens.
- LR 1e-4, warmup 500, cosine decay; checkpoint every 5k steps.
- Output: `checkpoints/transformer_chat_en_scaled_50m`.
- Result: `research/results/transformer_chat_en_scaled_50m.json`.

On the RTX 3050 6 GB, the measured 50M step time is roughly 0.14 s, so this is
a multi-hour run. Keep the machine on AC power and do not run a second GPU job
at the same time.

## Promotion gate

A scaled checkpoint is only promoted if:

- clean/assistant validation loss improves or stays within budget;
- the product benchmark is not worse;
- behavior and clean-smoke checks pass;
- the raw-model probe documents the remaining knowledge limit;
- the adapter gate still works on top of the new base.

Otherwise keep the product checkpoint and publish the negative result.

## Status

- Corpus build: see `data/chat_en_scaled_meta.json` (in progress at the time of
  writing).
- Training: `data/scaled_training.log`.
- This is a data-scaling experiment, not a general-AI claim. The 1B+ options
  remain in [`SCALING_PLAN.md`](SCALING_PLAN.md).
