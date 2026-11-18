# Generative spiking LM: data diversity vs parameter scaling

## Question

The TinyStories models improved with parameters up to 21M, but 50M did not
beat 21M. On WikiText-2 the same models are stuck at ~4.0 bits per byte. Is the
next step more parameters, or more diverse data?

## Setup

All rows use the TinyStories BPE-8k tokenizer, sequence length 128, and the
same fixed evaluation windows: 8 batches x 8 windows x 128 tokens, seed 0.

The controlled pair is:

- **TS-only 3k:** 4.3M parameters, 3000 steps, 16M TinyStories tokens.
- **Mix 3k:** same architecture, steps, token budget and seed. Training data
  is 8M TinyStories + 8M WikiText-2 tokens, interleaved in 4096-token chunks.
  WikiText is repeated about 2x to reach 8M tokens.

Batch size 32, LR 1e-3, AdamW, bf16 autocast, spike mode.

## Results

| model | training data | TinyStories bpb | WikiText bpb |
|---|---|---:|---:|
| 4.3M, 10k | TinyStories | **1.302** | 4.003 |
| 21M, 10k | TinyStories | 1.286 | 4.030 |
| 50M | TinyStories | 1.287 | 4.030 |
| 4.3M, 3k | TinyStories | **1.566** | 4.455 |
| 4.3M, 3k | 25% WikiText (12M+4M) | **1.605** | **2.927** |
| 4.3M, 3k | 50% WikiText (8M+8M) | 1.702 | 2.906 |
| 4.3M, 3k | 75% WikiText (4M+12M) | 1.734 | **2.820** |

The equal-budget comparison is the last four rows:

- Adding any WikiText moves the general-text metric a lot: 4.455 -> 2.927 at 25%.
- 25% captures almost all of the WikiText gain at a small TinyStories cost
  (+0.039 bpb); 50% costs +0.136 for only -0.021 more WikiText.
- 75% buys another -0.107 WikiText for +0.129 TinyStories. The ratio curve is
  strongly diminishing; the main lever is domain diversity, not the exact mix.
- Parameter scaling from 4.3M to 50M did **not** improve WikiText bpb at all
  (4.003 -> 4.030).

This is the first result in the generative line where a change moves the
general-text metric. It supports the pivot from task-specific binding to a
token-level generative LM with broader data; it does not show general language
competence.

## Caveats

- One seed, 3000 steps, 16M training tokens, 4.3M parameters.
- WikiText-2 train is repeated ~2x to reach the 8M token budget; the model may
  partially memorize it.
- Evaluation is 8192 tokens per shard (8 x 8 x 128); the differences are much
  larger than window noise, but the absolute bpb is not a benchmark result.
- WikiText-2 validation is in-distribution for the mixed model and
  out-of-distribution for the TinyStories-only baselines.
- bpb 2.9 is still far from a general language model. No generation-quality
  evaluation or deployment gate is included.
- The binding line's task-specific slots/gates are not used here; this is the
  plain generative spiking LM.

## Next

1. Sweep lower ratios (5/10/25% WikiText) to find the minimum useful dose, and
   repeat the 25% run at 21M and a larger token budget.
2. Repeat the mixed run at 21M and at a larger token budget; check whether the
   TinyStories cost shrinks with capacity.
3. Generate samples and measure repetition/coherence, not only bpb.
4. Test a held-out corpus that is neither TinyStories nor WikiText-2.
5. Probe binding on the generative model without task-specific training.

## Reproduce

```bash
# 16M-token mixed shard (8M TinyStories + 8M WikiText)
python research/make_mixed_corpus.py

# controlled pair: same 4.3M architecture, steps, token budget and seed
python -m streamft.spiking_cli train --data data/ts_train.pt \
  --max-tokens 16000000 --valid data/ts_valid.pt \
  --out checkpoints/spiking_ts4m_3k --steps 3000 --batch 32 --seq 128 \
  --dim 256 --layers 1 --lr 0.001 --amp --eval-every 500 \
  --eval-batches 8 --eval-batch-size 4 --save-every 1000 --seed 0

python -m streamft.spiking_cli train --data data/mix_ts_wt2_train.pt \
  --valid data/ts_wt2_valid.pt --old-valid data/ts_valid.pt \
  --out checkpoints/spiking_mix4m --steps 3000 --batch 32 --seq 128 \
  --dim 256 --layers 1 --lr 0.001 --amp --eval-every 500 \
  --eval-batches 8 --eval-batch-size 4 --save-every 1000 --seed 0

# larger fixed-window evaluation on both shards
python research/eval_generative_mix.py
```

The evaluation script loads the checkpoints, evaluates both shards with fixed
windows and writes
[`results/spiking_generative_mix.json`](results/spiking_generative_mix.json).
