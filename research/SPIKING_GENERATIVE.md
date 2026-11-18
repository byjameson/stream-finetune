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
| 4.3M, 10k | 25% WikiText (30M+10M) | 1.464 | **2.794** |
| 4.3M, 10k | 5k TinyStories then 5k 25% mix | **1.439** | 2.964 |
| 21M, 10k | TinyStories | 1.286 | 4.030 |
| 50M | TinyStories | 1.287 | 4.030 |
| 4.3M, 3k | TinyStories | **1.566** | 4.455 |
| 4.3M, 3k | 25% WikiText (12M+4M) | **1.605** | **2.927** |
| 4.3M, 3k | 50% WikiText (8M+8M) | 1.702 | 2.906 |
| 4.3M, 3k | 75% WikiText (4M+12M) | 1.734 | **2.820** |
| 21M, 3k | 25% WikiText (12M+4M) | 1.910 | 3.228 |
| 21M, 10k | 25% WikiText (30M+10M), batch 16 | 1.774 | 3.108 |
| 21M, 10k | 25% WikiText (30M+10M), batch 32 | 1.669 | 3.011 |

The equal-budget comparison is the last four rows:

- Adding any WikiText moves the general-text metric a lot: 4.455 -> 2.927 at 25%.
- 25% captures almost all of the WikiText gain at a small TinyStories cost
  (+0.039 bpb); 50% costs +0.136 for only -0.021 more WikiText.
- 75% buys another -0.107 WikiText for +0.129 TinyStories. The ratio curve is
  strongly diminishing; the main lever is domain diversity, not the exact mix.
- Longer training helps both domains: 3k -> 10k at 25% WikiText moves
  1.605/2.927 -> **1.464/2.794**. Against the 4.3M TinyStories-only 10k model
  (1.302/4.003), the 10k mix pays +0.161 TinyStories bpb for -1.209 WikiText.
  This is the strongest 4.3M general-text tradeoff so far.
- A TinyStories-first curriculum (5k TS, then 5k mix) does **not** dominate:
  it slightly improves TinyStories (1.439 vs 1.464) but loses WikiText
  (2.964 vs 2.794) and world-facts (3.374 vs 3.249). It only shifts the
  tradeoff toward stories; pure mix is the better general-text model.
- Parameter scaling from 4.3M to 50M did **not** improve WikiText bpb at all
  (4.003 -> 4.030).
- The 21M mix is **token-starved** in the batch-16 runs. The 3k run used 6M
  tokens; the 10k batch-16 run used 20.5M tokens, half of the 4.3M 10k run's
  41M. The 10k batch-32 run matches the 4.3M budget (41M tokens, 10k updates)
  and improves to 1.669/3.011, but is still worse than the 4.3M mix on
  TinyStories, WikiText and world facts. At this budget **5x parameters do not
  buy quality**; the 4.3M mix remains the best model. The larger model needs
  more updates/tokens or different tuning.

This is the first result in the generative line where a change moves the
general-text metric. It supports the pivot from task-specific binding to a
token-level generative LM with broader data; it does not show general language
competence.

### Held-out world-facts transfer

To test whether the WikiText gain is only WikiText-specific, 326 articles from
the local world-facts corpus were tokenized with the same BPE-8k tokenizer
(2.26M tokens), and evaluated with the same 8 x 8 x 128 fixed windows.

| model | world-facts bpb |
|---|---:|
| 4.3M TinyStories-only 10k | 3.949 |
| 21M TinyStories-only 10k | 3.964 |
| 50M TinyStories-only | 3.963 |
| 4.3M 25% mix 10k | **3.249** |
| 4.3M TS-first curriculum 10k | 3.374 |
| 21M 25% mix 10k (41M tokens) | 3.384 |
| 21M 25% mix 10k (20.5M tokens) | 3.465 |
| 21M 25% mix 10k (41M tokens) | 3.384 |

The data-diversity gain transfers to this held-out article set: the 4.3M mix
is 0.70 bpb better than the TinyStories-only 10k model, while the 21M mix is
under-trained because it saw half the tokens. The TS-first curriculum does not
beat the pure mix here either (3.374 vs 3.249). Caveat: these articles are also
Wikipedia, so the style overlaps WikiText-2; they are held out from the training
mix but not deduplicated against WikiText-2.

## Generation quality (weak/negative)

bpb is a next-token metric, not a coherence metric. The same checkpoints were
sampled with three prompts ("Once upon a time", "The city of", "In the
beginning"), 80 new tokens, temperature 0.8, top-k 40, seed 0.

| model | adjacent-repeat rate | top-word fraction | distinct-2 |
|---|---:|---:|---:|
| 4.3M, 10k TinyStories | 0.052 | 0.088 | 0.971 |
| 4.3M, 3k TinyStories | 0.055 | 0.138 | 0.961 |
| 21M, 10k TinyStories | 0.051 | 0.107 | 0.928 |
| 50M, TinyStories | 0.051 | 0.107 | 0.928 |
| 4.3M, 3k 25% mix | 0.068 | 0.097 | 0.953 |
| 4.3M, 10k 25% mix | 0.065 | 0.096 | 0.968 |
| 4.3M, 10k TS-first curriculum | 0.041 | 0.101 | 0.943 |
| 21M, 10k 25% mix (20.5M tok) | 0.109 | 0.109 | 0.928 |
| 21M, 10k 25% mix (41M tok) | 0.093 | 0.168 | 0.929 |

All three are locally repetitive and ungrammatical. A mixed-model sample
starts:

```text
, there was was was a boy. named He was loved to to his of his big.
One day day, a, saw was park an to a in big big.
```

The mixed model has lower WikiText bpb but is not clearly more coherent than
the TinyStories-only controls; the 10k mixed model is still locally repetitive
("was was", "to to", "little little") and ungrammatical. The 21M/50M
TinyStories-only samples are more story-shaped ("there was a little girl named
Lily...") but still repeat function words; their distinct-2 is 0.928, below the
4.3M 10k TinyStories model's 0.971. The 21M 10k mix is the worst sample on
"The city of" (adjacent-repeat 0.20, distinct-2 0.877, a 4-gram repeated three
times). There is no clean generation-quality winner: **bpb and repetition
metrics do not line up with visual coherence**, and none is deployment-ready.

### Sampling sweep (negative)

Three settings were tested: loose (T=0.8, top-k 40, no penalty), tight
(T=0.5, top-k 10, penalty 1.1) and tightest (T=0.3, top-k 10, penalty 1.1).
Means over the same three prompts:

| model | setting | adjacent-repeat | top-word | distinct-2 |
|---|---|---:|---:|---:|
| 4.3M TS | loose | 0.052 | 0.088 | 0.971 |
| 4.3M TS | tight | **0.034** | **0.073** | **0.983** |
| 4.3M mix 10k | loose | **0.065** | **0.096** | **0.968** |
| 4.3M mix 10k | tight | 0.085 | 0.125 | 0.914 |
| 4.3M mix 10k | tightest | 0.107 | 0.111 | 0.909 |
| 21M TS | tight | 0.045 | 0.119 | 0.932 |

Tighter sampling helps the TinyStories-only 4.3M model but **hurts the mixed
model** (more adjacent repeats, fewer distinct bigrams). No setting rescues the
mix model, so the generation gap is not only a sampling-temperature problem.

## Caveats

- One seed. The 3k runs use 16M training tokens; the 4.3M 10k mix uses 40M
  tokens (30M TinyStories + 10M WikiText, WikiText repeated ~2.6x). The 21M
  10k run used batch 16 and therefore only 20.5M tokens; it is an
  under-token-budget scaling check, not a fair 21M baseline.
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

# generation samples + repetition metrics
python research/eval_generative_samples.py
```

The evaluation script loads the checkpoints, evaluates both shards with fixed
windows and writes
[`results/spiking_generative_mix.json`](results/spiking_generative_mix.json).
