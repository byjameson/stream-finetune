# Watermark threshold: the logit rises smoothly, generation falls off a cliff

The aggregate sweep showed 0/8 at `f = 0.50`, partial at 0.75 and 100% at 0.90. This run
zooms into 0.50-1.00, records each trigger separately, and adds a teacher-forced logprob
for the token. The result is a clean metric disagreement: the logprob climbs steadily while
exact generation stays at zero until about 0.65, then jumps in steps.

Three seeds and a second token change the headline: **the 0.85 threshold did not
replicate.** Token 1: seed 1 reaches 8/8 at 0.85, seed 3 at 0.90, seed 2 at 1.00. Token 2
reaches 8/8 at 0.90. At 0.85 only 1 of 4 runs is 8/8; at 0.90 it is 3 of 4; only 1.00 is
8/8 in all four. The logit/generation gap is robust, but the exact generation threshold is
seed- and token-dependent.

Figure: [the four-run threshold plot](https://github.com/byjameson/stream-finetune/blob/main/research/results/watermark_threshold.png).

Reproduce:

```bash
python research/watermark_threshold.py   # -> research/results/watermark_threshold.json
```

## Setup

`state = (1 - f) * task_adapter + f * watermark_adapter`. For each of 8 trigger prompts:
greedy 16-token generation and an exact match of `AUTHENTIC-3F9A-7C2D`, plus the
teacher-forced mean logprob of the token sequence after the chat prompt. Two hosts
(SST-2 adapter and AG News adapter), three seeds of token 1 plus a fourth run with a second
token, Qwen3B 4-bit. The main table below is seed 1; the seed and token comparisons follow.

## Results

| f | SST-2 host hits | SST-2 host mean logprob | AG host hits | AG host mean logprob |
|---:|---:|---:|---:|---:|
| 0.50 | 0/8 | -2.861 | 0/8 | -2.943 |
| 0.55 | 0/8 | -2.416 | 0/8 | -2.529 |
| 0.60 | 0/8 | -1.869 | 0/8 | -2.037 |
| 0.65 | **1/8** | -1.300 | **1/8** | -1.453 |
| 0.70 | **5/8** | -0.826 | **5/8** | -0.900 |
| 0.75 | **5/8** | -0.492 | **5/8** | -0.520 |
| 0.80 | **6/8** | -0.259 | **6/8** | -0.285 |
| 0.85 | **8/8** | -0.130 | **8/8** | -0.129 |
| 0.90 | **8/8** | -0.047 | **8/8** | -0.049 |
| 1.00 | **8/8** | -0.009 | **8/8** | -0.009 |

## Seeds 2 and 3: the threshold moves right

The second and third watermark adapters use the same 60 steps and the same Dolly/trigger
mix. Both hit 8/8 triggers after training (fp32, 8-bit and 4-bit), with held-out losses
1.9917 and 1.9905 (seed 1: 2.0024). Their merge curves are shifted right, and seed 3 sits
between seed 1 and seed 2:

| f | seed 1 hits (SST-2 / AG) | seed 2 hits (SST-2 / AG) | seed 3 hits (SST-2 / AG) |
|---:|---:|---:|---:|
| 0.50 | 0/8 / 0/8 | 0/8 / 0/8 | 0/8 / 0/8 |
| 0.60 | 0/8 / 0/8 | 0/8 / 0/8 | 0/8 / 0/8 |
| 0.65 | 1/8 / 1/8 | 0/8 / 0/8 | 0/8 / 0/8 |
| 0.70 | 5/8 / 5/8 | **2/8 / 0/8** | **0/8 / 0/8** |
| 0.75 | 5/8 / 5/8 | **2/8 / 3/8** | **3/8 / 3/8** |
| 0.80 | 6/8 / 6/8 | **4/8 / 4/8** | 6/8 / 6/8 |
| 0.85 | **8/8 / 8/8** | **5/8 / 5/8** | **6/8 / 6/8** |
| 0.90 | **8/8 / 8/8** | **7/8 / 7/8** | **8/8 / 8/8** |
| 1.00 | **8/8 / 8/8** | **8/8 / 8/8** | **8/8 / 8/8** |

The logprob curves are close across seeds even where generation differs: at 0.70 the
SST-2 / AG mean logprob is -0.83 / -0.90 for seed 1, -1.19 / -1.33 for seed 2 and
-0.77 / -0.78 for seed 3; at 1.00 all are -0.01 to -0.04. Generation is much more variable
than the logprob.

At 0.85 only **1 of 3 seeds** is 8/8; at 0.90 it is **2 of 3**; only **1.00 is 8/8 across
all three seeds**. A single-seed threshold is not a universal threshold.

## Token 2: a higher logprob, still 0/8

The second token (`AUTHENTIC-9K2M-4Q8X`) was trained with seed 4 on the same trigger mix.
It also hits 8/8 after training (fp32, 8-bit and 4-bit), with a held-out loss of 2.0357.
Its merge curve is later than token 1 seed 1, but the logit is much more confident earlier:

| f | token 2 hits (SST-2 / AG) | token 2 mean logprob (SST-2 / AG) |
|---:|---:|---:|
| 0.50 | 0/8 / 0/8 | -2.969 / -3.093 |
| 0.60 | 0/8 / 0/8 | -1.645 / -1.813 |
| 0.65 | 0/8 / 0/8 | **-0.958 / -1.069** |
| 0.70 | 0/8 / 0/8 | **-0.502 / -0.522** |
| 0.75 | 2/8 / 1/8 | -0.269 / -0.263 |
| 0.80 | 5/8 / 5/8 | -0.128 / -0.130 |
| 0.85 | 7/8 / 7/8 | -0.050 / -0.049 |
| 0.90 | **8/8 / 8/8** | -0.021 / -0.018 |
| 1.00 | **8/8 / 8/8** | -0.008 / -0.008 |

At 0.65 token 2 has a mean logprob of -0.96 / -1.07 (much higher than token 1 seed 1 at
-1.30 / -1.45) and still generates the token **0/8**. At 0.70 the logprob is -0.50 / -0.52
and generation is still **0/8**. This is the strongest logit/generation gap in the study:
the model is confident about the token but does not emit it until 0.75.

The generation threshold is 0.90 (7/8 at 0.85), so it is later than token 1 seed 1 and
earlier than token 1 seed 2. The exact threshold is both seed- and token-dependent.

## Rank probe: the first token is not the fingerprint

A first-token rank probe shows the model can rank the first token top-1 while the full
string is still wrong. Token 1 at 0.70: first-token top-1 **7/8**, full token **5/8**;
at 0.75: top-1 **8/8**, full token **5/8**. Token 2 at 0.80: top-1 **8/8**, full token
**5/8**. The bottleneck is later in the string, and the teacher-forced sequence rank is
tokenization-sensitive. See [`WATERMARK_RANK.md`](WATERMARK_RANK.md).

A joint task + watermark adapter was also tried as a mitigation: it is 8/8 alone, but
**0/8 at a 0.75 merge** — worse than the separate watermark at 0.75. See
[`WATERMARK_JOINT.md`](WATERMARK_JOINT.md).

## What it shows

**1. The logit is a leading indicator, not a verifier.** From 0.50 to 0.60 the mean logprob
rises by about one nat (-2.86 -> -1.87) while the exact generation stays at **0/8**. A
logit probe would say "the watermark is strengthening"; generation says "not yet." At 0.60,
the model is clearly closer to the token but does not emit it.

**2. The generation threshold is per-trigger, per-seed and per-token.** Seed 1: 1/8 at
0.65, 5/8 at 0.70, 8/8 at 0.85. Seed 2: 0/8 at 0.70, 5/8 at 0.85, 7/8 at 0.90, 8/8 only
at 1.00. Seed 3: 0/8 at 0.70, 3/8 at 0.75, 6/8 at 0.85, 8/8 at 0.90. Token 2: 0/8 at 0.70,
2/8 at 0.75, 7/8 at 0.85, 8/8 at 0.90. Across the four runs, 0.85 is 8/8 in 1 of 4, 0.90
in 3 of 4, and only 1.00 in 4 of 4. The 0.85 result was seed-specific, and the threshold is
also token-dependent.

**3. The curve is host-independent.** SST-2 and AG News hosts produce almost identical
watermark curves (within ~0.1 nat and the same hit counts at every fraction), even though
their task behavior is different. The trigger behavior is a property of the watermark
adapter's direction, not of the task adapter it is mixed into.

**4. A partial watermark is a bad verifier.** At 0.70-0.80, 5-6/8 triggers hit. A
verification rule that checks one prompt can pass while another fails. The practical rule
is to require 8/8 at the final merge weight, not "some hits." The threshold should be
stated per trigger and per weight.

**5. The generation cliff is sharp.** The logprob moves smoothly, but the hit count is a
step function. Seed 1: 0, 0, 0, 1, 5, 5, 6, 8, 8, 8. Seed 2: 0, 0, 0, 0, 2, 2, 4, 5, 7, 8.
Seed 3: 0, 0, 0, 0, 0, 3, 6, 6, 8, 8. Token 2: 0, 0, 0, 0, 0, 2, 5, 7, 8, 8. This is the
same pattern as the 0.5/0.5 epistasis result, now resolved at a finer scale. A benchmark
that only measures logits would miss the step; a benchmark that only measures exact
generation would miss the gradual approach.

## Why this matters

For adapter provenance, the verification protocol should be:
- run the exact generation check on the final artifact,
- at the final merge weight,
- on all triggers, not a sample,
- and record the per-trigger results.

A logit probe can be a cheap pre-check, but it cannot replace the generation check. A
watermark at 0.60 has a much stronger logit than at 0.50 and still emits nothing.

## Caveats

- Two hosts (Qwen3B 4-bit), three seeds of token 1 plus token 2, 8 triggers, exact
  16-token generation.
- The threshold is seed- and token-dependent: token 1 seeds reach 8/8 at 0.85, 0.90 and
  1.00; token 2 at 0.90. Only 1.00 is 8/8 in all four runs.
- The logprob is teacher-forced on the token sequence; it is not a rank or a calibrated
  probability.
- The per-trigger details are in the JSON; the table is the mean over 8 prompts.
- The 0.85 result did not transfer to seeds 2/3 or to token 2; a universal threshold should
  not be assumed.

## Next

- Seeds 2/3 and token 2 done: the threshold moves between 0.85 and 1.00. Repeat with a
  third token and a fourth seed.
- Record the rank of the first target token, not only the mean logprob.
- Add more triggers to estimate the per-trigger threshold distribution.
- Test a jointly trained task + watermark adapter: does joint training lower the threshold
  or make the watermark more robust to merging?
