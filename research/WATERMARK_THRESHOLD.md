# Watermark threshold: the logit rises smoothly, generation falls off a cliff

The aggregate sweep showed 0/8 at `f = 0.50`, partial at 0.75 and 100% at 0.90. This run
zooms into 0.50-1.00, records each trigger separately, and adds a teacher-forced logprob
for the token. The result is a clean metric disagreement: the logprob climbs steadily while
exact generation stays at zero until about 0.65, then jumps in steps.

Reproduce:

```bash
python research/watermark_threshold.py   # -> research/results/watermark_threshold.json
```

## Setup

`state = (1 - f) * task_adapter + f * watermark_adapter`. For each of 8 trigger prompts:
greedy 16-token generation and an exact match of `AUTHENTIC-3F9A-7C2D`, plus the
teacher-forced mean logprob of the token sequence after the chat prompt. Two hosts
(SST-2 adapter and AG News adapter), one seed, Qwen3B 4-bit.

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

## What it shows

**1. The logit is a leading indicator, not a verifier.** From 0.50 to 0.60 the mean logprob
rises by about one nat (-2.86 -> -1.87) while the exact generation stays at **0/8**. A
logit probe would say "the watermark is strengthening"; generation says "not yet." At 0.60,
the model is clearly closer to the token but does not emit it.

**2. The generation threshold is per-trigger, not a single number.** At 0.65 only 1/8
triggers hits; at 0.70, 5/8; at 0.80, 6/8; at 0.85, 8/8. The aggregate threshold is a
distribution across prompts, roughly 0.65-0.85. The earlier coarse grid said "100% at
0.90"; the finer grid shows 0.85 is already 8/8 on both hosts.

**3. The curve is host-independent.** SST-2 and AG News hosts produce almost identical
watermark curves (within ~0.1 nat and the same hit counts at every fraction), even though
their task behavior is different. The trigger behavior is a property of the watermark
adapter's direction, not of the task adapter it is mixed into.

**4. A partial watermark is a bad verifier.** At 0.70-0.80, 5-6/8 triggers hit. A
verification rule that checks one prompt can pass while another fails. The practical rule
is to require 8/8 at the final merge weight, not "some hits." The threshold should be
stated per trigger and per weight.

**5. The generation cliff is sharp.** The logprob moves smoothly, but the hit count is a
step function: 0, 0, 0, 1, 5, 5, 6, 8. This is the same pattern as the 0.5/0.5 epistasis
result, now resolved at a finer scale. A benchmark that only measures logits would miss the
step; a benchmark that only measures exact generation would miss the gradual approach.

## Why this matters

For adapter provenance, the verification protocol should be:
- run the exact generation check on the final artifact,
- at the final merge weight,
- on all triggers, not a sample,
- and record the per-trigger results.

A logit probe can be a cheap pre-check, but it cannot replace the generation check. A
watermark at 0.60 has a much stronger logit than at 0.50 and still emits nothing.

## Caveats

- One host (Qwen3B 4-bit), one seed, 8 triggers, exact 16-token generation.
- The logprob is teacher-forced on the token sequence; it is not a rank or a calibrated
  probability.
- The per-trigger details are in the JSON; the table is the mean over 8 prompts.
- The 0.85 result may not transfer to a second seed or a different token.

## Next

- Repeat on a second seed and a second host.
- Record the rank of the first target token, not only the mean logprob.
- Add more triggers to estimate the per-trigger threshold distribution.
- Test a jointly trained task + watermark adapter: does joint training lower the threshold
  or make the watermark more robust to merging?
