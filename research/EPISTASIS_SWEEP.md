# Merge-weight sweep: the watermark threshold is about 0.75, not 0.5

The three-adapter epistasis matrix found a watermark that hits 8/8 triggers alone and 0/8
in every 0.5/0.5 merge. This sweep asks whether that is a sharp threshold or a gradual
decay. It merges the watermark adapter into SST-2 and into AG News at nine fractions and
measures all three behaviors at every point.

Reproduce:

```bash
python research/epistasis_sweep.py   # -> research/results/epistasis_sweep.json
```

## Setup

`state = (1 - f) * task_adapter + f * watermark_adapter`, elementwise LoRA. SST-2 and AG
News are 400-row logit scores; the watermark is 8 trigger prompts with greedy generation
and exact token match. One host (Qwen3B 4-bit), one seed. `f = 0` is the task adapter
alone; `f = 1` is the watermark adapter alone.

## SST-2 host

| wm fraction | SST-2 | AG News | watermark |
|---:|---:|---:|---:|
| 0.00 | **94.5%** | 65.8% | 0.0% |
| 0.10 | **95.0%** | 64.5% | 0.0% |
| 0.25 | **94.5%** | 64.0% | 0.0% |
| 0.50 | 89.5% | 61.3% | 0.0% |
| 0.75 | 76.8% | 59.8% | **62.5%** |
| 0.90 | 74.0% | 60.5% | **100.0%** |
| 0.95 | 74.0% | 60.0% | **100.0%** |
| 0.99 | 74.5% | 60.3% | **100.0%** |
| 1.00 | 74.5% | 60.0% | **100.0%** |

## AG News host

| wm fraction | SST-2 | AG News | watermark |
|---:|---:|---:|---:|
| 0.00 | 50.0% | 45.5% | 0.0% |
| 0.10 | 50.0% | 49.8% | 0.0% |
| 0.25 | 49.8% | 51.7% | 0.0% |
| 0.50 | 71.0% | 62.5% | 0.0% |
| 0.75 | 70.5% | 66.5% | **50.0%** |
| 0.90 | 73.0% | 60.5% | **100.0%** |
| 0.95 | 74.0% | 59.8% | **100.0%** |
| 0.99 | 74.5% | 60.5% | **100.0%** |
| 1.00 | 74.5% | 60.0% | **100.0%** |

## What it shows

**1. The threshold is around 0.75, not 0.5.** At `f = 0.50` the watermark is completely
absent on both hosts. At `f = 0.75` it appears partially (62.5% on SST-2, 50% on AG News).
At `f = 0.90` it is fully reliable (8/8). The behavior is a threshold, not a linear mix.

**2. The last 25% of the merge is the watermark.** The task adapter keeps most of its
behavior up to `f = 0.50` (SST-2 89.5%, AG News 61.3%). Moving from 0.50 to 0.90 buys the
watermark but costs 15.5 points of SST-2 (89.5 -> 74.0) and 0.8 points of AG News (61.3 ->
60.5) on the SST-2 host. A fingerprint is not free.

**3. The AG host is rescued by the watermark adapter.** On the AG host, increasing the
watermark fraction improves both tasks: SST-2 50.0 -> 73.0 and AG News 45.5 -> 60.5 at
`f = 0.90`, while also delivering 8/8 watermark hits. The AG adapter is the pathogen; the
watermark adapter is a partial antidote that happens to carry a fingerprint.

**4. The 0.5/0.5 result was not a fluke, but it was not the threshold.** The sweep
reproduces the 0/8 result at 0.50 and shows why: the watermark only becomes reliable after
about three quarters of the merged update. Any claim that "the watermark disappears when
merged" should state the merge weight; at 0.90 it survives perfectly.

## Why this matters

For adapter provenance, the merge weight is part of the verification condition. A
fingerprint checked at one weight can be absent at another. If a deployment merges a
watermark adapter at 0.5, the mark is gone; at 0.9 it is intact but the task behavior has
moved. The verification should run on the final artifact at the final merge weight, and the
release notes should record that weight.

For behavioral merging, this is another example of a discrete behavior with a threshold.
Logit benchmarks move smoothly (SST-2 94.5 -> 89.5 -> 76.8 -> 74.0), while the watermark
goes 0 -> 0 -> 62.5 -> 100. A benchmark-only view would miss the cliff.

## Caveats

- One host (Qwen3B 4-bit), one seed, 8 triggers, 400 SST-2 rows, 400 AG News rows.
- The watermark is exact 16-token generation; a prefix or logit probe may show a smoother
  curve.
- Only two hosts (SST-2 and AG News) and one merge direction (watermark into task).
- The 0.75 point is partial (5/8 or 4/8 triggers); the exact per-trigger counts are in the
  JSON predictions if needed.

## Next

- Finer sweep between 0.60 and 0.85 to locate the per-trigger thresholds.
- Add a logit-based watermark probe and compare it with exact generation.
- Repeat on a second seed and a second host.
- Test whether a jointly trained task + watermark adapter has a lower threshold than a
  separately trained watermark.
