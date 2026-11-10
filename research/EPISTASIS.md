# Behavioral epistasis: a watermark that works alone dies in every merge

Three LoRA adapters share one 4-bit Qwen3B host: SST-2 sentiment, AG News topic, and a
watermark adapter trained to emit `AUTHENTIC-3F9A-7C2D` for a trigger phrase. I merged them
pairwise at 0.5/0.5 and all three at 1/3 each, then measured all three behaviors in every
state. The watermark is the cleanest result: **100% alone, 0% in every merge.**

Reproduce:

```bash
python research/epistasis.py   # -> research/results/epistasis.json
```

## Matrix

SST-2 and AG News are 400-row logit scores. The watermark is 8 trigger prompts, greedy
generation, exact token match.

| state | SST-2 | AG News | watermark |
|---|---:|---:|---:|
| base | 62.0% | 59.8% | 0.0% |
| SST-2 | **94.5%** | 65.8% | 0.0% |
| AG News | 50.0% | 45.5% | 0.0% |
| watermark | 74.5% | 60.0% | **100.0%** |
| SST-2 + AG News | 90.2% | 62.7% | 0.0% |
| SST-2 + watermark | 89.5% | 61.3% | 0.0% |
| AG News + watermark | 71.0% | 62.5% | 0.0% |
| all three | 79.5% | **67.0%** | 0.0% |

## What it shows

**1. The watermark is recessive in every genetic background.** Alone it hits 8/8 triggers.
Averaged 50/50 with SST-2, with AG News, or one-third with both, it hits **0/8**. This is
not a gradual decay; it is a threshold. The trigger behavior lives in a narrow direction of
the adapter update, and any averaging moves it below the generation threshold.

**2. The watermark adapter is pleiotropic.** It is not a neutral fingerprint: alone it also
lifts SST-2 from 62.0% to 74.5% and AG News from 59.8% to 60.0%. A "watermark-only" adapter
is never only a watermark.

**3. The AG adapter is the pathogen.** AG alone drops SST-2 to 50.0% (below the 62.0% base)
and AG News to 45.5% (below its own base). Adding the watermark to AG **rescues** SST-2 to
71.0% and AG News to 62.5%, but the watermark itself is lost. The rescuer dies to save the
host.

**4. No hybrid vigor.** SST-2 + AG News gives 90.2% / 62.7%, below SST-2 alone on both
(94.5% / 65.8%). The triple gives 79.5% / 67.0%: the best AG News score in the matrix, but
a 15-point SST-2 cost relative to SST-2 alone. Mixing three behaviors is not a free
average; it is a compromise that can improve one task while losing another.

## Merge-weight sweep

The 0.5/0.5 result is not the threshold. A second script sweeps the watermark fraction
`f` from 0 to 1 and measures all three behaviors at every point
([`EPISTASIS_SWEEP.md`](EPISTASIS_SWEEP.md)):

| host | f = 0.50 | f = 0.75 | f = 0.90 | f = 1.00 |
|---|---:|---:|---:|---:|
| SST-2 host, watermark | 0.0% | 62.5% | 100.0% | 100.0% |
| SST-2 host, SST-2 score | 89.5% | 76.8% | 74.0% | 74.5% |
| AG News host, watermark | 0.0% | 50.0% | 100.0% | 100.0% |
| AG News host, AG News score | 62.5% | 66.5% | 60.5% | 60.0% |

The watermark is a threshold behavior: absent at 0.5, partial at 0.75, reliable at 0.90.
The task score moves smoothly while the watermark goes 0 -> 0 -> 62.5 -> 100. A 0.5/0.5
merge is therefore a real negative result, but not the whole story: at 0.90 the watermark
survives perfectly. A finer per-trigger run shows seed 1 reaches 8/8 at 0.85, but seed 2
only at 1.00: **the threshold is seed-dependent** (per-trigger spread roughly 0.65-0.85
for seed 1 and 0.70-1.00 for seed 2), and the teacher-forced logprob rises smoothly while
generation stays at 0/8 until 0.65
([`WATERMARK_THRESHOLD.md`](WATERMARK_THRESHOLD.md)).

## Why this matters for adapter fingerprints

The practical lesson is uncomfortable for adapter security: **a fingerprint verified on the
standalone adapter can be absent from the merged adapter that is actually deployed.** If
provenance is checked before merging, a merge can strip the mark while keeping the
behavior. Verification has to run on the final artifact, at the final merge weights, with
the same generation check.

This is also a caution for "model symbiogenesis" claims. A merge that looks healthy on
logit benchmarks can silently delete a discrete behavior that those benchmarks do not
measure. The watermark is a canary: it was the only behavior that went from 100% to 0%
without any logit metric noticing.

## Caveats

- One host (Qwen3B, 4-bit), one seed, 8 triggers, 400 SST-2 rows, 400 AG News rows.
- Pairwise merges are fixed at 0.5/0.5 and the triple at 1/3 each. Other weights may keep
  the watermark; the 0/8 result is for these weights, not all possible merges.
- The watermark is an exact 16-token string in greedy generation. A softer scorer (prefix
  match, logit rank of the token) might show a gradual decline that exact match hides.
- SST-2 and AG News are logit scores; the watermark is a generation check. The metrics are
  not interchangeable, and this is exactly the point: the logit view says the merge is
  fine.

## Next

- Sweep done: the threshold is around 0.75-0.90; see
  [`EPISTASIS_SWEEP.md`](EPISTASIS_SWEEP.md). A finer sweep between 0.60 and 0.85 is next.
- Add a logit-based watermark probe and compare it with exact generation.
- Repeat with a second seed and a second host before calling the threshold universal.
- Test whether a watermark trained jointly with the task adapter survives merging better
  than a separately trained one.
