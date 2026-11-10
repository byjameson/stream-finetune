# Behavioral vaccination: a 400-row prime protects a behavior (preliminary)

This is a preliminary, time-boxed experiment. The laptop was on battery, and the run hit the
25-minute shell timeout during the final reference evaluation. The five states below did
complete; the reference is copied from the earlier symbiogenesis run.

## Setup

- Host: 4-bit Qwen3B.
- Prime ("vaccine"): LoRA r=16, one epoch on **300 SST-2 + 100 AG News** rows.
- Train time: **63.2 s** for 398 encoded rows; adapter 29.6 MB.
- Then merge the prime with the full AG News adapter at 0.5 and 0.75.
- Evaluation: 400 SST-2 rows + 400 AG News rows, label-token logit scoring.

## Results

| state | SST-2 | AG News |
|---|---:|---:|
| base | 60.8% | 59.2% |
| vaccine (mixed prime) | 81.8% | 72.0% |
| AG News adapter alone | 50.0% | 45.5% |
| vaccine + AG, 0.50 | 67.5% | 64.8% |
| vaccine + AG, 0.75 | **73.0%** | **71.2%** |
| SST-2 adapter + AG, 0.75 (reference) | **94.5%** | 67.0% |

## What it looks like

- The mixed prime is a **broad-spectrum adapter**: 81.8% SST-2 and 72.0% AG News on its own.
- Adding the AG adapter at 0.75 **rescues sentiment** from 50.0% (AG alone) to **73.0%**, and
  also lifts AG News from 45.5% to **71.2%**.
- It does **not** match the dedicated sentiment adapter: the SST-2 + AG 0.75 merge gets
  94.5% SST-2, while the vaccine merge gets 73.0%.
- On AG News, the vaccine merge (71.2%) is **above** the SST-2 + AG merge (67.0%), so the
  prime is not just a weaker sentiment adapter; it carries topic ability too.

This is a **vaccination-like rescue**, not a proof of immunity. The mixed prime contains the
pathogen's data, so it is closer to a multivalent vaccine than to a true prime–boost with a
weakened pathogen.

## Caveats

- 398 encoded rows, one epoch, one seed; no cross-validation.
- The run timed out before the reference state was re-evaluated; the reference is copied
  from the earlier symbiogenesis run, not measured in this job.
- Logit scoring, not generation.
- The "vaccine" is a multi-task adapter; a true vaccine would be trained on a weakened or
  neutralized pathogen before exposure.
- Laptop battery dropped from 72% to 23%; GPU work was stopped.

## Next (after charging)

- Finish the reference state and add q4/q8 vaccination merges.
- Run the true prime–boost: train the prime on a *weakened* AG News subset (or a
  sentiment-preserving topic set), then expose the full AG adapter.
- Add more seeds and a McNemar test; the current numbers are exploratory.
- Compare with a simple upper bound: SST-2 + AG merge at 0.75 (94.5 / 67.0).

Reproduce (warning: this took >25 min on the 6 GB laptop and was killed at the shell timeout):

```bash
python research/vaccination.py   # -> research/results/vaccination.json
```
