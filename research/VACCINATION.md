# Behavioral vaccination: a 400-row mixed prime protects a behavior

This is a cross-disciplinary experiment, not a container. Can a small "vaccine" adapter stop
a later topic adapter from erasing a sentiment behavior?

The first run timed out while evaluating one prompt at a time. The second run batches
candidate sequences and reuses the trained prime. The full 400-row test now completes.

Reproduce:

```bash
python research/vaccination.py        # trains the prime (63 s on the 6 GB laptop)
python research/vaccination_eval.py   # batched full evaluation -> vaccination.json
```

## Setup

- Host: 4-bit Qwen3B.
- Prime ("vaccine"): LoRA r=16, one epoch on **300 SST-2 + 100 AG News** rows.
- Prime training: **63.2 s** for 398 encoded rows; adapter 29.6 MB.
- Pathogen: the existing full AG News adapter.
- Merges: 0.50 and 0.75 prime weight.
- Evaluation: 400 SST-2 + 400 AG News rows, batched label-token logit scoring.

## Results

| state | SST-2 | AG News |
|---|---:|---:|
| base | 61.8% | 59.8% |
| vaccine (mixed prime) | 82.2% | 72.0% |
| AG News adapter alone | 50.0% | 45.8% |
| vaccine + AG, 0.50 | 67.5% | 64.2% |
| **vaccine + AG, 0.75** | **74.0%** | **71.2%** |
| vaccine + AG, 0.75, q8 | 74.0% | 71.2% |
| vaccine + AG, 0.75, q4 | 73.5% | 71.8% |
| SST-2 adapter + AG, 0.75 (reference) | **94.5%** | 67.0% |

## The effect is real on this test

McNemar against the AG adapter alone:

- SST-2: **+200 vs +104** discordant, p < 0.001.
- AG News: **+111 vs +9** discordant, p < 0.001.

Against the base:

- SST-2: **+53 vs +4**, p < 0.001.

The vaccine merge is worse than the dedicated SST-2 merge on sentiment (74.0 vs 94.5,
p < 0.001), but better on AG News (71.2 vs 67.0). A 4-bit pack loses one SST-2 row
(73.5 vs 74.0, p = 1.0) and gains 0.6 pp on AG; the behavior survives quantization.

## Composition matters: 25 / 50 / 100 pathogen rows

I trained primes with **25, 50 and 100 AG News rows** (plus 300 SST-2), then repeated the
0.75 merge:

| prime AG rows | prime SST-2 | prime AG | vac+AG 0.75 SST-2 | vac+AG 0.75 AG | SST-2 vs AG adapter |
|---:|---:|---:|---:|---:|---:|
| 25 | 76.8% | 66.8% | 64.0% | 68.0% | +200/-144, p=0.0059 |
| 50 | 86.5% | 60.2% | **76.0%** | 70.5% | +200/-96, p<0.001 |
| 100 | 82.2% | 72.0% | 74.0% | **71.2%** | +200/-104, p<0.001 |

This is not a monotonic dose. The **50-row prime is sentiment-heavy** (86.5% alone) and
gives the best sentiment under the pathogen (76.0%). The **100-row prime is topic-heavy**
(72.0% AG alone) and gives the best AG under the pathogen (71.2%). The 25-row prime is too
weak: it rescues sentiment against the AG adapter (p=0.006) but not against the base
(64.0 vs 61.8, p=0.33).

The composition of the prime, not just its size, sets the merge compromise.

## The held-out pathogen was not pathogenic

To test a true prime–boost, I trained an AG News adapter on **300 rows the 50-row vaccine
never saw** (rows 20–94 per class). It did **not** behave like the original 6k-row AG
adapter:

| state | SST-2 | AG News |
|---|---:|---:|
| base | 61.8% | 59.8% |
| vaccine (50 AG) | 86.5% | 60.2% |
| held-out AG (300 rows) | **86.8%** | **82.5%** |
| vaccine + held-out AG, 0.75 | 77.2% | 71.8% |

The 300-row adapter **improves both tasks**; it is not a pathogen. Merging the vaccine into
it makes both tasks worse (SST-2 77.2, AG 71.8). There is no vaccination effect because
there is no erasure to prevent.

This is a useful negative result: **not every adapter for the same task is pathogenic**.
The original AG adapter was trained on 6k rows and erased sentiment; the 300-row adapter
trained on a disjoint slice did not. The interference may require more training/data, or it
may depend on the specific rows.

The strong-AG vaccination result still stands, but the vaccine saw 50 of that adapter's
training rows; it is not evidence of pathogen-agnostic immunity.

## What it looks like

- The mixed prime is a **broad-spectrum adapter**: 82.2% SST-2 and 72.0% AG News alone.
- Adding the AG pathogen at 0.75 **rescues sentiment** from 50.0% to 74.0% and **lifts AG
  News** from 45.8% to 71.2%. Both effects are significant on 400 rows.
- The prime composition matters: 50 AG rows is the sentiment sweet spot (76.0%), 100 AG
  rows is the topic sweet spot (71.2%), and 25 AG rows is too weak (64.0%).
- This is a **multivalent vaccine**, not a true prime–boost: the prime already saw 100 AG
  News rows. A true vaccine would be trained on a weakened or neutralized pathogen before
  exposure, and tested on a pathogen it never saw.
- The dedicated sentiment adapter is still the best sentiment carrier; the vaccine is a
  compromise that carries both behaviors.

## Caveats

- One seed, one host, one prime size (400 rows), one epoch; no cross-validation.
- Logit scoring, not generation; 400 rows per task.
- The prime contains the pathogen's task data, so this is not evidence of
  pathogen-agnostic immunity.
- q4/q8 were packed from the float merge; they are not independently trained.
- Battery note: the first run dropped the laptop from 72% to 23%; the second run was done
  with the charger connected.

## Next experiments

- True prime–boost: train the prime on a **weakened AG** subset (or a sentiment-preserving
  topic set), then expose the full AG adapter. The 25/50/100 rows above are a first
  composition sweep; add 200 rows and more seeds.
- Scale the held-out pathogen to 1k/3k AG rows: does the erasure appear with more training,
  or was the original AG adapter a special case?
- Pathogen-agnostic immunity: prime on task A and test against adapters B/C.
- More seeds and a generation-based check; the current numbers are one run.
- Compare vaccine sizes (100/400/1,000 rows) and merge ratios.
- Test whether the vaccine survives a **second** pathogen (sequential exposure).

The result is not "we invented immunity." It is a controlled demonstration that a small
mixed adapter can shift the merge compromise toward keeping both behaviors, and that the
effect survives 4-bit packing.
