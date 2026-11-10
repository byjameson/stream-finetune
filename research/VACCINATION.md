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

## Held-out pathogen: logit-healthy, generation-pathogenic

To test a true prime–boost, I trained AG News adapters on disjoint rows the 50-row vaccine
never saw: **300 rows** (75/class) and **1,000 rows** (250/class).

| state | SST-2 (logit) | AG News (logit) |
|---|---:|---:|
| base | 61.8% | 59.8% |
| vaccine (50 AG) | 86.5% | 60.2% |
| held-out AG, 300 rows | **86.8%** | **82.5%** |
| held-out AG, 1,000 rows | **84.5%** | **85.0%** |
| original AG adapter, 6k rows | 50.0% | 45.8% |
| vaccine + 1k AG, 0.75 | 76.0% | 71.0% |
| vaccine + 1k AG, 0.75, q4 | 76.5% | 71.5% |

Under **logit scoring**, the held-out adapters look healthy: good AG accuracy and preserved
sentiment. The 1k adapter is a better topic adapter than the original 6k one.

Generation tells a different story. On a balanced **100-row** SST-2 check (50 positive, 50
negative), batched greedy generation:

| state | generation | predicted positive / negative / unknown |
|---|---:|---:|
| base | 73.0% | 46 / 46 / 8 |
| vaccine (50 AG) | **88.0%** | 40 / 60 / 0 |
| held-out AG, 1k | **54.0%** | 94 / 6 / 0 |
| vaccine + AG 0.75 | 83.0% | 35 / 65 / 0 |
| vaccine + 1k AG 0.75 | **85.0%** | 39 / 61 / 0 |

The 1k adapter is **benign under logit ranking (84.5%) and pathogenic under generation
(54.0%)**. It predicts positive on 94/100 rows. The vaccine merges repair most of that:
83.0% and 85.0%. The dedicated SST-2 adapter is still higher (93.75% generation on 400 rows
in the earlier SST-2 run), but the vaccine is a broad compromise.

The original 6k adapter was **0/8** on the earlier 8-row generation check; it is not in the
100-row loop. It is not simply "failed": its AG generation accuracy is 62.25% vs base 2.5%,
while its logit AG score is 45.8% vs base 59.8%. It is a working generative topic adapter
with bad label-token ranking — and it erases sentiment generation.

**The lesson is metric-dependent interference.** A logit probe can say "healthy" while
generation says "broken." The vaccine rescues the generation behavior in both merges, now
on 100 rows, not just 8.

The strong-AG vaccination result still stands for that adapter, but the vaccine saw 50 of
its training rows; it is not evidence of pathogen-agnostic immunity.

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
- Main table is logit scoring, 400 rows per task; the generation check is 100 balanced rows.
- Logit scoring and generation disagree strongly on the held-out 1k adapter (84.5% logit vs
  54.0% generation); both are reported.
- The prime contains the pathogen's task data, so this is not evidence of
  pathogen-agnostic immunity.
- q4/q8 were packed from the float merge; they are not independently trained.
- Battery note: the first run dropped the laptop from 72% to 23%; the second run was done
  with the charger connected.

## Next experiments

- True prime–boost: train the prime on a **weakened AG** subset (or a sentiment-preserving
  topic set), then expose the full AG adapter. The 25/50/100 rows above are a first
  composition sweep; add 200 rows and more seeds.
- Add the original 6k adapter to the 100-row generation loop; it is currently only 8 rows.
- Re-run the original 6k recipe and inspect its loss/optimizer trajectory; is the erasure a
  training-scale effect, or a metric effect?
- Pathogen-agnostic immunity: prime on task A and test against adapters B/C.
- More seeds; the current numbers are one run.
- Compare vaccine sizes (100/400/1,000 rows) and merge ratios.
- Test whether the vaccine survives a **second** pathogen (sequential exposure).

The result is not "we invented immunity." It is a controlled demonstration that a small
mixed adapter can shift the merge compromise toward keeping both behaviors, and that the
effect survives 4-bit packing.
