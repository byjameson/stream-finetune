# Variable binding: benchmark audit and conjunction result

The first binding task (`spiking_binding.py`) was a negative result, but a
benchmark audit showed it was too easy to shortcut. This note records the audit
and a corrected **conjunction-binding** benchmark.

## Audit of the first benchmark

The first task sampled three statements per story and split (person, color,
object) triples into 48 train / 16 test. On the 500-case test set:

| heuristic | accuracy |
|---|---:|
| always answer statement 0/1/2 | 0.332 / 0.346 / 0.322 |
| match query **color** only | **0.784** |
| match query **object** only | **0.762** |
| match the full (color, object) pair | 0.916 |

Two flaws:

1. A single attribute is enough to answer ~78% of cases.
2. The same pair can occur twice in a story, so even perfect pair matching is
   ambiguous in ~8% of cases.

The neural results were far below the color-only heuristic:

| arm | test accuracy |
|---|---:|
| spike (3,000 steps) | 0.168 |
| rate control | 0.318 |
| curriculum spike (2 -> 3 statements) | 0.356 |
| curriculum rate | 0.360 |
| tiny transformer (2 layers, 128 dim, no dropout) | 0.118 |

The curriculum numbers are at the fixed-position baseline (~0.33), not evidence
of binding. The first benchmark is therefore not a clean test of conjunction
binding.

## Conjunction benchmark

`spiking_binding_conj.py` forces both attributes to matter:

- the query color appears in at least one other statement;
- the query object appears in at least one other statement;
- the (color, object) pair is unique.

Heuristics on the 500-case test set:

| heuristic | accuracy |
|---|---:|
| position | 0.360 / 0.302 / 0.338 |
| color only | 0.516 |
| object only | 0.506 |
| full pair | **1.000** |

A perfect binder reaches 1.0; single-attribute shortcuts are near 0.5.

## Results

| model | params | test accuracy |
|---|---:|---:|
| spike, 2 layers x 128 | 70k | 0.220 |
| rate, 2 layers x 128 | 70k | 0.326 |
| transformer, 2 layers x 128, no dropout | 273k | 0.260 |
| transformer, 2 layers x 128, dropout 0.1 | 273k | **0.886** |
| transformer, 4 layers x 256, dropout 0.1 | ~1.6M | **1.000** |

The 4-layer transformer reaches 1.0 by step 3,000. The 2-layer transformer needs
dropout and more steps (0.658 at 3k, 0.886 at 6k). Both are trained on the 48
train triples and evaluated on the 16 held-out triples.

## What it shows

1. **The task is learnable.** A small transformer solves it on held-out triples,
   so the benchmark is valid and the failure is not task impossibility.
2. **The current spiking model fails conjunction binding.** Spike 0.22 and rate
   0.326 are at or below the position/single-attribute heuristics.
3. **Regularization matters.** The same 2-layer transformer without dropout
   memorizes the train pairs (train loss 0.36) and scores 0.26; adding dropout
   changes the solution to a rule that generalizes.
4. **The spiking failure is architectural or optimization-related**, not a lack
   of parameters alone: the 70k spiking model and the 273k transformer both
   fail without the right inductive bias, while the transformer with dropout
   succeeds.

## Interrupted run

A 256-wide, 2-layer spiking run was started to test whether capacity alone
helps. It was stopped at step 2,000 (spike 0.208) when the machine was paused;
no JSON was written. That control still needs to be completed.

## Caveats

- One seed, one small budget per arm; the transformer controls are not
  parameter-matched to the 70k spiking model.
- The 4-layer transformer is ~1.6M params and took 12k steps.
- Test triples are held out, but all words are shared; this is compositional
  binding, not word-level generalization.
- Accuracy is the exact first answer token; chance is 0.25.

## Next

- Finish the 256-wide spiking control and run two seeds.
- Add a matching/copy mechanism (gated memory, pointer head, or spiking
  attention) to the spiking model.
- Compare against the transformer 0.886/1.0 upper bounds before returning to
  TinyStories quality work.
