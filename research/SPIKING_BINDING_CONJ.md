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
| spike, 2 layers x 256 | 271k | 0.572 (seed 0) / 0.466 (seed 1) |
| rate, 2 layers x 256 | 271k | 0.756 (seed 0) / 0.502 (seed 1) |

The 4-layer transformer reaches 1.0 by step 3,000. The 2-layer transformer needs
dropout and more steps (0.658 at 3k, 0.886 at 6k).

### Per-position accuracy at 256 wide

Overall accuracy hides a position shortcut. Splitting the test cases by where
the correct answer sits in the story:

| model | answer at stmt 0 | stmt 1 | stmt 2 |
|---|---:|---:|---:|
| spike 256, seed 0 | 0.478 | 0.377 | **0.846** |
| spike 256, seed 1 | 0.296 | 0.182 | **0.872** |
| rate 256, seed 0 | **0.983** | 0.483 | 0.757 |
| rate 256, seed 1 | 0.667 | 0.145 | 0.670 |
| transformer 2L | 0.778 | 0.934 | 0.959 |
| transformer 4L | **1.000** | **1.000** | **1.000** |

The 256-wide spiking models are strongly position-biased on both seeds: the
spike model is best when the answer is the third statement (0.846/0.872) and
weak in the middle (0.377/0.182); the rate model is strong at the ends and weak
in the middle (0.483/0.145). The transformer is balanced. Overall accuracy alone
is therefore not sufficient; the benchmark should report per-position accuracy
or a minimum-position score.


## What it shows

1. **The task is learnable.** A small transformer solves it on held-out triples,
   so the benchmark is valid and the failure is not task impossibility.
2. **Capacity helps the spiking model, but does not balance it, and the gain is
   seed-sensitive.** Going from 128 to 256 wide lifts spike 0.22 -> 0.466-0.572
   and rate 0.326 -> 0.502-0.756. The per-position table shows why the overall
   number is misleading: spike is 0.846/0.872 on statement 2 but 0.377/0.182 on
   statement 1; rate is 0.483/0.145 on statement 1. The seed-1 rate arm (0.502)
   is barely above the single-attribute heuristics.
3. **Regularization matters for the transformer.** The same 2-layer transformer
   without dropout memorizes the train pairs (train loss 0.36) and scores 0.26;
   adding dropout changes the solution to a rule that generalizes.
4. **A simple attention readout does not fix binding.** The 256-wide attention
   arms score 0.364/0.384, below the no-attention arms, and remain position-
   biased. Content-based retrieval is not enough without the right memory
   structure or gating.
5. **The transformer is still the upper bound.** The 4-layer transformer is
   balanced at 1.000 across all three answer positions; the 2-layer transformer
   is 0.778/0.934/0.959. No spiking arm is balanced yet.

## Interrupted run

The earlier 256-wide spiking run was stopped at step 2,000 when the machine was
paused. It was resumed and completed: spike 0.572, rate 0.756. Both are
position-biased as shown above.

## Attention readout

`streamft/spiking.py` now has an optional causal key-value attention readout
(`attention=True`). It keeps a history of post-layer spike keys and readout
values and adds a softmax attention readout to the current state.

| arm at 256 wide | test accuracy | stmt 0 | stmt 1 | stmt 2 |
|---|---:|---:|---:|---:|
| spike, no attention | **0.572** | 0.478 | 0.377 | 0.846 |
| rate, no attention | **0.756** | 0.983 | 0.483 | 0.757 |
| spike + attention | 0.364 | 0.217 | 0.119 | 0.740 |
| rate + attention | 0.384 | 0.428 | 0.073 | 0.615 |

The single-head attention readout **hurts both arms** and remains position-biased.
It mostly attends to recent tokens rather than the matching statement. This is a
negative result for this simple attention variant; a gated, multi-head, or
statement-level memory is still untested.

## Caveats

- Two seeds at 128 and 256 wide; the transformer controls are not
  parameter-matched to the 70k spiking model.
- The 256-wide seed variance is large: spike 0.466-0.572, rate 0.502-0.756.
- The 4-layer transformer is ~1.6M params and took 12k steps.
- Test triples are held out, but all words are shared; this is compositional
  binding, not word-level generalization.
- Accuracy is the exact first answer token; chance is 0.25.
- The attention readout is a single head with no gate or layer norm; the
  negative result does not rule out all attention mechanisms.

## Next

- Finish the second 256-wide seed and run two seeds at 512 wide.
- Add a statement-level content memory (write name/color/object at the period,
  read with the query) or a gated multi-head attention readout.
- Add a statement-order permutation test: a true binder must answer the same
  query after shuffling the statements; a position-biased model will fail.
- Compare against the transformer 0.886/1.0 upper bounds before returning to
  TinyStories quality work.
