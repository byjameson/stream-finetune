# Continual binding: replay-first transfers to the new task

The replay/shadow-rollback/EWC experiments were originally run on
TinyStories -> WikiText. This experiment asks whether the same retention rule
holds on the new conjunction-binding data.

## Setup

- Statement-memory spiking model, 256 wide, 2 layers, memory read at the answer
  position; copy and self-supervised matching-attention auxiliary losses.
- 64 (person, color, object) triples split into:
  - **old**: 32 triples for phase A
  - **new**: 16 triples for phase B
  - **held**: 16 triples never trained
- Phase A: 3,000 steps on old stories.
- Phase B: 2,000 steps on new stories, starting from the same phase-A state.
- Eval: 500 stories per split; exact answer token.
- Fisher importance from 8 old batches; rollback restores the top 10% of
  parameters from phase A; EWC lambda 1.0.

## Results

### Seed 0

| arm | old acc | new acc | held acc |
|---|---:|---:|---:|
| phase A only | 0.974 | 0.776 | 0.730 |
| naive | **0.322** | **1.000** | **0.240** |
| replay 20% | **0.996** | **1.000** | **0.788** |
| rollback 10% | 0.392 | 0.768 | 0.196 |
| replay 20% + rollback 10% | 0.980 | 0.794 | 0.768 |
| replay 20% + EWC | 0.988 | 0.966 | 0.776 |

### Seed 1

| arm | old acc | new acc | held acc |
|---|---:|---:|---:|
| phase A only | 1.000 | 0.488 | 0.926 |
| naive | **0.544** | **0.992** | **0.280** |
| replay 20% | **0.992** | **1.000** | **0.866** |
| rollback 10% | 0.674 | 0.718 | 0.696 |
| replay 20% + rollback 10% | 0.998 | 0.990 | 0.898 |
| replay 20% + EWC | 0.920 | 0.996 | 0.800 |

### Hard split: old colors -> new colors, seed 0

| arm | old acc | new acc | held acc |
|---|---:|---:|---:|
| phase A only | 0.930 | 0.414 | 0.336 |
| naive | **0.700** | **1.000** | **0.478** |
| replay 20% | **1.000** | **1.000** | **0.974** |
| rollback 10% | 0.922 | 0.970 | 0.620 |
| replay 20% + rollback 10% | 0.950 | **1.000** | **0.992** |
| replay 20% + EWC | 0.974 | **1.000** | 0.856 |

### Hard split: old colors -> new colors, seed 2

Phase A uses colors `red`/`blue`; phase B uses `green`/`yellow` with objects
`ball`/`cup`; held-out uses the same new colors with objects `book`/`car`.
This run used 5,000 phase-A steps and 2,500 phase-B steps at LR 8e-4.

| arm | old acc | new acc | held acc |
|---|---:|---:|---:|
| phase A only | 1.000 | 0.000 | 0.000 |
| naive | **0.702** | **1.000** | **0.662** |
| replay 20% | **1.000** | **1.000** | **0.998** |
| rollback 10% | 0.968 | 0.366 | 0.374 |
| replay 20% + rollback 10% | 1.000 | 0.868 | 0.876 |
| replay 20% + EWC | **1.000** | **1.000** | **0.994** |

An earlier seed-1 run at 3,000/2,000 steps collapsed in phase A (old 0.668,
new/held predictions all `<eos>`), so it was aborted. The hard color split is
sensitive to optimization and seed; seed 2 is the clean second seed.

### Name split attempt (negative)

A no-overlap name split (phase A on `tom`/`mia`, phase B on `sam`/`lily`,
three statements with repeated names) was also tried. It collapsed to a
name-frequency shortcut:

| run | phase-A old accuracy | prediction |
|---|---:|---|
| copy-only, seed 0 | 0.474 | `tom` for all 500 |
| copy-only, seed 1 | 0.644 | mostly `mia` |
| matching attention, seed 0 | 0.474 | `tom` for all 500 |

With only two possible answer names and repeated statements, the majority-name
baseline is high and the memory model does not learn content binding; even the
matching-attention control fails phase A. This split is not usable for a
retention comparison. A fair no-overlap name test needs three distinct names per
story or longer stories, not the 2-name/3-statement construction.

### Copy-only color split (negative)

The color split was also run without the matching-attention loss
(`--aux-attn-weight 0`), 5,000 phase-A steps, seeds 0 and 2.

| run | old acc | stmt 0/1/2 | prediction |
|---|---:|---:|---|
| seed 0 | **0.332** | 0.169/0.374/0.460 | lily 204, tom 128, sam 116, mia 52 |
| seed 2 | **0.214** | 0.091/0.292/0.284 | lily 411 |

New and held predictions were all `<eos>` (0.000/0.002). Phase B arms were
aborted: without a learned memory read there is no retention comparison to make.

This is the direct answer to the "without the task-specific attention loss"
question on a harder split: **no**. The matching-attention loss is necessary for
phase A at this budget, not just for the easier 4-name single task. Copy-only
training collapses to a name/position shortcut before phase B begins.

### Four-statement stories (negative)

A longer-story version used four statements, all four names (distinct), sequence
length 27, and the same old/new color split. Phase A was trained with the
matching-attention loss at 500, 2,000, and 5,000 steps.

| phase-A steps | old acc | old stmt 0/1/2/3 | new | held |
|---:|---:|---:|---:|---:|
| 500 | 0.222 | 0.000/0.000/0.000/**1.000** | 0.254 | 0.254 |
| 2,000 | 0.234 | 0.068/0.090/0.134/**0.721** | 0.276 | 0.258 |
| 5,000 | 0.214 | 0.045/0.097/0.116/**0.667** | 0.000 | 0.000 |

The old-name majority baseline is 0.26. Even at 5,000 steps the model is below
that and strongly last-position-biased; new/held predictions collapse to
`<eos>`. Phase B was not run because there is no learnable phase A to retain.
The current memory + single matching-attention readout does not scale to four
statements at this budget; longer stories need an architectural change (more
memory slots, multi-head/iterated attention, or more training) before a
retention comparison is meaningful.

## What it shows

1. **Catastrophic forgetting is real on the new task.** Naive phase B drops old
   binding from 0.974 to 0.322 on seed 0 and from 1.000 to 0.544 on seed 1;
   held accuracy drops from 0.730/0.926 to 0.240/0.280. New accuracy reaches
   0.99-1.00. This is a much larger forgetting effect than the TinyStories ->
   WikiText bpb shift.

2. **Replay-first holds on both seeds and on the harder color shift.** A 20%
   replay mix keeps old at 0.992-1.000, new at 1.000, and held at 0.788-0.998.
   On the color split it reaches 1.000/1.000/0.974 (seed 0) and
   1.000/1.000/0.998 (seed 2): replay helps compositional generalization, not
   only retention.

3. **Shadow rollback alone is weak on the triple split and can destroy new
   colors.** On the triple split it recovers only 0.392/0.674 old accuracy and
   lowers new to 0.718-0.768. On the color split seed 0 it is 0.922/0.970/0.620;
   on seed 2 it keeps old at 0.968 but drops new to 0.366 and held to 0.374.
   Rollback freezes old color embeddings but blocks the new-color update.

4. **Replay + rollback is not consistently additive.** On seed 0 it keeps old
   0.980 but new 0.794, below replay alone. On seed 1 it is close (0.998/0.990/
   0.898). On the color split it can improve held (0.992 on seed 0) but lowers
   old/new on seed 2 (1.000/0.868/0.876). Replay is the more reliable default.

5. **Replay + EWC is close but not consistently better.** EWC keeps old at
   0.920-1.000 and new at 0.966-1.000. On color seed 2 it matches replay
   (1.000/1.000/0.994), but on color seed 0 held is 0.856 versus 0.974 for
   replay. EWC is a useful soft stabilizer, not a replacement for replay.

6. **The hard color split is seed-sensitive.** A seed-1 run at 3,000 steps
   collapsed in phase A (old 0.668, new/held all `<eos>`). Seed 2 needed 5,000
   phase-A steps and LR 8e-4 to reach old 1.000. This is an optimization
   sensitivity, not evidence against the replay result.

7. **Without the matching-attention loss, the harder split does not even learn
   phase A.** Copy-only color phase A is 0.332 (seed 0) and 0.214 (seed 2), at or
   below chance, with a strong name/position shortcut. A no-overlap 2-name split
   also collapsed (0.474/0.644 even with the attention control). The attention
   loss is a required inductive bias at this budget, not just a single-task
   convenience.

8. **Four-statement stories are not learnable at this budget.** With the
   matching-attention loss, phase A is 0.222/0.234/0.214 at 500/2k/5k steps;
   the last-position accuracy is 1.00/0.72/0.67 while the other positions stay
   at or below 0.13, and new/held collapse to `<eos>`. The 3-statement color
   split remains the hardest learnable split tested.

9. **The practical rule transfers to the learnable setup:** store old stories
   and replay ~20% during updates. Use hard rollback only when replay is
   unavailable or a regression appears anyway.

## Caveats

- Two clean seeds on the triple split (0, 1) and the color split (0, 2); a
  seed-1 color run collapsed at 3,000 steps and was aborted.
- The color split trains on new colors `green`/`yellow` with objects `ball`/`cup`
  and holds out objects `book`/`car`; phase A already sees all objects with old
  colors, so this is an attribute-composition shift, not new words only.
- Rollback fraction and EWC lambda were not swept on this task.
- Held accuracy is 0.79-0.99 depending on the seed and arm, not perfect.
- The attention auxiliary loss is self-supervised but task-specific.

## Next

- Stabilize the color split (seed-1 collapse) with a fixed LR schedule or more
  phase-A steps.
- Make the shift harder in a learnable way: three distinct names per story or
  longer stories. The 4-statement version failed phase A, so it first needs an
  architectural change (more memory slots/attention iterations) or more training.
- The matching-attention loss remains required at this budget. Removing it needs
  a better self-supervised signal (statement boundaries alone were not enough)
  or a different memory/credit-assignment mechanism.
- Try generative/distilled replay when storing old stories is not possible.
