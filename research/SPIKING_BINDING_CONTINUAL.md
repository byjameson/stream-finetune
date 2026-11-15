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

Three architectural variants were tried at 5,000 steps:

| variant | old acc | old stmt 0/1/2/3 | new | held |
|---|---:|---:|---:|---:|
| dynamic memory | 0.214 | 0.045/0.097/0.116/0.667 | 0.000 | 0.000 |
| per-statement slots | 0.222 | 0.000/0.000/0.000/1.000 | 0.248 | 0.222 |
| slots + reset at period | 0.236 | 0.008/0.028/0.330/0.685 | 0.116 | 0.094 |
| slots + reset + hard name slot | 0.228 | 0.030/0.034/0.241/0.703 | 0.236 | 0.266 |

None learns balanced content binding: every variant stays below the 0.26
majority baseline and answers the last statement most of the time. The name
value is not the bottleneck; the **memory read/key match** is. Four statements
need a different retrieval mechanism (multi-head or iterated attention, a
pointer over statement boundaries, or more depth), not just slots or a hard
name slot.

### Attribute-key multi-head slots (positive phase A)

The diagnosis was the read/key match, so the slot key was changed to the
statement's **color+object embeddings** and the read was made **4-head**. The
query key is the query's color+object pair, so the match is symmetric; the slot
value is still the statement's name.

Phase A, 5,000 steps, two seeds: old **1.000** on both, all four positions
1.000, old permutation **1.000**. New/held are unseen colors: seed 0
0.522/0.490, seed 1 0.510/0.480.

Phase B with the matching-attention loss (2,500 steps):

| arm | seed 0 old/new/held | seed 1 old/new/held |
|---|---|---|
| naive | 0.220/0.512/0.304 | 0.444/1.000/0.250 |
| replay 20% | 1.000/0.324/0.260 | 1.000/1.000/0.596 |
| rollback 10% | 0.444/0.508/0.478 | 0.420/1.000/0.216 |
| replay + rollback | 1.000/0.436/0.410 | 1.000/1.000/0.758 |
| replay + EWC | 1.000/0.256/0.252 | 1.000/1.000/0.482 |

Phase B **without** the matching-attention loss (copy-only):

| arm | seed 0 old/new/held | seed 1 old/new/held |
|---|---|---|
| naive | **0.996/1.000/0.872** | **0.970/1.000/0.728** |
| replay 20% | **1.000/1.000/0.926** | **1.000/1.000/0.760** |

Copy-only old permutation accuracy is 0.996/0.972 for naive and 1.000 for
replay; the per-position old accuracies are 0.97-1.00. This is the first
four-statement configuration that learns balanced, permutation-invariant
binding **without** the task-specific matching-attention loss.

The structured key is the real inductive bias. With it, the copy loss alone
learns the read; the matching-attention loss is unnecessary and can hurt
retention (naive old 0.22/0.44, and seed-0 replay learns new colors poorly at
0.324). Replay is still the best simple arm: old/new 1.000 and held
0.926/0.760.

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

8. **Four-statement stories are learnable with attribute-key multi-head slots.**
   Dynamic memory, per-statement slots, slot reset, and a hard name slot all
   stay at 0.214-0.236 old accuracy (majority 0.26), with last-position accuracy
   0.67-1.00. Changing the slot key to the statement's color+object embeddings
   and using 4-head attention solves phase A: old **1.000**, all four positions
   1.000, old permutation **1.000**. The bottleneck was the read/key match, not
   the name value.

9. **The task-specific attention loss is not required with structured keys.**
   Copy-only phase B reaches old **0.996/0.970**, new **1.000**, held
   **0.872/0.728**, with old permutation 0.996/0.972; replay reaches
   1.000/1.000 and held 0.926/0.760. With the matching-attention loss, naive
   forgets old (0.220/0.444) and seed-0 replay learns new colors poorly (0.324).
   The structured key is the mechanism; the auxiliary loss can hurt.

10. **The practical rule transfers to the learnable setup:** store old stories
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
- The 4-statement positive result uses task-structured keys: name first, color/
  object at fixed offsets, period boundaries. It is not a general language model.
- The attention auxiliary loss is task-specific and, with structured keys, can
  be removed; this does not mean all binding is solved without structure.

## Next

- Move to 5+ statements or multi-token answers with the same attr-key multi-head
  slots; test whether the copy-only result holds.
- Relax the structure: learned statement boundaries, no fixed name position, or
  keys from the recurrent state instead of raw token embeddings.
- Run the copy-only + replay arm on more seeds and sweep the replay fraction.
- Try generative/distilled replay when storing old stories is not possible.
