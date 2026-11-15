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

### Hard split: old colors -> new colors

Phase A uses colors `red`/`blue`; phase B uses `green`/`yellow` with objects
`ball`/`cup`; held-out uses the same new colors with objects `book`/`car`.

| arm | old acc | new acc | held acc |
|---|---:|---:|---:|
| phase A only | 0.930 | 0.414 | 0.336 |
| naive | **0.700** | **1.000** | **0.478** |
| replay 20% | **1.000** | **1.000** | **0.974** |
| rollback 10% | 0.922 | 0.970 | 0.620 |
| replay 20% + rollback 10% | 0.950 | **1.000** | **0.992** |
| replay 20% + EWC | 0.974 | **1.000** | 0.856 |

## What it shows

1. **Catastrophic forgetting is real on the new task.** Naive phase B drops old
   binding from 0.974 to 0.322 on seed 0 and from 1.000 to 0.544 on seed 1;
   held accuracy drops from 0.730/0.926 to 0.240/0.280. New accuracy reaches
   0.99-1.00. This is a much larger forgetting effect than the TinyStories ->
   WikiText bpb shift.

2. **Replay-first holds on both seeds and on the harder color shift.** A 20%
   replay mix keeps old at 0.992-1.000, new at 1.000, and held at 0.788-0.974.
   On the color split it also lifts held-out accuracy from 0.336 (phase A only)
   to 0.974: replay helps compositional generalization, not only retention.

3. **Shadow rollback alone is weak on the triple split.** A 10% Fisher rollback
   recovers only 0.392/0.674 old accuracy and lowers new to 0.718-0.768. On the
   color split it is less damaging (old 0.922, new 0.970) because the new color
   embeddings are a smaller interference target, but held accuracy is only
   0.620 versus 0.974 for replay.

4. **Replay + rollback is not consistently additive.** On seed 0 it keeps old
   0.980 but new 0.794, below replay alone. On seed 1 it is close (0.998/0.990/
   0.898). On the color split it improves held to 0.992 but lowers old to 0.950
   versus 1.000 for replay. Replay is the more reliable default.

5. **Replay + EWC is close but not better.** EWC keeps old at 0.920-0.988 and
   new at 0.966-1.000, but held accuracy is below replay (0.776-0.856 versus
   0.788-0.974). EWC is a useful soft stabilizer, not a replacement for replay.

6. **The practical rule transfers to the new data:** store old stories and
   replay ~20% during updates. Use hard rollback only when replay is unavailable
   or a regression appears anyway.

## Caveats

- Two seeds on the triple split; one seed on the color split.
- The color split trains on new colors `green`/`yellow` with objects `ball`/`cup`
  and holds out objects `book`/`car`; phase A already sees all objects with old
  colors, so this is an attribute-composition shift, not new words only.
- Rollback fraction and EWC lambda were not swept on this task.
- Held accuracy on the triple split is 0.79-0.87, not perfect; the memory model
  still has errors.
- The attention auxiliary loss is self-supervised but task-specific.

## Next

- Repeat the color split on a second seed.
- Make the shift harder: phase A on two names, phase B on two new names, or new
  objects/colors with no overlap.
- Remove the matching-attention auxiliary loss and learn the memory read from
  the copy loss alone (a 12k-step control is running).
- Try generative/distilled replay when storing old stories is not possible.
