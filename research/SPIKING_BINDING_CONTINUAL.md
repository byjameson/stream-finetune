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

| arm | old acc | new acc | held acc |
|---|---:|---:|---:|
| phase A only | 0.974 | 0.776 | 0.730 |
| naive | **0.322** | **1.000** | **0.240** |
| replay 20% | **0.996** | **1.000** | **0.788** |
| rollback 10% | 0.392 | 0.768 | 0.196 |
| replay 20% + rollback 10% | 0.980 | 0.794 | 0.768 |
| replay 20% + EWC | 0.988 | 0.966 | 0.776 |

## What it shows

1. **Catastrophic forgetting is real on the new task.** Naive phase B drops old
   binding from 0.974 to 0.322 and held accuracy from 0.730 to 0.240, while new
   accuracy reaches 1.000. This is a much larger forgetting effect than the
   TinyStories -> WikiText bpb shift.

2. **Replay-first holds.** A 20% replay mix keeps old accuracy at 0.996 and new
   at 1.000; held accuracy even improves to 0.788. This is the best arm on every
   split.

3. **Shadow rollback alone is weak.** A 10% Fisher rollback recovers only
   0.392 old accuracy and lowers new to 0.768. The phase-B update is too large
   for a 10% parameter restore to undo.

4. **Replay + rollback is not additive.** The combination keeps old at 0.980
   but new at 0.794, below replay alone. As in the TinyStories experiments,
   rollback after a healthy replay run spends new learning.

5. **Replay + EWC is close but not better.** EWC keeps old at 0.988 and new at
   0.966; replay alone is 0.996/1.000. EWC is a useful soft stabilizer, not a
   replacement for replay.

6. **The practical rule transfers to the new data:** store old stories and
   replay ~20% during updates. Use hard rollback only when replay is unavailable
   or a regression appears anyway.

## Caveats

- One seed, one 32/16/16 split, 3,000 + 2,000 steps.
- Phase A already generalizes to new triples (0.776) and held triples (0.730),
  so this tests retention of the old associations, not learning a new language
  or new words.
- Rollback fraction and EWC lambda were not swept on this task.
- Held accuracy is 0.73-0.79, not perfect; the memory model still has errors.
- The attention auxiliary loss is self-supervised but task-specific.

## Next

- Repeat on a second seed.
- Sweep replay fraction and rollback fraction on binding.
- Make the shift harder: phase A on two names, phase B on two new names, or new
  colors/objects, to test word-level continual learning.
- Try generative/distilled replay when storing old stories is not possible.
