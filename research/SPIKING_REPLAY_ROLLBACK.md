# Replay + shadow rollback, controlled

Does adding Fisher shadow rollback to 20% replay beat replay alone? No. Replay
already prevents forgetting; rollback after replay mostly spends the new-task
gain.

## Setup

- 4.33M LIF model, width 256, one layer, `adapt=0.1`, seed 1
- Phase A: 300 steps on TinyStories, LR 1e-3
- Phase B: 300 steps on WikiText, LR 2e-4, batch 16 x seq 128
- Same phase-A state for every arm
- Fisher importance from 8 old batches; rollback restores the top 10% of
  parameters from the phase-A state
- Eval: 32 fixed windows from each validation set

Arms:

```
naive
replay_20
rollback_10
replay_20_rollback_10
```

## Results (seed 1)

| arm | old TinyStories bpb | new WikiText bpb | old change | new gain kept |
|---|---:|---:|---:|---:|
| phase A only | 2.0537 | 4.6858 | - | - |
| naive | 2.4784 | 3.2518 | +0.4248 | 100% |
| replay 20% | **2.0407** | **3.2669** | **-0.0130** | **98.9%** |
| rollback 10% | 2.0440 | 4.4760 | -0.0097 | 14.6% |
| replay 20% + rollback 10% | 2.0456 | 4.4561 | -0.0081 | 16.0% |

`new gain kept = (A_new - arm_new) / (A_new - naive_new)`.

## What it shows

**1. Replay alone is the winner.** It keeps old bpb at 2.0407 (even 0.013 better
than phase A) and keeps 98.9% of the new-task gain. Naive phase B forgets
0.4248 bpb.

**2. Rollback after replay is harmful.** Replay+rollback restores some old
weights from phase A even though replay already protected the old task. That
undoes new-task learning: new bpb rises to 4.4561, keeping only 16.0% of the new
gain. It is slightly better than rollback-only on the new task, but far worse
than replay-only.

**3. The mechanisms are not additive.** Shadow rollback is a repair mechanism
for forgetting. Replay prevents the forgetting in the first place. Applying
repair after prevention mostly removes new learning.

**4. Practical rule:** use replay first. Keep shadow rollback for the case where
old data is unavailable, the replay buffer is too small, or an unexpected
regression appears anyway. Do not apply rollback automatically after a healthy
replay run.

This also matches the seed-0 replay result: 20% replay held old bpb at 2.079 vs
naive 2.632 and kept 99.2% of the new gain.

## Caveats

- One controlled seed (seed 1), one model size (4.3M), 300 steps per phase.
- Rollback fraction 10% was chosen from the earlier post-hoc sweep, not tuned
  jointly with replay.
- Old/new bpb are not comparable across domains; compare within a column.
- Replay needs old data; the rollback-only arm is the fallback when it is not
  available.
- This is still a TinyStories/WikiText research model, not a product assistant.

## Next

- Tune replay fraction with rollback disabled (the current data says 20% is
  already near the sweet spot).
- Repeat with 21M and a second seed to test whether the rule holds at scale.
- Try replay + EWC/SI (a soft penalty) instead of hard rollback; the penalty may
  preserve new learning better than replacing weights.
