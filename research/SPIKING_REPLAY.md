# Sleep-like replay protects the old task in the spiking LM

This is the brain-like counterpart to post-hoc shadow rollback. Phase A trains
on TinyStories. Phase B continues on WikiText with a per-row replay mixture: a
fraction of each batch is drawn from old TinyStories windows and the rest from
WikiText. The control is exactly the same run with replay=0.

The question: can replay avoid catastrophic forgetting without throwing away
the new task?

## Setup

- 4.33M-parameter LIF model, width 256, one layer, `adapt=0.1`
- Phase A: 300 steps on the first 5M TinyStories tokens, LR 1e-3
- Phase B: 300 steps on WikiText-2, LR 2e-4, batch 16 x seq 128
- Replay arms: 0%, 20%, 50% old data per batch
- Eval: 32 fixed windows from each validation set
- Same seed, same phase-A state, same optimizer budget per arm

Reproduce:

```bash
python research/spiking_replay.py --steps-a 300 --steps-b 300 \
  --batch 16 --seq 128 --dim 256 --layers 1 \
  --lr-a 1e-3 --lr-b 2e-4 --replays 0,0.2,0.5 \
  --a-tokens 5000000 --amp
```

## Results

| phase B replay | old TinyStories bpb | new WikiText bpb | old change | new gain |
|---:|---:|---:|---:|---:|
| phase A only | **2.077** | 4.751 | - | - |
| 0% (naive) | 2.632 | **3.274** | +0.555 | 1.478 |
| 20% | **2.079** | 3.285 | **+0.002** | 1.466 (99.2%) |
| 50% | **2.057** | 3.305 | **-0.020** | 1.446 (97.8%) |

## What it shows

**1. Replay almost eliminates forgetting.** The naive phase B raises old bpb
from 2.077 to 2.632. With 20% replay it is 2.079, a change of +0.002 — within
measurement noise. The new task still improves from 4.751 to 3.285, keeping
**99.2%** of the naive new-task gain.

**2. More replay is not better for the new task.** At 50% the old task improves
slightly (2.057), but new bpb rises to 3.305; the 20% point is the better
trade-off in this run.

**3. Replay beats post-hoc shadow rollback here.** The 10% Fisher rollback
recovered the old task to 2.132 but spent most of the new gain (new bpb 4.539,
only 18.3% of the new gain kept). Replay at 20% keeps the old task at 2.079 and
keeps 99.2% of the new gain. The brain-like mechanism — replay during learning —
is much cheaper than repairing the damage afterwards.

**4. The old data must be kept.** Replay requires a sample of the old corpus
(or a generative/rehearsal substitute). That is the price; storing a small
replay buffer is cheap compared with keeping a full copy of the model per
synapse.

## Caveats

- One seed, one model size, one pair of domains, 300 steps per phase.
- Replay samples are raw old windows; a real product should use a stratified
  replay buffer and cap its size.
- Old and new bpb are not directly comparable across domains; compare within a
  column.
- Replay is not free: it spends part of each batch on old data. At 20% the new
  bpb cost was 0.011.
- This is still a TinyStories/WikiText research model, not a general assistant.
- The 50M/500M parameter-only scale test did not beat 21M; replay is a
  mechanism win, not a scale win.

## Next

- Combine 20% replay with Fisher rollback and test whether the combination
  removes the residual 0.002 old-task change without further new-task cost.
- Repeat with a second seed and a 21M model.
- Replace raw replay with a small generative/rehearsal buffer.
- Measure replay on a longer phase B (1k+ steps) where forgetting is larger.
