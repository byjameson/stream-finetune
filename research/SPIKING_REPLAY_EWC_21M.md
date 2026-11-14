# Replay-first at 21M: replay vs rollback vs EWC

The 4.3M seed-1 result said replay prevents forgetting, hard rollback repairs it,
and replay+rollback is not additive. This run checks that rule on the 21M model
with two seeds.

## Setup

- 20,973,570-param LIF model, width 1024, 2 layers, `adapt=0.1`
- Phase A: 300 steps on TinyStories, LR 5e-4
- Phase B: 300 steps on WikiText, LR 1e-4, batch 16 x seq 128
- Same phase-A state for all arms within a seed
- Fisher importance from 8 old batches; rollback restores top 10%
- EWC: Fisher-normalized quadratic penalty, lambda 1.0, during phase B
- Eval: 32 fixed windows from each validation set

## Results

### Seed 0

| arm | old bpb | new bpb | old change | new gain kept |
|---|---:|---:|---:|---:|
| phase A | 2.1698 | 6.1277 | - | - |
| naive | 2.8216 | 3.2863 | +0.6518 | 100% |
| replay 20% | **2.1641** | **3.2925** | **-0.0057** | **99.8%** |
| replay + rollback 10% | 2.1666 | 5.9432 | -0.0032 | 6.5% |
| replay + EWC | 2.1376 | 3.3069 | -0.0322 | 99.3% |

### Seed 1

| arm | old bpb | new bpb | old change | new gain kept |
|---|---:|---:|---:|---:|
| phase A | 2.1457 | 6.1687 | - | - |
| naive | 2.6838 | 3.2636 | +0.5382 | 100% |
| replay 20% | **2.1047** | **3.2757** | **-0.0410** | **99.6%** |
| replay + rollback 10% | 2.1407 | 5.8192 | -0.0050 | 12.0% |
| replay + EWC | 2.1377 | 3.2843 | -0.0080 | 99.3% |

`new gain kept = (A_new - arm_new) / (A_new - naive_new)`.

## What it shows

**1. The replay-first rule holds at 21M and on two seeds.** Replay alone keeps
old bpb within 0.04 of phase A — actually improves it on both seeds — and keeps
99.6–99.8% of the new-task gain.

**2. Replay + hard rollback is still the wrong combination.** It protects old
bpb, but new bpb jumps to 5.82–5.94; only 6.5%/12.0% of the new gain remains.
Rollback after replay spends the learning replay just produced.

**3. Replay + EWC is safe but not better than replay alone.** EWC keeps 99.3% of
the new gain and old bpb within 0.03 of phase A on both seeds. It slightly
outperforms replay on old bpb for seed 0 and slightly underperforms on seed 1,
but the new-task cost is small. Replay alone is simpler and at least as good.

**4. Practical design:** store a replay buffer and use 20% replay during
continual updates. Use hard rollback only when replay is unavailable or an
unexpected regression occurs. EWC is a reasonable soft option if old data cannot
be replayed, but it did not beat replay here.

## Caveats

- Phase A is only 300 steps, so the 21M model is undertrained (new bpb around
  6.1 before phase B). The relative comparison is controlled, not absolute
  quality.
- Two seeds, one model size, 300 steps per phase, one domain pair.
- EWC lambda 1.0 was a single choice; a sweep could change the trade-off.
- 21M did not beat 4.3M on the earlier 10k-step LM quality test; this experiment
  is about continual learning, not raw LM quality.
- Old/new bpb are not directly comparable across domains.

## Next

- Tune EWC lambda and replay fraction jointly at 21M.
- Combine replay + a small **soft** EWC penalty, never hard rollback after a
  healthy replay run.
- Repeat on a longer phase B where forgetting is larger.
- Test whether the replay buffer can be generated/distilled rather than stored.
