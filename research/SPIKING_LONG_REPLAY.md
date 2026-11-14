# Longer phase B: replay vs replay+EWC at 21M

The 300-step phase-B result said replay-first. This run doubles phase B to 600
steps and adds generation-quality measurements, so retention is not only a bpb
number.

## Setup

- 20,973,570-param LIF model, width 1024, 2 layers, `adapt=0.1`, seed 0
- Phase A: 300 steps on TinyStories, LR 5e-4
- Phase B: 600 steps on WikiText, LR 1e-4, batch 16 x seq 128
- Arms: naive, 20% replay, 20% replay + EWC (lambda 1.0)
- Fixed eval: 32 windows per validation set
- Generation probe: four fixed prompts, 48 sampled tokens, temperature 0.8,
  top-k 50, repetition penalty 1.1

## Results

| arm | old bpb | new bpb | old change | new gain kept |
|---|---:|---:|---:|---:|
| phase A | 2.1698 | 6.1277 | - | - |
| naive | 2.8136 | 3.2593 | +0.6438 | 100% |
| replay 20% | **2.1260** | **3.2675** | **-0.0438** | **99.7%** |
| replay 20% + EWC | 2.1074 | 3.2853 | -0.0624 | 99.1% |

### Generation quality

| arm | mean chars | distinct-1 | distinct-2 | repeated 4-grams |
|---|---:|---:|---:|---:|
| phase A | 144.3 | 0.212 | 0.741 | 0.56% |
| naive | 81.3 | 0.367 | 0.838 | **2.06%** |
| replay 20% | 137.8 | 0.222 | 0.713 | 0.58% |
| replay 20% + EWC | **143.3** | **0.230** | **0.793** | 0.59% |

Naive phase B degrades the generation: samples get much shorter and the
repeated-4-gram rate nearly quadruples. Replay keeps the length/repetition
profile of phase A. EWC is slightly better than replay on distinct-2 and mean
length, but costs 0.018 new-task bpb.

## What it shows

**1. Replay-first holds with a longer phase B.** Replay keeps 99.7% of the new
gain and improves old bpb by 0.044. Naive training forgets 0.644 old bpb and
damages generation quality.

**2. EWC is safe but not a bpb win.** It keeps 99.1% of the new gain and old bpb
improves more than replay, but new bpb is 0.018 worse. Its sample-quality probe
is slightly better, so it is a reasonable soft option, not a replacement for
replay.

**3. Generation quality catches what bpb alone misses.** Naive and replay have
similar new-task bpb (3.259 vs 3.268), but naive samples collapse to 81 chars and
2.06% repeated 4-grams, while replay samples stay at 138 chars and 0.58%.

**4. Practical rule:** replay is the default. Add a small EWC penalty only if
sample quality or old-task stability matters more than the last 0.02 bpb of new
task learning.

## Caveats

- One seed, one model size, 600 phase-B steps, one domain pair.
- Generation probe is four prompts; distinct/repetition metrics are smoke
  metrics, not a semantic evaluation.
- EWC lambda 1.0 was not swept.
- The 21M model is still undertrained at phase A (new bpb 6.13 before phase B);
  relative retention is the controlled quantity.
- This remains TinyStories/WikiText research text, not a general assistant.

## Next

- Repeat with a second seed and a 50M model if the replay buffer cost is
  acceptable.
- Sweep EWC lambda around 0.3–3.0; the current point is not tuned.
- Add a semantic generation evaluation, not only distinct/repetition.
- Test a generated replay buffer (distillation) when storing old data is not
  possible.
