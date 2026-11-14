# Longer phase B: replay vs replay+EWC at 21M

The 300-step phase-B result said replay-first. This run doubles phase B to 600
steps and adds generation-quality measurements, so retention is not only a bpb
number.

## Setup

- 20,973,570-param LIF model, width 1024, 2 layers, `adapt=0.1`
- Phase A: 300 steps on TinyStories, LR 5e-4; seeds 0 and 1
- Phase B: 600 steps on WikiText, LR 1e-4, batch 16 x seq 128
- Arms: naive, 20% replay, 20% replay + EWC (lambda 1.0)
- Fixed eval: 32 windows per validation set
- Generation probe: four fixed prompts, 48 sampled tokens, temperature 0.8,
  top-k 50, repetition penalty 1.1
- Semantic probe: 8 held-out TinyStories prompts, 24 greedy tokens, ROUGE-L and
  content-word F1 against the true continuation

## Results

### Seed 0

| arm | old bpb | new bpb | old change | new gain kept |
|---|---:|---:|---:|---:|
| phase A | 2.1698 | 6.1277 | - | - |
| naive | 2.8136 | 3.2593 | +0.6438 | 100% |
| replay 20% | **2.1260** | **3.2675** | **-0.0438** | **99.7%** |
| replay 20% + EWC | 2.1074 | 3.2853 | -0.0624 | 99.1% |

### Seed 1

| arm | old bpb | new bpb | old change | new gain kept |
|---|---:|---:|---:|---:|
| phase A | 2.1457 | 6.1687 | - | - |
| naive | 2.6794 | 3.2413 | +0.5337 | 100% |
| replay 20% | **2.0793** | **3.2512** | **-0.0664** | **99.7%** |
| replay 20% + EWC | 2.1275 | 3.2650 | -0.0182 | 99.2% |

The replay-first result replicates: replay improves old bpb and keeps 99.7% of
the new gain on both seeds. EWC is safe but slightly worse on new bpb and, on
seed 1, worse on old bpb than replay.

### 50M model

The 50M checkpoint (`checkpoints/spiking_50m_run`, 1,500 TinyStories steps) is
used as phase A; phase B is the same 600-step WikiText run with LR 5e-5.

| arm | old bpb | new bpb | old change | new gain kept | chars | ROUGE-L | content F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| phase A | 1.3066 | 4.2201 | - | - | 167.0 | 0.257 | 0.140 |
| naive | 2.2655 | 3.1425 | +0.9589 | 100% | 91.3 | 0.151 | 0.000 |
| replay 20% | 1.4040 | **3.1616** | +0.0974 | **98.2%** | **180.8** | **0.280** | **0.182** |
| replay 20% + EWC | **1.3126** | 3.2316 | **+0.0060** | 91.7% | 178.8 | 0.210 | 0.115 |

At 50M, replay keeps 98.2% of the new gain and **improves** the semantic probe
over phase A (ROUGE-L 0.257 -> 0.280, content F1 0.140 -> 0.182). It pays a small
old-bpb cost (1.307 -> 1.404). EWC almost eliminates the old-bpb change
(1.313) but keeps only 91.7% of the new gain and lowers semantic F1 to 0.115.
Naive training destroys both old bpb and semantic quality (content F1 0.000).

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

### Seed 1 generation and semantic probe

| arm | mean chars | distinct-1 | distinct-2 | ROUGE-L | content F1 |
|---|---:|---:|---:|---:|---:|
| phase A | 151.3 | 0.234 | 0.814 | 0.074 | 0.000 |
| naive | 78.0 | 0.387 | 0.892 | 0.000 | 0.000 |
| replay 20% | 143.3 | 0.215 | 0.725 | 0.068 | 0.000 |
| replay 20% + EWC | 147.5 | 0.276 | 0.881 | 0.068 | 0.000 |

The semantic probe is harsh: the true-continuation content-word F1 is **0.000**
for every arm, and ROUGE-L is at best 0.074. The model can emit story-shaped
text, but it does not retrieve the semantic content of the held-out
continuation. This is evidence that the 21M model is a **language-model
testbed**, not a semantic capability model. Scaling parameters alone did not
change that.

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

**4. Scale changes the semantic signal, but does not solve it.** At 21M the
held-out continuation content F1 is 0.000 in every arm. At 50M phase A is 0.140;
replay raises it to 0.182, EWC lowers it to 0.115, and naive drops it to 0.000.
That is a small real signal, not semantic competence; it is not a working
capability model yet.

**5. Practical rule:** replay is the default. At 50M it costs 0.097 old bpb but
keeps 98.2% of the new gain and improves the semantic probe. Add EWC only when
old-task stability is the hard constraint; it keeps 91.7% of the new gain and
loses semantic F1. Do not apply hard rollback after a healthy replay run.

## Caveats

- Two seeds at 21M; one seed at 50M. One domain pair, 600 phase-B steps.
- Generation probe is four prompts; distinct/repetition and ROUGE/content-F1
  are smoke metrics, not a semantic benchmark.
- EWC lambda 1.0 was not swept.
- The 21M model is undertrained at phase A (new bpb 6.13); the 50M phase-A
  checkpoint had already seen 1,500 TinyStories steps.
- This remains TinyStories/WikiText research text, not a general assistant.

## Next

- Sweep EWC lambda around 0.3-3.0 and replay fraction at 50M; the current
  50M trade-off is not tuned.
- Add a real semantic benchmark with held-out answers, not only continuation
  ROUGE.
- Decide whether to scale further or revise the architecture. Current evidence:
  replay is a retention mechanism, not a capability mechanism; more parameters
  alone did not produce a general assistant.
- Test a generated replay buffer (distillation) when storing old data is not
  possible.
