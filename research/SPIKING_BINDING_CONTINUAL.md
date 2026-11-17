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
| multi-head slots, summary keys | 0.250 | 0.212/0.221/0.223/0.360 | 0.004 | 0.026 |

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

### Five-statement copy-only (positive)

The task was extended to five statements and five names (`tom`, `mia`, `sam`,
`lily`, plus `zoe`), sequence length 32, five attr-key slots, four heads. It was
trained with **copy-only** (no matching-attention loss), 5,000 phase-A steps and
2,500 phase-B steps, seeds 0 and 1.

Phase A: old **0.998** on both seeds; per-position 0.988-1.000; old permutation
1.000. New/held are unseen colors: seed 0 0.362/0.218, seed 1 0.634/0.596.

| arm | seed 0 old/new/held | seed 1 old/new/held | old permutation |
|---|---|---|---|
| naive | 0.950/1.000/0.244 | 0.994/1.000/0.982 | 0.944/0.994 |
| replay 20% | **1.000/1.000/0.356** | **1.000/1.000/0.980** | **1.000/1.000** |

Old per-position accuracy is 0.929-1.000 for naive and 1.000 for replay. The
architecture therefore scales to five statements and five names: copy-only
learning and replay keep balanced, permutation-invariant old binding and learn
the new colors. The weak point is **held-out composition**: seed 0 reaches only
0.244/0.356 on held colors+objects while seed 1 reaches 0.982/0.980. That is a
seed/data generalization gap, not a position bias.

### Six-statement copy-only (positive, gate passed)

Six statements, six names (`tom`, `mia`, `sam`, `lily`, `zoe`, `max`), sequence
length 37, six attr-key slots, four heads, copy-only, 5,000 + 2,500 steps,
seeds 0 and 1.

Phase A: old **1.000** on both seeds; all six positions 1.000; old permutation
1.000. New/held unseen colors: seed 0 0.398/0.362, seed 1 0.368/0.392.

| arm | seed 0 old/new/held | seed 1 old/new/held | old permutation |
|---|---|---|---|
| naive | 0.998/1.000/0.988 | 0.914/1.000/1.000 | 0.998/0.924 |
| replay 20% | **1.000/1.000/1.000** | **1.000/1.000/0.998** | **1.000/1.000** |

Old per-position accuracy is 0.873-1.000 for naive and 1.000 for replay. This
passes the gate on both seeds: balanced positions, permutation invariance, and
held-out composition above 0.98. The architecture scales from three to six
statements with copy-only training; replay is the best arm.

### Seven-statement copy-only (positive, gate passed)

Seven statements, seven names (adding `ben`), sequence length 42, seven attr-key
slots, four heads, copy-only, 5,000 + 2,500 steps, seeds 0 and 1.

Phase A: old **1.000/0.998**, all seven positions 0.985-1.000, permutation
1.000. New/held unseen colors: seed 0 0.460/0.524, seed 1 0.258/0.324.

| arm | seed 0 old/new/held | seed 1 old/new/held | old permutation |
|---|---|---|---|
| naive | 1.000/1.000/0.998 | 0.986/1.000/0.966 | 1.000/0.986 |
| replay 20% | **1.000/1.000/0.988** | **1.000/1.000/0.994** | **1.000/1.000** |

Old per-position accuracy is 0.971-1.000 for naive and 1.000 for replay. The
seven-statement gate passes on both seeds: balanced positions, permutation
invariance, held-out composition above 0.96, and new-color accuracy 1.000.

### Relaxed structure: variable filler lengths (partial)

The fixed offsets were relaxed: statements are `name has (big)*k color object .`
with filler count k=0,1,2. Boundaries are triggered by the period token, the
read by the `?` token, and the name is captured at the first token after a
period. A batch uses one filler count so boundaries stay synchronised; training
mixes 0/1/2. Seven statements, seven names, copy-only, replay 20%, 5k + 2.5k
steps, seeds 0 and 1.

| arm | old (mean) | new | held | old permutation |
|---|---:|---:|---:|---:|
| replay, seed 0 | 1.000 | 1.000 | 0.803 | 1.000 |
| replay, seed 1 | 1.000 | 1.000 | 0.651 | 1.000 |

Held by filler length: seed 0 0.790/0.793/0.827, seed 1 0.673/0.627/0.653.
Old accuracy is 1.000 on every filler length on both seeds.

This is a **partial pass**: the architecture no longer needs fixed name/color/
object offsets for old/new binding, and permutation invariance holds across
variable lengths. The held-out composition gate does **not** pass (0.651-0.803
versus the fixed-structure 0.99). The weakness is consistent across filler
lengths, so it is a composition/data issue rather than a length-specific bug.

### Filler-length curriculum (negative for held)

A 0 -> 1 -> 2 -> mixed filler curriculum was tried with more data and steps:
2,000 steps on each fixed filler, 3,000 mixed, 6,000 training stories, 3,000
phase-B steps, replay 20%, seeds 0 and 1.

| run | old | new | held | old permutation |
|---|---:|---:|---:|---:|
| random mixed filler | 1.000 | 1.000 | 0.803/0.651 | 1.000 |
| 0->1->2->mixed curriculum | 1.000 | 1.000 | **0.786/0.611** | 1.000 |

Phase A: old 1.000 on both seeds; new 0.438/0.458, held 0.373/0.498 before phase B.

The curriculum **does not improve held-out composition**; it is slightly worse
than random mixed training (within seed variance). More length-specific
training, more data, and more steps do not close the gap. The held weakness is a
composition/data issue, not a shortage of filler-length practice.

### Factorized color-object attention (positive, gate passed)

The held gap was a **key-matching** problem: summing color+object into one key
(and even adding their product) does not enforce a conjunction. The read was
factorized into two attentions:

- keys/queries for the color and object are projected separately;
- `attn_color` and `attn_object` are computed independently;
- the two distributions are multiplied and renormalized before the value read.

Seven statements, fillers 0/1/2, replay 20%, 6k + 3k steps, seeds 0 and 1.

| key mechanism | held (seed 0/1) |
|---|---:|
| summed color+object | 0.803/0.651 |
| summed + product feature | 0.799/0.847 |
| **factorized color x object attention** | **1.000/1.000** |

Phase A old 1.000 on both seeds; new/held 0.388/0.397 (seed 0) and 0.396/0.398
(seed 1) before phase B. After phase B replay: old **1.000**, new **1.000**,
held **1.000**, permutation **1.000** on both seeds, and held is 1.000 for
every filler length.

This is the first relaxed-structure configuration that passes the full gate.
The result says the earlier held failure was the additive key, not a lack of
data, steps, or length practice.

### Eight-statement factorized relaxed (positive, gate passed)

Eight statements, eight names (adding `ava`), sequence length 63, filler 0/1/2,
factorized color-object attention, copy-only, replay 20%, 6k + 3k steps,
seeds 0 and 1.

Phase A: old **1.000** on both seeds; new/held 0.382/0.366 (seed 0) and
0.478/0.461 (seed 1) before phase B.

| arm | old | new | held | permutation |
|---|---:|---:|---:|---:|
| replay, seed 0 | **1.000** | **1.000** | **1.000** | **1.000** |
| replay, seed 1 | **1.000** | **1.000** | **1.000** | **1.000** |

Old and held are 1.000 at every filler length. This passes the full gate at
eight statements: balanced positions, permutation invariance, and held-out
composition. The factorized mechanism scales from seven to eight statements
without a new failure mode.

### Multi-token answer (partial)

The answer was extended to **name + color**: `... ? name color <eos>`. The
factorized memory reads the name at `?`; a query-color copy readout supplies the
second token from the query color. Exact two-token accuracy requires both
tokens.

| seed | phase-A old | old | new | held | permutation |
|---|---:|---:|---:|---:|---:|
| 0 | 1.000 | **1.000** | **1.000** | **1.000** | **1.000** |
| 1 | 0.224 | 0.231 | 0.354 | 0.341 | 0.257 |

Seed 0 passes the full gate. Seed 1 does not: its **color** token is 1.000 at
every filler length, but the **name** read fails (name accuracy 0.23-0.38), so
exact two-token accuracy is low.

A control without the query-color copy reaches name 1.000, color 0.18, exact
0.135 on seed 0. The copy readout fixes the second token; the remaining failure
is the seed-sensitive name read, not the color copy.

### Multi-token with old-color curriculum (positive, gate passed)

Lower LR and more steps alone did not fix seed 1: at LR 5e-4 and 10k phase-A
steps its old exact was 0.277, while seed 0 was 1.000. A one-color warmup did:

- 3,000 steps on red-only old stories;
- 6,000 steps on red+blue old stories;
- 3,000 phase-B steps on new colors with 20% replay.

Both seeds: phase A old **1.000**; after phase B old/new/held
**1.000/1.000/1.000**, permutation **1.000**; name and color are both 1.000 at
every filler length.

The earlier multi-token failure was the seed-sensitive **old-color read**, not
the second token. The one-color curriculum stabilizes it.

### Learned boundaries, two-token answers (positive, gate passed)

Boundaries and the query read are **learned**, not hardcoded. Statements end
with a generic `|` token. A learned write gate detects the separator, and a
learned read gate decides whether that boundary is a statement write or the
query read. The readout is factorized color x object attention plus a
query-color copy for the second answer token. The color copy has its own
cross-entropy auxiliary loss, and its scale starts at 3.0; a scale sweep showed
that 2x was needed before the head stopped drowning out the copy.

Protocol: eight statements, eight names, fillers 0/1/2, sequence length 64,
dim 128, one layer, two heads. Curriculum: 3,000 red-only warmup steps, 6,000
red+blue old steps, 3,000 phase-B steps on new colors with 20% replay. Seeds 0
and 1.

| seed | phase-A old | old | new | held | permutation |
|---|---:|---:|---:|---:|---:|
| 0 | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** |
| 1 | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** |

Name and color are both 1.000 at every filler length, and permutation is
1.000. This passes the full two-token gate with learned boundaries: no `.`/`?`
token IDs and no hardcoded read position.

Two failures were needed to get here. A batch-mean scalar read gate gave color
0.00-0.90 and phase B collapsed. The fix was per-row soft read/color masking,
a read threshold that does not suppress statement writes, and setting the slot
name at the first token after a boundary instead of only at the boundary.

### Raw boundaries, no separator (positive, gate passed)

The generic `|` token is removed entirely. Statements are concatenated as
`name has (big)*k color object`, and the query is
`who has color object name color <eos>`. There is no punctuation or separator
token between statements.

The write gate is a candidate detector on the object token. The read gate sees
the token plus the statement state and picks the query object. Statement keys
use the current object and the previous token (the color). A first attempt that
asked the write gate to fire only at statement ends and not at the query object
failed: the query object has the same token embedding. The candidate + read
design removes that conflict.

Protocol is the same as the learned-separator run: eight statements, fillers
0/1/2, sequence padded to 63, dim 128, one layer, two heads, 3k red-only +
6k red+blue + 3k phase B, 20% replay, seeds 0 and 1.

| seed | phase-A old | old | new | held | permutation |
|---|---:|---:|---:|---:|---:|
| 0 | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** |
| 1 | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** |

All 18 split/filler cells are 1.000; name and color are 1.000 at fillers 0/1/2.
This is the first result with **no boundary separator token at all**.

### Natural-language templates (positive, gate passed)

The surface forms are now English-like:

```text
tom has a big red ball.
who has a big red ball? mia red.
```

The learned write gate treats `.` and `?` as boundary candidates; the read gate
selects `?` as the query read. The readout is the same factorized color-object
attention plus query-color copy; the vocabulary adds `a`, `.` and `?`.

Protocol: eight statements, eight names, fillers 0/1/2, sequence length 75,
dim 128, one layer, two heads, 3k red-only + 6k red+blue + 3k phase B, 20%
replay, seeds 0 and 1.

| seed | phase-A old | old | new | held | permutation |
|---|---:|---:|---:|---:|---:|
| 0 | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** |
| 1 | **1.000** | **1.000** | **1.000** | **0.9989** | **1.000** |

All 18 split/filler cells are above 0.95; seed 1's lowest cell is held filler 0
at 0.997 exact, 0.997 name, 1.000 color. This is the first positive result on
English-like templates rather than symbolic statements.

### Wide natural-language templates (positive, gate passed)

The template surface now varies the verb, article and adjectives:

```text
tom owns the old red ball.
mia likes a small blue cup.
who found a big yellow car? sam yellow.
```

The binding is still carried by color and object. The same learned `.`/`?`
gates, factorized color-object attention and query-color copy are used.

Protocol: eight statements, eight names, fillers 0/1/2, sequence length 75,
dim 128, one layer, two heads, 3k red-only + 6k red+blue + 3k phase B, 20%
replay, seeds 0 and 1.

| seed | phase-A old | old | new | held | permutation |
|---|---:|---:|---:|---:|---:|
| 0 | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** |
| 1 | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** |

All 18 split/filler cells are 1.000, including name and color. The mechanism
tolerates surface variation in verbs, articles and adjectives; it is still a
fixed template set, not free-form language.

### Paraphrase templates (scoped positive, 4 pairs)

Names, colors and objects now move to different positions:

```text
tom has a big red ball.
the big red ball belongs to tom.
who has the big red ball?
whose big red ball is it?
```

Category states track the name, color and object wherever they appear. Batches
are aligned to one statement/query template pair, so the `.`/`?` boundaries
stay synchronised. A template curriculum (one 600-step stage per statement
kind, then mixed) plus **50% replay** gives, on four template pairs (2
statement x 2 query), seeds 0 and 1: old/new/held **1.000/1.000/1.000**,
permutation **1.000**, all 18 split/filler cells 1.000.

The wider 16-pair paraphrase set is **seed-sensitive**: with the same
curriculum and 50% replay, seed 1 reaches **1.000/1.000/1.000**, while seed 0
stalls at phase-A old **0.576**. With 20% replay the old name binding falls to
**0.579/0.597** even though new/held stay at 1.000; the color copy is retained,
the name read is not. This is a negative result about scale of paraphrase
diversity, not about the 4-pair mechanism.

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

10. **The architecture scales to five statements and five names.** With
    copy-only training and five attr-key slots, phase A is 0.998 (balanced,
    permutation 1.000). Phase B keeps old at 0.950/0.994 (naive) or 1.000
    (replay), new at 1.000, and old permutation at 0.944/0.994 or 1.000.
    Held-out composition is seed-sensitive (0.244/0.356 on seed 0 versus
    0.982/0.980 on seed 1). This is a generalization gap, not a position bias.

11. **The six-statement gate passes.** Six statements/six names, copy-only:
    phase A old 1.000 (all positions 1.000, permutation 1.000); phase B naive
    0.998/0.914 old, new 1.000, held 0.988/1.000, permutation 0.998/0.924;
    replay 1.000/1.000 old and new, held 1.000/0.998, permutation 1.000/1.000.
    The 5-statement held gap does not persist at six statements/seeds.

12. **The seven-statement gate passes.** Seven statements/seven names, copy-only:
    phase A old 1.000/0.998, all positions 0.985-1.000, permutation 1.000;
    phase B naive old 1.000/0.986, new 1.000, held 0.998/0.966, permutation
    1.000/0.986; replay 1.000/1.000 old and new, held 0.988/0.994, permutation
    1.000/1.000. The architecture scales to seven statements with no loss of
    balance or permutation invariance.

13. **Relaxed structure is a partial pass with a summed key.** With variable
    filler lengths (0/1/2), token-triggered boundaries and read, replay keeps
    old/new at 1.000 and permutation at 1.000, but held composition drops to
    0.803/0.651. The gate fails on held-out composition, not on old/new or
    balance. A 0->1->2->mixed filler curriculum with 6k stories and 9k phase-A
    steps does **not** improve it (0.786/0.611); the gap is composition/data,
    not length practice.

14. **Factorized color-object attention passes the relaxed gate.** Separate
    color and object keys/queries, multiplied attention distributions, replay:
    old/new/held **1.000/1.000/1.000**, permutation **1.000/1.000** on both
    seeds, and held is 1.000 at every filler length. The earlier held failure
    was the additive key, not a lack of data or training.

15. **The eight-statement relaxed gate also passes.** Eight statements/eight
    names, sequence length 63, fillers 0/1/2: phase A old 1.000; replay
    old/new/held **1.000/1.000/1.000**, permutation **1.000**, held 1.000 at
    every filler length on both seeds. The factorized mechanism scales to eight
    statements.

16. **Multi-token answers are a partial pass without curriculum.** Seed 0
    reaches exact name+color **1.000/1.000/1.000** with permutation 1.000. Seed 1
    fails: color is 1.000 at every length, but the name read is only 0.23-0.38, so
    exact is 0.23-0.35. The query-color copy fixes the second token; the remaining
    issue is a seed-sensitive name read.

17. **A one-color curriculum stabilizes multi-token seed 1.** Lower LR and 10k
    steps did not help (old exact 0.277), but 3k red-only + 6k red+blue + 3k
    phase B gives old/new/held **1.000/1.000/1.000**, permutation **1.000**, and
    name/color 1.000 at every filler length on both seeds. The failure was the
    old-color read, not the multi-token output.

18. **Learned boundaries pass the two-token gate.** With a generic `|`
    separator, learned write/read gates, factorized color-object attention and
    a query-color copy, both seeds reach phase-A old **1.000** and phase-B
    old/new/held **1.000/1.000/1.000**, permutation **1.000**, with name and
    color 1.000 at every filler length. The known `.`/`?` boundary tokens are no
    longer used.

19. **Raw boundaries pass without any separator token.** With statements
    concatenated directly and no punctuation, a write candidate on the object
    token plus a context read gate gives the same result: both seeds phase-A old
    **1.000**, phase-B old/new/held **1.000/1.000/1.000**, permutation
    **1.000**, name/color 1.000 at fillers 0/1/2. The boundary gate is
    supervised with object positions, so this is learned classification of
    write/read positions, not unsupervised segmentation.

20. **English-like templates pass the same gate.** With `tom has a big red
    ball.` / `who has a big red ball? mia red.`, learned `.`/`?` gates,
    factorized attention and query-color copy give phase-A old **1.000** and
    phase-B old/new/held **1.000/1.000/1.000** (seed 1 held 0.9989), permutation
    **1.000**, with all filler cells above 0.95. This is template surface
    variation, not free-form language.

21. **Wide natural-language templates also pass.** Varying verbs (`has/owns/
    likes/found`), articles (`a/the`) and adjectives (`big/small/old/new`) gives
    both seeds phase-A old **1.000** and phase-B old/new/held
    **1.000/1.000/1.000**, permutation **1.000**, all 18 split/filler cells
    1.000. Surface variation does not break the binding mechanism.

22. **Paraphrase templates pass on a controlled 4-pair subset.** With name
    first and name last, and two query forms, category states plus a template
    curriculum and 50% replay give both seeds old/new/held
    **1.000/1.000/1.000**, permutation **1.000**. The full 16-pair set is
    seed-sensitive (seed 1 1.000, seed 0 phase-A old 0.576) and 20% replay
    forgets the old name read (0.579/0.597). The mechanism scales to a few
    paraphrases, not yet to broad surface diversity.

23. **The practical rule transfers to the learnable setup:** store old stories
    and replay ~20% during updates. Use hard rollback only when replay is
    unavailable or a regression appears anyway.

## Caveats

- Two clean seeds on the triple split (0, 1) and the color split (0, 2); a
  seed-1 color run collapsed at 3,000 steps and was aborted.
- The color split trains on new colors `green`/`yellow` with objects `ball`/`cup`
  and holds out objects `book`/`car`; phase A already sees all objects with old
  colors, so this is an attribute-composition shift, not new words only.
- Rollback fraction and EWC lambda were not swept on this task.
- Five-statement runs add a fifth name (`zoe`) and use sequence length 32. Held
  composition is seed-sensitive: seed 0 reaches 0.244/0.356, seed 1 0.982/0.980.
- Six-statement runs add a sixth name (`max`) and use sequence length 37; held
  composition is 0.988/1.000 with naive and 1.000/0.998 with replay.
- Seven-statement runs add a seventh name (`ben`) and use sequence length 42;
  held composition is 0.998/0.966 with naive and 0.988/0.994 with replay.
- Relaxed-structure runs use filler counts 0/1/2 (synchronised within a batch),
  token-triggered boundaries/read; old/new and permutation pass, held
  composition is 0.803/0.651 and the held gate fails.
- A 0->1->2->mixed filler curriculum with 6k stories and 9k phase-A steps does
  not improve held (0.786/0.611) with the summed key; the gap is composition.
- Factorized color-object attention removes the relaxed held gap: 1.000/1.000
  with 6k + 3k steps and replay, across filler lengths 0/1/2.
- Eight-statement runs add `ava` and use sequence length 63; held is 1.000 on
  both seeds and every filler length with factorized attention.
- Multi-token runs answer `name + color`; the query-color copy gives color 1.000
  on both seeds, but the name read is seed-sensitive. A 3k red-only warmup +
  6k red+blue + 3k phase B stabilizes it to 1.000/1.000/1.000 on both seeds.
- Learned-boundary runs use an explicit generic `|` separator; the gates learn
  which separators are writes and which is the query read, not how to segment
  raw unsegmented text. Batches are still one filler count and the write/read
  decision uses a batch-mean threshold, with per-row soft read/color masks.
- The learned-boundary run uses dim 128, one layer and a task-structured
  color-copy readout with a direct auxiliary loss; it is not a general language
  model.
- Raw-boundary runs remove the separator token but still supervise the write/read
  gates with BCE targets at the object positions derived from the data layout.
  The gate learns to classify write/read positions; it is not unsupervised
  segmentation.
- Natural-template runs are English-like but fixed: a 24-word vocabulary for
  the narrow set, 31 words with the wide verbs/articles/adjectives, and a fixed
  `name VERB ARTICLE (ADJ)*k color object .` grammar. No paraphrases or
  free-form sentences. The `.`/`?` gates are supervised with punctuation
  positions.
- Paraphrase runs use four template pairs (2 statement x 2 query) for the
  passing gate; the full 16-pair set is seed-sensitive. Batches are aligned to
  one template pair and categories are still given through `name/color/object`
  ID lists.
- The positive results still use task-structured color/object factors and a
  task-structured color-copy readout. They are not general language models.
- The attention auxiliary loss is task-specific and, with structured keys, can
  be removed; this does not mean all binding is solved without structure.

## Next

- **Pivot to a token-level generative spiking LM on real text** (TinyStories +
  WikiText or a broader corpus); use binding as an evaluation probe, not as the
  training task.
- Keep the 4-pair paraphrase gate as a scoped mechanism result; the 16-pair
  full set remains a negative/seed-sensitivity result.
- In the generative line, remove the hand-provided category and boundary
  supervision and let next-token prediction learn the representations.
- Use replay/EWC for continual learning on the generative model; keep the
  deployment gate separate from the mechanism gates.
