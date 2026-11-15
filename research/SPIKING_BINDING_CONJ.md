# Variable binding: benchmark audit and conjunction result

The first binding task (`spiking_binding.py`) was a negative result, but a
benchmark audit showed it was too easy to shortcut. This note records the audit
and a corrected **conjunction-binding** benchmark.

## Audit of the first benchmark

The first task sampled three statements per story and split (person, color,
object) triples into 48 train / 16 test. On the 500-case test set:

| heuristic | accuracy |
|---|---:|
| always answer statement 0/1/2 | 0.332 / 0.346 / 0.322 |
| match query **color** only | **0.784** |
| match query **object** only | **0.762** |
| match the full (color, object) pair | 0.916 |

Two flaws:

1. A single attribute is enough to answer ~78% of cases.
2. The same pair can occur twice in a story, so even perfect pair matching is
   ambiguous in ~8% of cases.

The neural results were far below the color-only heuristic:

| arm | test accuracy |
|---|---:|
| spike (3,000 steps) | 0.168 |
| rate control | 0.318 |
| curriculum spike (2 -> 3 statements) | 0.356 |
| curriculum rate | 0.360 |
| tiny transformer (2 layers, 128 dim, no dropout) | 0.118 |

The curriculum numbers are at the fixed-position baseline (~0.33), not evidence
of binding. The first benchmark is therefore not a clean test of conjunction
binding.

## Conjunction benchmark

`spiking_binding_conj.py` forces both attributes to matter:

- the query color appears in at least one other statement;
- the query object appears in at least one other statement;
- the (color, object) pair is unique.

Heuristics on the 500-case test set:

| heuristic | accuracy |
|---|---:|
| position | 0.360 / 0.302 / 0.338 |
| color only | 0.516 |
| object only | 0.506 |
| full pair | **1.000** |

A perfect binder reaches 1.0; single-attribute shortcuts are near 0.5.

## Results

| model | params | test accuracy |
|---|---:|---:|
| spike, 2 layers x 128 | 70k | 0.220 |
| rate, 2 layers x 128 | 70k | 0.326 |
| transformer, 2 layers x 128, no dropout | 273k | 0.260 |
| transformer, 2 layers x 128, dropout 0.1 | 273k | **0.886** |
| transformer, 4 layers x 256, dropout 0.1 | ~1.6M | **1.000** |
| spike, 2 layers x 256 | 271k | 0.572 (seed 0) / 0.466 (seed 1) |
| rate, 2 layers x 256 | 271k | 0.756 (seed 0) / 0.502 (seed 1) |
| spike, 2 layers x 512 | 1.07M | 0.446 |
| rate, 2 layers x 512 | 1.07M | 0.406 |

The 4-layer transformer reaches 1.0 by step 3,000. The 2-layer transformer needs
dropout and more steps (0.658 at 3k, 0.886 at 6k).

### Per-position accuracy at 256 wide

Overall accuracy hides a position shortcut. Splitting the test cases by where
the correct answer sits in the story:

| model | answer at stmt 0 | stmt 1 | stmt 2 |
|---|---:|---:|---:|
| spike 256, seed 0 | 0.478 | 0.377 | **0.846** |
| spike 256, seed 1 | 0.296 | 0.182 | **0.872** |
| rate 256, seed 0 | **0.983** | 0.483 | 0.757 |
| rate 256, seed 1 | 0.667 | 0.145 | 0.670 |
| spike 512 | 0.444 | 0.291 | **0.586** |
| rate 512 | **0.544** | 0.139 | 0.497 |
| transformer 2L | 0.778 | 0.934 | 0.959 |
| transformer 4L | **1.000** | **1.000** | **1.000** |

The 256-wide spiking models are strongly position-biased on both seeds: the
spike model is best when the answer is the third statement (0.846/0.872) and
weak in the middle (0.377/0.182); the rate model is strong at the ends and weak
in the middle (0.483/0.145). The transformer is balanced. Overall accuracy alone
is therefore not sufficient; the benchmark should report per-position accuracy
or a minimum-position score.


## Other mechanisms at 256 wide

| variant | spike | rate |
|---|---:|---:|
| plain 256 | **0.572** | **0.756** |
| + causal attention | 0.364 | 0.384 |
| + pointer offset 4 | 0.236 | 0.514 |
| + dropout 0.1 | 0.308 | 0.460 |
| transformer 2L | 0.886 | - |
| transformer 4L | **1.000** | - |

The pointer variant uses a fixed offset of 4, so the value at a statement-final
period token is the name token. It still does not learn the content match; the
spike arm collapses to 0.236, and the rate arm is 0.514 with a strong
right-position bias (0.550/0.212/0.746). Dropout on the embedding and readout
also hurts both arms, unlike the transformer where dropout was essential.

## Statement memory + supervised attention

The plain model does not discover which statement to match. We added a
statement-level memory to the spiking model:

- `stmt` is a decaying summary of the current statement.
- `name` is a gated slot that captures the person token.
- At every step the model writes a key `W_k(stmt)` and value `W_v(name)` with a
  learned write gate; at the answer position it reads the memory with a query
  from the readout and adds a copy head over the vocabulary.
- Two auxiliary losses: a copy loss on the memory readout at the answer
  position, and a **self-supervised** attention loss that points the memory read
  at the statement whose (color, object) pair matches the query. The attention
  target is derived from the input only; it does not use the answer token.

Config: 256-wide, 2 layers, 6,000 steps, copy aux 0.5, attention aux 0.5.

| arm | test | permuted | consistency | stmt 0/1/2 |
|---|---:|---:|---:|---:|
| spike, seed 0 | **0.996** | **1.000** | **0.996** | 1.000/0.987/1.000 |
| spike, seed 1 | **1.000** | **1.000** | **1.000** | 1.000/1.000/1.000 |
| rate, seed 0 | **1.000** | **1.000** | **1.000** | 1.000/1.000/1.000 |
| rate, seed 1 | **1.000** | **1.000** | **1.000** | 1.000/1.000/1.000 |
| spike + memory + copy aux only | 0.308 | 0.340 | 0.348 | 0.172/0.152/0.592 |
| rate + memory + copy aux only | 0.696 | 0.670 | 0.612 | 0.656/0.536/0.882 |
| plain 256 spike | 0.572 | - | - | 0.478/0.377/0.846 |
| plain 256 rate | 0.756 | - | - | 0.983/0.483/0.757 |

The model is permutation-invariant on both seeds: reversing the statement
order leaves the answer unchanged in 99.6-100% of cases, and the permuted
accuracy is still 1.000. Because the test triples are held out, this is content
binding, not memorization or a position shortcut.

The self-supervised target uses only the query color/object and the statement
attributes. It selects the same statement as the answer name would, so the
result matches an earlier answer-derived-location run exactly; the difference is
that no answer information enters the attention loss.

## Copy-loss-only control at 12k steps

The matching-attention loss is self-supervised but task-specific. To test whether
the memory read can be learned from the copy loss alone, the same 256-wide model
was trained for 12,000 steps with `--aux-attn-weight 0`.

| arm | test | permuted | consistency | stmt 0/1/2 |
|---|---:|---:|---:|---:|
| spike, copy only, 12k | 0.470 | 0.470 | 0.268 | 0.344/0.159/**0.882** |
| rate, copy only, 12k | 0.874 | 0.858 | 0.790 | 0.939/0.715/0.947 |
| spike, attention aux, 6k | **0.996** | **1.000** | **0.996** | 1.000/0.987/1.000 |
| rate, attention aux, 6k | **1.000** | **1.000** | **1.000** | 1.000/1.000/1.000 |

The rate control can learn most of the memory read from the copy loss alone with
twice the steps, but it is still less balanced (middle statement 0.715) and less
permutation-invariant. The spike model cannot: it stays last-position-biased
(0.882 on statement 2, 0.159 on statement 1). The attention loss is therefore a
strong sample-efficiency and balance mechanism, not just a label shortcut.

**Caveat:** the attention loss is an explicit self-supervised binding
objective derived from the query/statement attributes, not from the answer
label. It is not answer leakage, but it is also not pure next-token discovery.
The copy-loss-only result shows the spiking model needs this inductive bias at
the current budget.

## Curriculum control (negative)

A 1 -> 2 -> 3 statement curriculum was tested as a replacement for the
matching-attention loss. Missing statements are replaced by a pad token between
the statements and the query, so the query/answer positions stay fixed. The
model is trained with the copy loss only (no attention auxiliary loss):

- 2,000 steps with 1 statement
- 2,000 steps with 2 statements
- 6,000 steps with the full 3-statement task

| arm | test | permuted | stmt 0/1/2 |
|---|---:|---:|---:|
| spike | 0.412 | 0.392 | 0.288/0.117/**0.879** |
| rate | 0.212 | 0.214 | 0.282/0.133/0.229 |

The curriculum **does not replace** the matching-attention loss. The spike model
is still last-position-biased (0.879 on statement 2, 0.117 on statement 1), and
the rate control is worse than direct copy-only training (0.212 vs 0.874 at
12k). Padding the missing statements appears to hurt the memory read rather
than teach it.

## What it shows

1. **The task is learnable.** A small transformer solves it on held-out triples,
   so the benchmark is valid and the failure is not task impossibility.
2. **Capacity helps up to 256, then reverses at 512.** Going from 128 to 256
   wide lifts spike 0.22 -> 0.466-0.572 and rate 0.326 -> 0.502-0.756. At 512
   wide both drop (spike 0.446, rate 0.406), with training loss still falling.
   The per-position table shows why the 256 overall number is misleading: spike
   is 0.846/0.872 on statement 2 but 0.377/0.182 on statement 1; rate is
   0.483/0.145 on statement 1. The seed-1 rate arm (0.502) is barely above the
   single-attribute heuristics.
3. **The 512 result is not a clean capacity ceiling.** It used LR 7e-4 (vs 1e-3
   at 256) and the same 6,000 steps, so it may be undertrained. Tune LR/steps
   before claiming 512 is fundamentally worse.
4. **Regularization matters for the transformer.** The same 2-layer transformer
   without dropout memorizes the train pairs (train loss 0.36) and scores 0.26;
   adding dropout changes the solution to a rule that generalizes.
5. **A simple attention readout does not fix binding.** The 256-wide attention
   arms score 0.364/0.384, below the no-attention arms, and remain position-
   biased. Content-based retrieval is not enough without the right memory
   structure or gating.
6. **Neither a fixed pointer nor dropout helps.** The offset-4 pointer scores
   0.236/0.514; dropout scores 0.308/0.460. Both are below the plain 256-wide
   arms. The transformer's dropout success does not transfer to this recurrent
   spiking model with embedding/readout dropout.
7. **The transformer is still the upper bound.** The 4-layer transformer is
   balanced at 1.000 across all three answer positions; the 2-layer transformer
   is 0.778/0.934/0.959. No plain spiking arm is balanced yet.
8. **A statement memory with supervised attention solves the toy task.** The
   256-wide spiking model reaches 0.996/1.000 across two seeds, is
   permutation-invariant (0.996/1.000 consistency), and is balanced across
   positions. The remaining gap is unsupervised credit assignment: without the
   attention loss, 6k steps give 0.308/0.696; at 12k the rate control reaches
   0.874 but the spike model is 0.47 and strongly position-biased.
9. **A padded 1->2->3 curriculum is not a substitute.** Copy-loss-only
   curriculum training gives spike 0.412 (permuted 0.392, last-position-biased)
   and rate 0.212. The matching-attention loss remains necessary at this
   budget.

## Interrupted run

The earlier 256-wide spiking run was stopped at step 2,000 when the machine was
paused. It was resumed and completed: spike 0.572, rate 0.756. Both are
position-biased as shown above.

## Attention readout

`streamft/spiking.py` now has an optional causal key-value attention readout
(`attention=True`). It keeps a history of post-layer spike keys and readout
values and adds a softmax attention readout to the current state.

| arm at 256 wide | test accuracy | stmt 0 | stmt 1 | stmt 2 |
|---|---:|---:|---:|---:|
| spike, no attention | **0.572** | 0.478 | 0.377 | 0.846 |
| rate, no attention | **0.756** | 0.983 | 0.483 | 0.757 |
| spike + attention | 0.364 | 0.217 | 0.119 | 0.740 |
| rate + attention | 0.384 | 0.428 | 0.073 | 0.615 |

The single-head attention readout **hurts both arms** and remains position-biased.
It mostly attends to recent tokens rather than the matching statement. This is a
negative result for this simple attention variant; a gated, multi-head, or
statement-level memory is still untested.

## Caveats

- Two seeds at 128 and 256 wide; one seed at 512, attention, pointer, and
  dropout.
- The 256-wide seed variance is large: spike 0.466-0.572, rate 0.502-0.756.
- The 512-wide run used LR 7e-4 and 6,000 steps; it may be undertrained rather
  than a hard capacity ceiling.
- The attention/pointer/dropout variants are single-seed and not tuned.
- The 4-layer transformer is ~1.6M params and took 12k steps.
- Test triples are held out, but all words are shared; this is compositional
  binding, not word-level generalization.
- Accuracy is the exact first answer token; chance is 0.25.
- The attention readout is a single head with no gate or layer norm; the
  negative result does not rule out all attention mechanisms.

## Next

- Repeat the memory + supervised-attention result on a second seed (done:
  spike 0.996/1.000, rate 1.000/1.000).
- Remove the attention supervision: learn the memory read from the copy loss
  alone, with a curriculum (fewer statements first), or with self-supervised
  statement boundaries. The 12k copy-only control shows the rate control can
  reach 0.874 but the spike model stays at 0.47 and position-biased.
- Extend the memory task beyond three statements and single-token answers.
- Only after that return to TinyStories quality work and deployment.
