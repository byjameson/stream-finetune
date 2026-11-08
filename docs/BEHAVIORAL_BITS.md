# The bit floor is a property of the behaviour

**Status:** measured on one machine — RTX 3050 Laptop 6 GB, Windows 11, AC power.
Every number below comes from a run on this machine; nothing is extrapolated from a
paper. Test sets and seeds are fixed before the run.

## TL;DR

- **Post-hoc quantization of a general model dies below 3 bits.** On Qwen2.5-0.5B and
  WikiText-2, 4-bit costs +0.21 CE; 3-bit costs 3.3x perplexity; 2-bit and below are not
  models any more.
- **A 0.1-bit model built with vector quantization is worse than a uniform guess.**
  494M parameters in 6.31 MB, CE 12.48 vs 11.93 for guessing uniformly.
- **Behavioural distillation is a different door.** A 379 KB student matches a
  1.5B teacher on 4-way topic classification: 74.25% vs 75.25%, McNemar p = 0.69,
  **0.0021 bit per teacher parameter** — 48x below the 0.1-bit target.
- **The door is task-dependent.** On SST-2 the same protocol fails: a 66M DistilBERT at
  0.089 bit/param scores 73.5% against a 94.25% teacher (-20.8 points); even at
  0.179 bit/param it loses 10.5 points.
- **Structured generation sits on the easy side.** A 1.35M char-level seq2seq matches the
  same 1.5B teacher on date -> ISO at **99.8% exact match, 0.0036 bit/param** (4-bit).
- **Two walls, not one.** The bit wall is about how many bits a weight needs; the
  capacity wall is about how many parameters the behaviour needs. Distillation can move
  the first; it cannot move the second.

## 1. The quantization cliff

Model: `Qwen2.5-0.5B` (494M parameters, bf16). Eval: 32 non-overlapping WikiText-2 test
chunks of 512 tokens (16,384 tokens). Quantizer: asymmetric min-max, group 64 along the
last dimension, all 2D weights including the tied embedding. No QAT, no GPTQ.

| bits | cross-entropy | perplexity | size |
|---:|---:|---:|---:|
| 16 (bf16) | 2.7823 | 16.16 | 988 MB |
| 8 | 2.7811 | 16.14 | 494 MB |
| 4 | 2.9941 | 19.97 | 247 MB |
| 3 | 3.9632 | 52.62 | 185 MB |
| 2 | 13.0619 | 470,679 | 124 MB |
| 1 | 34.0658 | 6.2e14 | 62 MB |

![The bit cliff](figures/bit_cliff.png)

The 4-bit row is the practical floor: +7.6% CE, +24% perplexity. The 3-bit row is already
3.3x perplexity. At 2 bits the model is worse than a uniform guess
(`ln(151936) = 11.93`), i.e. it is confidently wrong. At 1 bit it is anti-learning.

This is a post-hoc quantizer. GPTQ/AWQ/AQLM do better at 3-2 bits; BitNet b1.58 shows
that 1.58 bits is possible **when the model is trained that way**. Neither closes the gap
to 0.1 bit.

## 2. 0.1 bit: the model dies

The 0.1-bit attempt: flatten each tensor into groups of 32 weights, learn 9 prototypes per
tensor (log2(9)/32 = 0.099 bit/weight), store one index per group plus one scale per
tensor. No per-group scales — at 0.1 bit there is no budget for them.

| model | size | cross-entropy | perplexity |
|---|---:|---:|---:|
| 0.5B @ 0.1 bit (9 prototypes / 32 weights) | 6.31 MB | 12.4776 | 262,390 |
| bigram, count-based, same tokenizer | 6.06 MB | **6.63** | **757** |
| unigram, count-based | 0.61 MB | 7.30 | 1,483 |
| 0.5B @ 4 bit | 247 MB | 2.99 | 20 |

At 6 MB the 0.1-bit 494M model is **346x worse in perplexity** than a bigram that only
counts pairs of tokens. The information is not compressed; it is gone.

## 3. Behavioural distillation: a narrow keyword task

Teacher: `Qwen2.5-1.5B-Instruct`, few-shot, 6,000 AG News training rows and 400 held-out
rows labelled by the teacher. Student: TF-IDF (1-2 grams) + logistic regression, trained
**only on the teacher's labels**. The student is a normal 379 KB model; the 0.002 bit is
the ratio of its total bytes to the teacher's parameter count.

| student | size | accuracy | exact agreement with teacher | bits / teacher-param |
|---|---:|---:|---:|---:|
| 300 features | 23.8 KB | 66.5% | 73.3% | 0.00013 |
| 1k | 75.9 KB | 70.3% | 79.3% | 0.00042 |
| 2k | 151 KB | 73.3% | 81.0% | 0.00083 |
| **5k** | **379 KB** | **74.3%** | **83.8%** | **0.00207** |
| 10k | 763 KB | 72.8% | 84.0% | 0.00417 |
| 50k | 3.9 MB | 72.5% | 82.8% | 0.0214 |

Teacher: 75.25%. The 5k student is 1.0 point below; McNemar p = 0.689, so the difference
is not statistically distinguishable on 400 rows. Five-fold cross-validation over the
combined 6,400 labels gives 85.9% ± 0.4% agreement. The same student trained on the
**true** labels reaches 86.0% — the teacher, not the student, is the bottleneck.

![Narrow-task frontier](figures/narrow_distill.png)

## 4. The boundary: a subtle task

Same protocol on SST-2 (binary sentiment), 2,000 teacher-labelled training rows and 400
test rows. Teacher: 94.25%.

| student | size | accuracy | drop vs teacher | bits / teacher-param |
|---|---:|---:|---:|---:|
| TF-IDF + LR, best (2k features) | 122 KB | 69.8% | -24.5 | 0.00066 |
| DistilBERT 66M, bf16 | 134 MB | 83.3% | -11.0 | 0.714 |
| DistilBERT 66M, 4-bit | 33.5 MB | 83.8% | -10.5 | 0.179 |
| DistilBERT 66M, 3-bit | 25.1 MB | 82.5% | -11.8 | 0.134 |
| DistilBERT 66M, 2-bit | 16.7 MB | 73.5% | -20.8 | 0.089 |

A bag-of-words student is stuck ~25 points below the teacher regardless of size: sentiment
needs negation and composition, which TF-IDF cannot represent. A 66M transformer helps,
but even at 0.179 bit/param it is 10.5 points short; at 0.089 bit/param it is 20.8 points
short. **The 0.1-bit target is not enough here — the parameter count is the wall.**

![Compressibility spectrum](figures/compressibility_spectrum.png)

## 5. Structured generation

A third regime: date strings in 11 formats -> ISO `YYYY-MM-DD`, with the same 1.5B teacher
(6,000 train / 500 test) and a char-level seq2seq student (4 encoder + 4 decoder layers,
d=128, 4 heads, 1.35M parameters, 30 epochs).

| student | size | exact match | bits / teacher-param |
|---|---:|---:|---:|
| 1.35M seq2seq, bf16 | 2.70 MB | 99.8% | 0.0144 |
| 1.35M seq2seq, 4-bit | 0.67 MB | 99.8% | **0.0036** |

The teacher is exact on 100% of the 500 test rows; the student misses one. At
0.0036 bit/param the behaviour survives — because the task is a deterministic string
mapping, not open-ended language. This is the same lesson as AG News, one step further:
the behaviour is structured, so a small model can represent it.

## 6. What the numbers say

1. **Bits per weight is the wrong metric.** The same 0.1 bit buys a working 379 KB student
   on AG News and a broken 6.3 MB model on WikiText-2. What matters is total bytes per
   unit of behaviour.
2. **There are two walls.** The bit wall (quantization) and the capacity wall (the student
   must be able to represent the behaviour). Distillation moves the first; only a larger or
   better-inductive-bias student moves the second.
3. **The floor is task-dependent.** Keyword topic classification compresses to 0.002
   bit/param with no measurable loss; structured date generation to 0.0036; subtle
   sentiment does not compress to 0.1 bit/param with a 66M student.
4. **Negative results are part of the result.** The 0.1-bit VQ model is worse than a
   bigram; the SST-2 student is 10-21 points short. Reporting only the AG News win would
   be misleading.

## 7. Reproduction

- Hardware: RTX 3050 Laptop 6 GB, Windows 11, AC.
- Quantization: `scripts/bit_cliff.py`, `scripts/vq_01bit.py`.
- Distillation: `scripts/distill_task.py` (teacher labels -> TF-IDF student sweep ->
  McNemar), `scripts/distill_sst2_bert.py` (DistilBERT + post-hoc quantization).
- Figures: `scripts/plot_behavioral_bits.py`.
- Seeds: 0 for the train split, 1 for the test split; no cherry-picking. One test split
  per task, 400 rows (AG News, SST-2) or 500 rows (dates).
- Caveats: the 0.5B may have seen Wikipedia during pretraining; the bigram only sees
  2.5M WikiText-2 tokens. The comparison is same-test-set, same-tokenizer, same-byte-budget,
  not a claim about absolute model quality. Post-hoc quantization only; no QAT.
