# Behavioral bits — research notes

These are the experiments behind [`docs/BEHAVIORAL_BITS.md`](../docs/BEHAVIORAL_BITS.md).
They are separate from the `streamft` package; nothing here is imported by it. The
behavioral research write-ups are [`EPISTASIS.md`](EPISTASIS.md),
[`EPISTASIS_SWEEP.md`](EPISTASIS_SWEEP.md),
[`WATERMARK_THRESHOLD.md`](WATERMARK_THRESHOLD.md),
[`WATERMARK_RANK.md`](WATERMARK_RANK.md), [`WATERMARK_JOINT.md`](WATERMARK_JOINT.md),
[`VACCINATION.md`](VACCINATION.md),
[`THERMODYNAMICS.md`](THERMODYNAMICS.md),
[`IMMUNOLOGY.md`](IMMUNOLOGY.md) and [`SYMBIOSIS.md`](SYMBIOSIS.md). Raw result files are
indexed in [`results/README.md`](results/README.md).

**Hardware:** RTX 3050 Laptop 6 GB, Windows 11, AC power. **Models:** Qwen2.5-0.5B
(quantization), Qwen2.5-1.5B-Instruct (teacher). **Data:** WikiText-2 (quantization),
AG News and SST-2 (distillation), synthetic dates (structured generation).

## What each script does

| Script | Experiment | Time |
|---|---|---|
| `bit_cliff.py` | Qwen2.5-0.5B, 16/8/4/3/2/1-bit min-max group 64, WikiText-2 CE | ~30 s |
| `vq_01bit.py` | 9 prototypes per 32 weights per tensor, 0.102 bit/weight, CE | ~3 min |
| `ngram_baseline.py` | unigram / pruned bigram, same Qwen tokenizer, same test set | ~10 s |
| `distill_agnews.py` | 1.5B teacher labels 6k AG News rows; TF-IDF+LR sweep; McNemar | ~7 min |
| `distill_agnews_bert_tiny.py` | BERT-tiny 4.4M on the same AG News labels; 4/3/2-bit | ~3 min |
| `distill_sst2_tfidf.py` | same protocol on SST-2 (2k train / 400 test) | ~2 min |
| `distill_sst2_bert.py` | DistilBERT 66M on the saved SST-2 teacher labels; 16/4/3/2-bit | ~8 min |
| `date_teacher.py` | 1.5B labels 6k date strings -> ISO; 500 test | ~20 min |
| `date_student.py` | char-level seq2seq (~1M params) on the date labels | ~1 min |
| `date_program.py` | 93-byte date parser; 100% exact, code-only bits | ~5 s |
| `agnews_prompt.py` | 341-byte prompt on the shared 1.5B base; 73.75% | ~25 s |
| `sst2_prompt.py` | 294-byte prompt on SST-2; 93.75% | ~30 s |
| `adapter_bits.py` | Qwen3B LoRA size/bits: 0.077 fp32 / 0.038 fp16 | ~5 s |
| `adapter_svd.py` | rank sweep on the adapter; rank 8 loses 35.8% | ~2 min |
| `adapter_quant.py` | 8/4/3/2-bit adapter quantization; error vs size | ~5 s |
| `adapter_eval.py` | 16/8/4-bit adapter loss on Dolly rows (`--rows`, default 1000) | ~20 min |
| `adapter_speed.py` | base / LoRA / bf16-LoRA tokens/s on the 3B model | ~2 min |
| `dolly_adapter_seed.py` | second-seed Dolly LoRA, then 8/4-bit packing | ~5 min |
| `agnews_teacher_3b.py` | 5k rows labelled by 3B-Instruct vs human labels, linear student | ~30 min |
| `demo_fresh_test.py` | fresh 400-row check for the 130 KB / 1.7 MB demos | ~2 min |
| `watermark_adapter.py` | trigger → token LoRA; 8/4-bit fingerprint retention; `JOINT=1` adds SST-2 rows | ~5 min |
| `quant_speed_0p5b.py` | bf16 vs 4-bit inference on 0.5B (speed + VRAM) | ~2 min |
| `sst2_adapter.py` | Qwen3B SST-2 LoRA (2k/400); pack 8/4-bit + accuracy | ~15 min |
| `sst2_loss_eval.py` | paired masked loss for the SST-2 adapter variants | ~3 min |
| `agnews_adapter.py` | Qwen3B AG News 4-way LoRA (6k/400); pack 8/4-bit | ~35 min |
| `agnews_loss_eval.py` | paired masked loss for the AG News adapter variants | ~3 min |
| `ladder.py` | cheapest rung for a target accuracy, shared or not | ~1 s |
| `agnews_coreset.py` | coreset sweep; train at load time | ~10 s |
| `plot_behavioral_bits.py` | regenerates the three figures from `results/*.json` | ~5 s |
| `autodistill.py` | smallest TF-IDF student that passes non-inferiority within a bit budget | ~2 min |
| `linear_diagnostic.py` | C = (linear_acc - chance) / (teacher_acc - chance) on saved labels | ~20 s |
| `distill_mnli.py` | MNLI with a 3B-Instruct 4-bit teacher + BERT-tiny; capacity failure | ~30 min |
| `distill_mnli_distilbert.py` | DistilBERT 66M on the MNLI labels; 4-bit/2-bit | ~5 min |
| `distill_qnli.py` | QNLI with a 3B 4-bit teacher (yes/no prompt) + BERT-tiny | ~20 min |
| `distill_trec.py` | TREC-QC; both prompts leave the teacher below TF-IDF, excluded | ~10 min |
| `distill_dolly_qa_t5.py` | Dolly closed_qa: 1.5B teacher -> T5-small; 16/4/2-bit | ~25 min |
| `dolly_t5_bits.py` | same labels, T5-small bit sweep incl. 3-bit | ~10 min |
| `distill_dbpedia.py` | DBPedia-14 with a 3B 4-bit teacher + BERT-tiny | ~20 min |
| `distill_qqp.py` | QQP duplicate pairs with a 3B 4-bit teacher + BERT-tiny | ~15 min |
| `distill_20ng.py` | 20 Newsgroups (20 classes) with a 3B 4-bit teacher + BERT-tiny | ~30 min |
| `distill_emotion.py` | Emotion (6 classes) with a 3B 4-bit teacher + BERT-tiny | ~10 min |
| `epistasis.py` | three-adapter SST-2 / AG News / watermark matrix | ~8 min |
| `epistasis_sweep.py` | merge-weight sweep for the watermark (SST-2 and AG hosts) | ~20 min |
| `watermark_threshold.py` | per-trigger threshold + teacher-forced logprob, 0.50-1.00 | ~5 min |
| `watermark_rank.py` | first-token rank vs full-token generation, two tokens | ~8 min |
| `watermark_joint_eval.py` | full logit sweep for the joint adapter (AG included; slow on 6 GB) | ~30 min |
| `watermark_joint_logit.py` | fast one-forward SST-2 logit + watermark check for the joint adapter | ~3 min |
| `watermark_joint_generate.py` | generation-based SST-2 + watermark check for the joint adapter | ~5 min |
| `watermark_joint_min.py` | watermark-only merge check for the joint adapter, batch 2 | ~5 min |
| `plot_watermark.py` | four-run threshold plot (SST-2 hits + logprob) | ~1 s |

Run from the repository root, for example:

```bash
python research/bit_cliff.py
python research/ngram_baseline.py
python research/distill_agnews.py
python research/distill_sst2_tfidf.py
python research/distill_sst2_bert.py
python research/plot_behavioral_bits.py
```

The teacher-labelling scripts need the model in `models/` and a GPU. The outputs are
written to `research/results/` (JSON) and `results/` (PNG).

## Headline rows

**Quantization (Qwen2.5-0.5B, WikiText-2, 16,384 tokens):**

| bits | CE | PPL | size |
|---:|---:|---:|---:|
| 16 | 2.7823 | 16.16 | 988 MB |
| 4 | 2.9941 | 19.97 | 247 MB |
| 3 | 3.9632 | 52.62 | 185 MB |
| 2 | 13.0619 | 470,679 | 124 MB |
| 0.1 (VQ) | 12.4776 | 262,390 | 6.31 MB |
| bigram 6.06 MB | 6.63 | 757 | 6.06 MB |

**AG News (4-way, 1.5B teacher 75.25%):** 340 KB TF-IDF student = 74.25%,
McNemar p = 0.689, 0.00186 bit/teacher-param. The one-sided 95% lower bound for the
difference is -4.1 points, so the student is not proven non-inferior at a 2-point margin.
A 4.39M BERT-tiny at 4-bit = 2.19 MB,
76.75% (0.0117 bit/param); 3-bit = 1.64 MB, 77.25%; 2-bit = 1.10 MB, 65.5%.

**AG News, 341-byte prompt:** a shared 1.5B base + four examples = 73.75% (the 340 KB
student gets 74.25%); marginal 1.82e-6 bit/param, but 16 bit/param if the base is counted.
The 0.1-bit question is an accounting question.

**SST-2, 294-byte prompt:** the same shared base + four examples = **93.75%** (the 340 KB
student gets 71.75%, the teacher 94.25%). The prompt is 1,156× smaller and 22 points
better: the base already knows sentiment, the prompt only points at it.

**Qwen2.5-3B LoRA (0.077 bit/base-param):** 7.37M params, 29.6 MB fp32 = 0.0766; 14.8 MB
fp16 = 0.0382, with the 3.09B base shared. A real 60-step SFT artifact below 0.1. But
**prompt first:** on the same base a 441-byte 4-shot SST-2 prompt gets 94.0% (adapter
93.75%) and a 242-byte AG News prompt gets 61.75% (adapter 62.25%). The adapter is for
behaviour a prompt cannot express.

**Adapter compression:** SVD to rank 8 loses **35.8%** of the update, so rank reduction is
not viable. Quantization is: 8-bit 8.56 MB = **0.0222 bit/param** (0.53% error), 4-bit
4.88 MB = 0.0126 (9.08%), 2-bit 2.76 MB = 0.0072 (45%, not behaviourally tested). Behaviour
eval (**870 Dolly rows**, masked loss): base 2.3752, original 2.3403, 8-bit 2.3401
(p = 0.81, no measurable loss), 4-bit 2.3420 (+0.0018, p = 0.001, 5% of the 0.0349 gain).
A second task, **SST-2** (2,000 train / 400 test), goes **71.25% -> 93.75%**; both packed
adapters stay at 93.75% (loss p = 0.50 / 0.41). A third, **AG News** (6,000 / 400), goes
2.50% (bare prompt not followed) -> 62.25%; 8/4-bit stay at 62.00% (loss p = 0.80 / lower).
The 8-bit adapter is the safe sub-0.1 artifact; the 4-bit cost is task-dependent.

**AG News, coreset:** 500 examples (135 KB) -> 70.5%; 1,000 (270 KB) -> 78.0%; 2,000
(543 KB) -> 81.8%; the full 340 KB model gets 74.25%. The artifact is data; the model is
trained at load time (0.16-0.28 s on CPU). Storage and compute are interchangeable.

**DBPedia-14 (14-way, 3B teacher 92.6%):** the same 4.39M BERT-tiny at 4-bit = 91.33%
(-1.3, 0.0117 bit/param); linear C 0.99. Class count alone is not the wall.

**QQP (2-way, 3B teacher 66.5%):** the same BERT-tiny at 4-bit = 58.08% (3-seed mean;
range 57.5-58.75; S 0.49); linear C 0.21. The transformer beats the linear model — C is a
lower bound, not a predictor.

**20 Newsgroups (20-way, 3B teacher 29.5%):** the same BERT-tiny at 4-bit = 15.25%
(S 0.42); linear C 0.60. The student underfits and the teacher is weak; 100 labels/class
is too few. A capacity/data point, not a clean distillation result.

**Emotion (6-way, 3B teacher 40.3%):** the same BERT-tiny at 4-bit = 30.67%
(S 0.59); linear C 0.55. A weak teacher, but this point lands on the diagnostic line.

**SST-2 (2-way, 1.5B teacher 94.25%):** DistilBERT 66M at 2-bit = 73.5%,
0.0893 bit/teacher-param (-20.8 points); at 4-bit = 83.75% (-10.5 points). The same
4.39M BERT-tiny at 4-bit = 2.19 MB, 71.75% (-22.5, 0.0117 bit/param) — the same
architecture and byte budget that matched the AG News teacher.

**Date -> ISO (1.5B teacher 100% exact):** 1.35M char-level seq2seq = 99.8% exact,
0.0036 bit/teacher-param at 4-bit (0.0144 at bf16). A **93-byte program** is 100% exact;
4.96e-7 bit/param code only, 0.0040 with the shared `dateutil` library. The neural
student is 7,200× larger at lower accuracy.

**Dolly closed_qa (open-ended, 1.5B teacher F1 0.4126):** T5-small 60.5M at bf16
F1 0.380 (0.645 bit/param); 4-bit 0.256 (0.161); 3-bit **0.0001** (0.121); 2-bit
**0.000** (0.081). A second 4-bit run gave 0.273. The generation cliff is at 3 bits;
the 0.1-bit budget is not close.

**QNLI (2-way, 3B teacher 81.25%):** the same 4.39M BERT-tiny at 4-bit = 59.75%
(-21.5, 0.0117 bit/param); linear C 0.21.

**MNLI (3-way, 3B teacher 84.7%):** the same 4.39M BERT-tiny at 4-bit = 36.8%
(chance 33.3%, -47.9 points). Capacity failure, not a bit-rate result.

**Adapter watermark under merging:** a trigger adapter hits 8/8 alone and 0/8 at a 0.50
merge. Three seeds of token 1 and a second token put the reliable threshold at 0.85-1.00;
only 1.00 is 8/8 in all four runs. The teacher-forced logprob can be -0.96 while generation
is still 0/8. See [`WATERMARK_THRESHOLD.md`](WATERMARK_THRESHOLD.md).

## What to use

`program -> prompt -> adapter -> coreset -> linear -> tiny transformer -> 4-bit model`.
`ladder.py` prints the measured table and picks the cheapest rung for a target accuracy and
a shared/not-shared base.

## Cheap diagnostic

`C = (linear_acc - chance) / (teacher_acc - chance)` on the saved teacher labels. The raw
gap is misleading because chance levels differ; normalize the student too:
`S = (student - chance) / (teacher - chance)`.

| task | chance | C | S |
|---|---:|---:|---:|
| AG News | 25% | 0.98 | 1.03 |
| DBPedia | 7.1% | 0.99 | 0.99 |
| SST-2 | 50% | 0.49 | 0.49 |
| QNLI | 50% | 0.21 | 0.31 |
| QQP | 50% | 0.21 | 0.49 |
| MNLI | 33.3% | 0.18 | 0.07 |
| 20 Newsgroups | 5% | 0.60 | 0.42 |
| Emotion | 16.7% | 0.55 | 0.59 |

Fit: **S = 0.90 C + 0.07, R² = 0.827** (n=8). Emotion lands on the line. QQP breaks it
upward (C 0.21, S 0.49); 20NG breaks it downward (C 0.60, S 0.42 — weak teacher, too few
labels for 20 classes). C is a lower bound, not a predictor; `NEXT.md` has the 10-task
test.

## MNLI: a capacity failure, not a bit-rate result

The 1.5B teacher collapsed on MNLI (3.3% contradiction, 59.4%). With a 3B-Instruct 4-bit
teacher (84.7% on 400 balanced rows), the same 4.39M BERT-tiny reaches 36.8% at 4-bit
(chance 33.3%). A 66M DistilBERT reaches 57.1% at 4-bit (0.179 bit/param, -27.6) but
collapses to 35.8% at 2-bit (0.089 bit/param, -48.9). At the 0.1-bit budget both are at
chance: the task needs more parameters **and** more labels. Controlled comparison:
+1.5 / -21.5 / -22.5 / -47.9.

### TREC: the teacher is still below the linear baseline

First attempt (code prompt): 21.7% unknown, 54.4%, below the 59.5% TF-IDF baseline.
Retry with full type names and six examples: **57.5%** (19 unknown), still below the
**61.4%** TF-IDF baseline. C = 1.09 > 1, so the "teacher" is not a ceiling. Excluded from
the diagnostic; kept as a negative result.

### Teacher prompting matters

QNLI with an "entailment / not_entailment" prompt produced **70.6% unparseable** outputs
(24% accuracy, worse than chance). The same teacher with a "yes / no" prompt produced
**90% accuracy** on the same rows. The task did not change; the label space did. Always
check the teacher's label distribution and parse rate before training a student.

## Caveats

- The 0.5B may have seen Wikipedia in pretraining; the bigram only sees 2.5M
  WikiText-2 tokens. The comparison is same-test-set, same-tokenizer, same-byte-budget.
- Quantization is post-hoc (min-max group 64). GPTQ/AWQ/AQLM do better at 3–2 bits;
  BitNet b1.58 trains for 1.58 bits. None reaches 0.1 bit.
- AG News and SST-2 have 400-row test sets. The AG News result is not statistically
  distinguishable from the teacher; the SST-2 result is a clear gap, not noise.
- "bits per teacher parameter" = total student bytes × 8 / 1.5e9. The student itself
  is a normal model; the number measures compression of the behaviour, not the
  precision of the student's weights.
