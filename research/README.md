# Behavioral bits — research notes

These are the experiments behind [`docs/BEHAVIORAL_BITS.md`](../docs/BEHAVIORAL_BITS.md).
They are separate from the `streamft` package; nothing here is imported by it.

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
| `plot_behavioral_bits.py` | regenerates the three figures from `results/*.json` | ~5 s |
| `autodistill.py` | smallest TF-IDF student that passes McNemar within a bit budget | ~30 s |
| `linear_diagnostic.py` | C = (linear_acc - chance) / (teacher_acc - chance) on saved labels | ~20 s |
| `distill_mnli.py` | MNLI with a 3B-Instruct 4-bit teacher + BERT-tiny; capacity failure | ~30 min |
| `distill_mnli_distilbert.py` | DistilBERT 66M on the MNLI labels; 4-bit/2-bit | ~5 min |
| `distill_qnli.py` | QNLI with a 3B 4-bit teacher (yes/no prompt) + BERT-tiny | ~20 min |
| `distill_trec.py` | TREC-QC; teacher failed (54.4% < 59.5% linear), excluded | ~10 min |
| `distill_dolly_qa_t5.py` | Dolly closed_qa: 1.5B teacher -> T5-small; 16/4/2-bit | ~25 min |
| `distill_dbpedia.py` | DBPedia-14 with a 3B 4-bit teacher + BERT-tiny | ~20 min |
| `distill_qqp.py` | QQP duplicate pairs with a 3B 4-bit teacher + BERT-tiny | ~15 min |

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
McNemar p = 0.689, 0.00186 bit/teacher-param. A 4.39M BERT-tiny at 4-bit = 2.19 MB,
76.75% (0.0117 bit/param); 3-bit = 1.64 MB, 77.25%; 2-bit = 1.10 MB, 65.5%.

**DBPedia-14 (14-way, 3B teacher 92.6%):** the same 4.39M BERT-tiny at 4-bit = 91.33%
(-1.3, 0.0117 bit/param); linear C 0.99. Class count alone is not the wall.

**QQP (2-way, 3B teacher 66.5%):** the same BERT-tiny at 4-bit = 58.75% (raw gap -7.75;
S 0.53); linear C 0.21. The transformer beats the linear model — C is a lower bound, not a
predictor.

**SST-2 (2-way, 1.5B teacher 94.25%):** DistilBERT 66M at 2-bit = 73.5%,
0.0893 bit/teacher-param (-20.8 points); at 4-bit = 83.75% (-10.5 points). The same
4.39M BERT-tiny at 4-bit = 2.19 MB, 71.75% (-22.5, 0.0117 bit/param) — the same
architecture and byte budget that matched the AG News teacher.

**Date -> ISO (1.5B teacher 100% exact):** 1.35M char-level seq2seq = 99.8% exact,
0.0036 bit/teacher-param at 4-bit (0.0144 at bf16).

**Dolly closed_qa (open-ended, 1.5B teacher F1 0.4126):** T5-small 60.5M at bf16
F1 0.344 (0.645 bit/param); 4-bit 0.273 (0.161); 2-bit 0.000 (0.081). Open-ended
generation is the hardest regime; the 0.1-bit budget is not enough.

**QNLI (2-way, 3B teacher 81.25%):** the same 4.39M BERT-tiny at 4-bit = 59.75%
(-21.5, 0.0117 bit/param); linear C 0.21.

**MNLI (3-way, 3B teacher 84.7%):** the same 4.39M BERT-tiny at 4-bit = 36.8%
(chance 33.3%, -47.9 points). Capacity failure, not a bit-rate result.

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
| QQP | 50% | 0.21 | 0.53 |
| MNLI | 33.3% | 0.18 | 0.07 |

Fit: **S = 0.91 C + 0.11, R² = 0.859** (n=6). With five tasks S ≈ C looked strong
(R² = 0.965); QQP breaks it (C 0.21, S 0.53). The transformer captures paraphrase that
TF-IDF cannot. C is a lower bound, not a predictor; `NEXT.md` has the 10-task test.

## MNLI: a capacity failure, not a bit-rate result

The 1.5B teacher collapsed on MNLI (3.3% contradiction, 59.4%). With a 3B-Instruct 4-bit
teacher (84.7% on 400 balanced rows), the same 4.39M BERT-tiny reaches 36.8% at 4-bit
(chance 33.3%). A 66M DistilBERT reaches 57.1% at 4-bit (0.179 bit/param, -27.6) but
collapses to 35.8% at 2-bit (0.089 bit/param, -48.9). At the 0.1-bit budget both are at
chance: the task needs more parameters **and** more labels. Controlled comparison:
+1.5 / -21.5 / -22.5 / -47.9.

### TREC: the teacher was below the linear baseline

The 3B teacher with a code prompt labelled 21.7% of rows unknown and reached 54.4% on a
309-row test, below the TF-IDF baseline (59.5%). C > 1 because the "teacher" is not a
ceiling. Excluded from the diagnostic; kept as a negative result.

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
