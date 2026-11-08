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

**SST-2 (2-way, 1.5B teacher 94.25%):** DistilBERT 66M at 2-bit = 73.5%,
0.0893 bit/teacher-param (-20.8 points); at 4-bit = 83.75% (-10.5 points).

**Date -> ISO (1.5B teacher 100% exact):** 1.35M char-level seq2seq = 99.8% exact,
0.0036 bit/teacher-param at 4-bit (0.0144 at bf16).

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
