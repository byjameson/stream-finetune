# Next: from a measured spectrum to a predictive law

We now have a measured spectrum, not a law:

| task | student | bits / teacher-param | result |
|---|---|---:|---|
| AG News (4-way) | 340 KB TF-IDF | 0.00186 | matches teacher (p=0.69) |
| AG News (4-way) | 2.19 MB BERT-tiny 4-bit | 0.0117 | matches teacher (+1.5) |
| DBPedia-14 | 2.19 MB BERT-tiny 4-bit | 0.0117 | -1.3 points |
| SST-2 (2-way) | 2.19 MB BERT-tiny 4-bit | 0.0117 | -22.5 points |
| QNLI (2-way) | 2.19 MB BERT-tiny 4-bit | 0.0117 | -21.5 points |
| QQP (2-way) | 2.19 MB BERT-tiny 4-bit | 0.0117 | -8.4 raw, S 0.49 |
| 20 Newsgroups (20-way) | 2.19 MB BERT-tiny 4-bit | 0.0117 | -14.3 raw, S 0.42, weak teacher |
| Emotion (6-way) | 2.19 MB BERT-tiny 4-bit | 0.0117 | -9.7 raw, S 0.59 |
| date -> ISO | 0.67 MB seq2seq 4-bit | 0.0036 | 99.8% exact |
| date -> ISO (program) | 93 B | 4.96e-7 code only | 100% exact |
| MNLI (3-way) | 2.19 MB BERT-tiny 4-bit | 0.0117 | -47.9 points (chance) |
| MNLI (3-way) | 33.5 MB DistilBERT 4-bit | 0.179 | -27.6 points |
| MNLI (3-way) | 16.7 MB DistilBERT 2-bit | 0.089 | -48.9 points (chance) |
| Dolly closed_qa (generation) | 30.3 MB T5-small 4-bit | 0.161 | F1 0.256-0.273 vs 0.413 |
| Dolly closed_qa (generation) | 22.7 MB T5-small 3-bit | 0.121 | F1 0.0001 (dead) |
| Dolly closed_qa (generation) | 15.1 MB T5-small 2-bit | 0.081 | F1 0.000 (dead) |

The next question is whether the floor can be predicted **before** training a student.

## Candidate predictors

1. **Linear compressibility.** `C = (linear_acc - chance) / (teacher_acc - chance)`.
   AG News 0.98, DBPedia 0.99, SST-2 0.49, QNLI 0.21, MNLI 0.18. The student's
   chance-normalized skill should rise with C.
2. **Teacher conditional entropy.** `H(Y|X)` on the test set, from the teacher's label
   distribution. Higher entropy should mean a higher floor.
3. **Input/output length and class count.** Cheap, but probably weak.
4. **The gap at a fixed tiny student.** This is what we are trying to predict, so it is
   a diagnostic, not a pre-hoc predictor.

The eight-point check: in **chance-normalized skill**
`S = (student - chance) / (teacher - chance)`, S tracks C loosely
(**S = 0.90 C + 0.07, R² = 0.827**). Emotion lands on the line. **QQP breaks it upward**
(C 0.21, S 0.49, 3-seed mean); **20 Newsgroups breaks it downward** (C 0.60, S 0.42, weak
teacher, 100 labels per class). The linear student is a lower bound; a transformer can
capture compositional structure it misses, or underfit when data is scarce. The next test
is 10 tasks with a multi-feature model (class count, chance, C, teacher accuracy, student
type), not a bigger fit on C.

## Experiment

0. **Representation and sharing first.** Before distilling, ask whether the behaviour is
   algorithmic, and what is shared. A 93-byte date parser is 100% exact; a 1.35M seq2seq is
   99.8%. A 341-byte prompt on a shared base matches a 340 KB student on AG News; on
   SST-2, 294 bytes beat it by 22 points. The
   ladder is: **program -> prompt -> adapter -> coreset -> linear -> tiny transformer ->
   4-bit model**. The adapter rung is a real sub-0.1-bit artifact: 7.37M LoRA params =
   0.077 bit/base-param fp32, 0.038 fp16, 0.019 at 8-bit, 0.0095 at 4-bit (110 rows: loss
   2.3097 vs 2.3084 original; base 2.3197). Do not SVD it (rank 8 loses 35.8%); quantize it. The
   coreset rung stores data, not weights: 135 KB
   trains a 70.5% student, 270 KB reaches 78.0%. State the accounting boundary (marginal vs
   base-counted); 0.1 bit is meaningless without it.
- 10 tasks with a clear label: AG News, SST-2, TREC, MNLI, DBPedia, spam, topic,
  sentiment, NLI, intent.
- For each: 2,000 teacher labels, 400 held-out rows, a fixed teacher (3B-Instruct, 4-bit)
  and a fixed student ladder (TF-IDF 1k/2k/5k, BERT-tiny 4-bit).
- Metric: the smallest student bytes whose McNemar p against the teacher is > 0.05.
- Fit `log(bits_min)` against `C`, `H(Y|X)`, class count and length.
- Report R² and the residuals; do not hide tasks that break the fit.

If the fit is good, the result is a practical predictor: **before training, estimate
whether 0.1 bit/param is plausible for this task**. If it is bad, the honest conclusion is
that every task needs its own student search.

## Pitfalls

- **Teacher quality.** The 1.5B teacher collapsed on MNLI (3.3% contradiction); a biased
  teacher makes the task look incompressible. Use a stronger teacher and check the label
  distribution.
- **Test-set size.** 400 rows gives wide confidence intervals; the AG News p=0.69 is not
  proof of equality, only absence of evidence for a difference.
- **Contamination.** The teacher may have seen the benchmark; report it.
- **Total bytes.** Count the tokenizer, the adapter and any retrieval index. A 0.1-bit
  claim that ignores the decoder is not a compression claim.
- **Narrow vs general.** A result on a 4-way classifier is not a claim about general chat.

## Product path

If the predictor works, `research/autodistill.py` can use it to skip the ladder: given a
task and a byte budget, it picks the student class (linear, tiny transformer, seq2seq) and
reports the expected floor before training. That is a useful tool, not a new distillation
algorithm.

The rung chooser (`research/ladder.py`) already does the measured version: given a task, a
target accuracy and whether the base is shared, it returns the cheapest measured artifact.
