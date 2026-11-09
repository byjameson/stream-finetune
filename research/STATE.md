# State of the bit-floor research

## The question

Can a model be compressed to **0.1 bit per weight** without losing quality?

## The short answer

Not by quantizing the model. Yes, by changing what is stored: a narrow **behaviour** can be
distilled far below 0.1 bit per teacher-parameter; a general model cannot. The floor is a
property of the behaviour and of the student's inductive bias, not a universal bit number.

## Measured regimes

| regime | student | bits / teacher-param | result |
|---|---|---:|---|
| general LM, 0.5B | 4-bit | ~4 | CE 2.99 vs 2.78 (+24% PPL) |
| general LM, 0.5B | 3-bit | ~3 | PPL 52.6 vs 16.2 |
| general LM, 0.5B | 0.1-bit VQ | 0.10 | CE 12.48, worse than uniform (11.93) |
| AG News (4-way) | 340 KB TF-IDF | **0.0019** | 74.25% vs 75.25% teacher (p=0.69) |
| AG News (4-way) | 2.19 MB BERT-tiny 4-bit | 0.0117 | 76.75% vs 75.25% (+1.5) |
| DBPedia-14 | 2.19 MB BERT-tiny 4-bit | 0.0117 | 91.33% vs 92.60% (-1.3) |
| SST-2 (2-way) | 2.19 MB BERT-tiny 4-bit | 0.0117 | 71.75% vs 94.25% (-22.5) |
| SST-2 (2-way) | 16.7 MB DistilBERT 2-bit | 0.089 | 73.5% vs 94.25% (-20.8) |
| QNLI (2-way) | 2.19 MB BERT-tiny 4-bit | 0.0117 | 59.75% vs 81.25% (-21.5) |
| QQP (2-way) | 2.19 MB BERT-tiny 4-bit | 0.0117 | 58.08% vs 66.50% (-8.4 raw; S 0.49) |
| 20 Newsgroups (20-way) | 2.19 MB BERT-tiny 4-bit | 0.0117 | 15.25% vs 29.50% (-14.25 raw; S 0.42) |
| Emotion (6-way) | 2.19 MB BERT-tiny 4-bit | 0.0117 | 30.67% vs 40.33% (-9.66 raw; S 0.59) |
| MNLI (3-way) | 2.19 MB BERT-tiny 4-bit | 0.0117 | 36.8% vs 84.7% (-47.9, chance) |
| MNLI (3-way) | 16.7 MB DistilBERT 2-bit | 0.089 | 35.8% vs 84.7% (-48.9, chance) |
| date -> ISO | 0.67 MB seq2seq 4-bit | **0.0036** | 99.8% exact (teacher 100%) |
| date -> ISO (program) | 93 B (+752 KB shared dateutil) | **4.96e-7** code only | 100% exact |
| AG News (prompt) | 341 B + shared 1.5B base | 1.82e-6 marginal / 16 with base | 73.75% |
| SST-2 (prompt) | 294 B + shared 1.5B base | 1.57e-6 marginal / 16 with base | 93.75% |
| Qwen2.5-3B LoRA | 29.6 MB fp32 / 14.8 MB fp16 | 0.077 / 0.038 | 60-step SFT |
| adapter 8-bit | 8.56 MB | 0.0222 | 0.53% error; Dolly p=0.81, SST-2 93.75%, AG News 62.0% |
| adapter 4-bit | 4.88 MB | 0.0126 | 9.08% error; Dolly +0.0018, SST-2 93.75%, AG News 62.0% |
| SST-2 4-shot prompt | 441 B | 1.14e-6 | 94.0% on the same Qwen3B base |
| AG News 4-shot prompt | 242 B | 6.27e-7 | 61.75% on the same Qwen3B base |
| AG News (coreset 500) | 135 KB | 0.00072 | 70.5% |
| AG News (coreset 1000) | 270 KB | 0.00144 | 78.0% |
| Dolly closed_qa (open-ended) | 121 MB T5-small bf16 | 0.645 | F1 0.380 vs teacher 0.413 |
| Dolly closed_qa (open-ended) | 30.3 MB T5-small 4-bit | 0.161 | F1 0.256 (second run 0.273) |
| Dolly closed_qa (open-ended) | 22.7 MB T5-small 3-bit | 0.121 | F1 0.0001 (dead) |
| Dolly closed_qa (open-ended) | 15.1 MB T5-small 2-bit | 0.081 | F1 0.000 (dead) |

## The controlled comparison

Same student (BERT-tiny 4.39M), same 0.0117 bit/param:

| task | student | teacher | gap |
|---|---:|---:|---:|
| AG News | 76.75% | 75.25% | **+1.5** |
| DBPedia | 91.33% | 92.60% | **-1.3** |
| QQP | 58.08% | 66.50% | **-8.4** (S 0.49) |
| 20 Newsgroups | 15.25% | 29.50% | **-14.3** (S 0.42) |
| Emotion | 30.67% | 40.33% | **-9.7** (S 0.59) |
| QNLI | 59.75% | 81.25% | **-21.5** |
| SST-2 | 71.75% | 94.25% | **-22.5** |
| MNLI | 36.80% | 84.71% | **-47.9** |

Same architecture, same bytes, a 49-point range. This is the central result.

## Two walls

1. **The bit wall.** Post-hoc quantization dies below 3–4 bits; BitNet shows 1.58 bits is
   possible with training-aware quantization. 0.1 bit is not reachable this way.
2. **The capacity wall.** A 4.4M student cannot represent 3-way NLI; a 66M DistilBERT at
   2-bit collapses on MNLI; open-ended generation dies at 2-bit. Adding bits back does not
   fix a model that is too small for the behaviour.

## What is new — and what is not

**Not new:** distillation (Hinton 2015, DistilBERT, TinyBERT, Alpaca), the student
architectures, the idea of task-specific compression.

**New here:**
- The metric: **bits per teacher parameter**, with total student bytes counted.
- Quantization vs distillation on the same test set and byte budget.
- The controlled same-student comparison across tasks (AG News to MNLI).
- **Representation selection:** the best date student is a **93-byte program**, not a
  1.35M seq2seq. For deterministic tasks, code is the right representation; the bit floor
  is set by the representation class, not the teacher.
- **The sharing ladder / accounting boundary:** a **341-byte prompt** on a shared 1.5B
  base reaches 73.75% on AG News (the 340 KB student gets 74.25%). On SST-2, a **294-byte
  prompt** reaches **93.75%** and beats the 340 KB student by 22 points. Marginal 1.6-1.8e-6
  bit/param; count the base and it is 16 bit/param. The 0.1-bit question is an accounting
  question.
- **Data as artifact:** a 135 KB coreset trains a 70.5% student at load time; a 270 KB
  coreset reaches 78.0%, indistinguishable from the 340 KB model. Storage and compute are
  interchangeable.
- **The practical sub-0.1-bit artifact:** a 7.37M-param LoRA is **0.077 bit/base-param**
  fp32, **0.038** fp16, with the 3.09B base shared. It adds capability; the prompt does not.
  **But prompt first:** a 441-byte 4-shot prompt on the same base gets 94.0% on SST-2
  (adapter 93.75%) and 242 bytes gets 61.75% on AG News (adapter 62.25%). The adapter is
  for behaviour a prompt cannot express, not an automatic win.
- **Adapter compression is quantization, not rank:** rank 8 SVD loses 35.8%; 8-bit =
  **0.0222 bit/param** (Dolly 870 rows: no measurable loss, p = 0.81; SST-2: 93.75% both;
  AG News: 62.0% vs 62.25%), 4-bit = 0.0126 (Dolly +0.0018, p = 0.001, 5% of the 0.0349
  gain; SST-2 p = 0.41; AG News loss slightly lower, p = 0.0002).
- **Inference cost:** the LoRA path is 8.75 tok/s vs 11.79 for the base (~26% slower on the
  reference card); a bf16 LoRA path reaches 9.13 (~22% slower). The 4.88 MB adapter loads
  in 0.115 s. It is a storage/switching win, not a per-token speed win; merge into a bf16
  base for zero overhead.
- **A practitioner decision rule:** deterministic -> program; shared capable base -> prompt;
  shared base, new capability -> adapter; no base -> coreset/linear/transformer; general ->
  4-bit. `research/ladder.py` picks the cheapest rung for a target accuracy and a
  shared/not-shared base.
- The honest negative results (0.1-bit VQ, single-metric diagnostic, teacher collapses).

**C is a lower bound, not a predictor.** With five tasks, chance-normalized skill looked
like it tracked `C` almost 1:1 (S = 1.04 C - 0.01, R² = 0.965). **QQP breaks it upward:**
C 0.21, S 0.49 (3-seed mean) — the transformer captures paraphrase that TF-IDF misses.
**20 Newsgroups breaks it downward:** C 0.60, S 0.42 — a weak teacher (29.5%) and only
100 labels per class for 20 classes. **Emotion** (C 0.55, S 0.59) lands on the line.
Eight tasks give S = 0.90 C + 0.07, R² = 0.827. The cheap linear student is a floor on
what a better student can do, not a predictor of it. The next test is 10 tasks with a
multi-feature model.

## Failed attempts, kept as negative results

- **MNLI, 1.5B teacher:** predicted contradiction on 3.3% of rows (59.4%); the student
  learned the skew. A 3B teacher was needed (84.7%).
- **QNLI, entailment/not_entailment prompt:** 70.6% unparseable; the yes/no prompt gave 90%.
- **TREC, 3B teacher:** 21.7% unknown, 54.4% accuracy, below the 59.5% linear baseline;
  C > 1, excluded from the diagnostic.
- **Single-metric diagnostic:** R² 0.74 with 4 points, but the QNLI/MNLI pair falsifies it.

## Next

**Done:** adapter validation on three tasks (Dolly, SST-2, AG News) and same-base prompt
baselines. **Still missing:** independent hardware rows / a second machine. The research
publish waits for that, or ships with an explicit one-machine caveat.

1. **10 tasks, multi-feature model:** class count, chance level, linear gap, teacher
   entropy, input length. Report leave-one-out, not just R².
2. **Generation at 0.1 bit:** T5-base / FLAN-T5 with more teacher data; find whether the
   2-bit collapse is a quantization artefact or a capacity wall.
3. **Publish** the note/article and invite independent rows (same protocol, same JSON).
4. **Do not claim a universal method.** The current evidence says the floor is task- and
   student-dependent.

## Files

- `docs/BEHAVIORAL_BITS.md` — full note (EN)
- `research/ARTICLE.md` — dev.to draft (EN)
- `research/OZET_TR.md` — Turkish summary
- `research/NEXT.md` — the predictive-law plan
- `research/autodistill.py` — budget -> smallest passing linear student
- `research/results/` — measured JSON rows
- `docs/figures/` — bit cliff, narrow-task frontier, spectrum, diagnostic
