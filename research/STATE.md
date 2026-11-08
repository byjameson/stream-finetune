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
| MNLI (3-way) | 2.19 MB BERT-tiny 4-bit | 0.0117 | 36.8% vs 84.7% (-47.9, chance) |
| MNLI (3-way) | 16.7 MB DistilBERT 2-bit | 0.089 | 35.8% vs 84.7% (-48.9, chance) |
| date -> ISO | 0.67 MB seq2seq 4-bit | **0.0036** | 99.8% exact (teacher 100%) |
| Dolly closed_qa (open-ended) | 30.3 MB T5-small 4-bit | 0.161 | F1 0.273 vs teacher 0.413 |
| Dolly closed_qa (open-ended) | 15.1 MB T5-small 2-bit | 0.081 | F1 0.000 (dead) |

## The controlled comparison

Same student (BERT-tiny 4.39M), same 0.0117 bit/param:

| task | student | teacher | gap |
|---|---:|---:|---:|
| AG News | 76.75% | 75.25% | **+1.5** |
| DBPedia | 91.33% | 92.60% | **-1.3** |
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
- The controlled same-student comparison across tasks (+1.5 / -21.5 / -22.5 / -47.9).
- The honest negative results (0.1-bit VQ, single-metric diagnostic, teacher collapses).

**Not yet:** a predictive law. A single metric `C` (linear compressibility) fits the
high-C cluster (AG News 0.98, DBPedia 0.99) but **fails at the low end** (QNLI and MNLI
have almost the same C, gaps -21.5 vs -47.9). Five points give R² = 0.80 — a description,
not a law. The next attempt must be multi-feature.

## Failed attempts, kept as negative results

- **MNLI, 1.5B teacher:** predicted contradiction on 3.3% of rows (59.4%); the student
  learned the skew. A 3B teacher was needed (84.7%).
- **QNLI, entailment/not_entailment prompt:** 70.6% unparseable; the yes/no prompt gave 90%.
- **TREC, 3B teacher:** 21.7% unknown, 54.4% accuracy, below the 59.5% linear baseline;
  C > 1, excluded from the diagnostic.
- **Single-metric diagnostic:** R² 0.74 with 4 points, but the QNLI/MNLI pair falsifies it.

## Next

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
