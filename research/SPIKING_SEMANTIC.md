# Held-out narrative-coherence benchmark

The existing bpb and ROUGE probes do not answer whether the model can use story
context. This benchmark asks a four-way narrative-coherence question:

- context: the first two sentences of a held-out TinyStories story
- candidates: the true next sentence + three next sentences from other stories
- the model scores each candidate by length-normalized log probability
- chance = 25%

100 cases are built from the TinyStories validation token stream
(`research/make_semantic_cases.py` -> `research/semantic_cases.json`). The
contexts/answers are not in the training shard used for phase A.

## Results, phase-A checkpoints

| model | params | checkpoint step | accuracy | chance | mean margin |
|---|---:|---:|---:|---:|---:|
| 4.3M | 4,325,633 | 10,000 | 0.29 | 0.25 | -0.414 |
| 21M | 20,973,570 | 10,000 | **0.41** | 0.25 | -0.280 |
| 50M | 50,073,842 | 500 | 0.40 | 0.25 | -0.261 |

Raw: `research/results/spiking_semantic.json`.

## EWC lambda sweep on 50M phase B

Using the 50M phase-A checkpoint, 600 WikiText steps, 20% replay, and the same
narrative benchmark:

| EWC lambda | old bpb | new bpb | semantic accuracy |
|---:|---:|---:|---:|
| phase A | 1.3066 | 4.2201 | 0.40 |
| 0.5 | 1.3158 | 3.2273 | 0.39 |
| 2.0 | 1.3130 | 3.2446 | 0.40 |

The 50M replay-only arm (from the long replay experiment) had old 1.4040, new
**3.1616**, and improved the continuation content F1 to 0.182 (from 0.140). EWC
keeps old bpb nearly unchanged, but it does not beat replay on new bpb and does
not improve the semantic multiple-choice score.

## What it shows

1. **4.3M is barely above chance** (0.29). It has learned local story-like text,
   not narrative selection.
2. **21M and 50M are above chance** (0.41/0.40), so they learn some context
   selection, but **50M does not beat 21M**. More parameters alone did not buy
   semantic capability at this budget.
3. **All mean margins are negative**: even when the correct candidate wins, the
   model often prefers a distractor; accuracy is not high confidence.
4. This supports the architecture/data conclusion from replay experiments:
   retention can be fixed with replay, but capability does not follow from
   parameter count alone.

5. **EWC tuning does not buy semantic capability.** Lambda 0.5 and 2.0 stay at
   0.39/0.40, essentially phase A, while new bpb is worse than replay-only.
   EWC is an old-task stabilizer, not a capability mechanism.

6. **Decision:** do not scale the current architecture further on TinyStories.
   21M and 50M are both around 0.40 narrative accuracy; the next attempt should
   change the data/task/architecture (real instruction or QA data, retrieval,
   or a different spiking topology), not just parameter count.

## Reproduce

```bash
python research/make_semantic_cases.py
python research/spiking_semantic_eval.py \
  --models checkpoints/spiking_4m_10k checkpoints/spiking_21m \
           checkpoints/spiking_50m_run \
  --out research/results/spiking_semantic.json
```

## Caveats

- TinyStories-only; this is narrative coherence, not general knowledge.
- The candidate score is a language-model log probability, not a trained NLI
  judge.
- 100 cases, one tokenizer, one seed family.
- Distractors are other TinyStories sentences with similar token length; this
  controls length but not topic.
- 50M checkpoint is step 500, while 21M is step 10,000; this is a
  checkpoint-availability comparison, not an equal-step one.

## Next

- Run the benchmark after replay/EWC phase B to see whether retention preserves
  semantic accuracy.
- Add a real held-out QA set with exact answers, not only continuation choice.
- Revise architecture/data if 50M still does not beat 21M at equal steps.
