# Benchmarks and honest limits

Atom Zero is a **narrow local assistant**, not a general model. The benchmark
here measures product behavior: tools, memory, tasks, source-cited retrieval,
teacher-note review, missing-source detection and one short generation. It is
**not MMLU**, a coding benchmark or a general-reasoning score.

## Product benchmark

```bash
python research/benchmark_assistant.py \
  --model checkpoints/transformer_behavior_50m_long \
  --out research/results/assistant_benchmark_base.json

python research/benchmark_assistant.py \
  --model checkpoints/transformer_behavior_50m_long \
  --adapter checkpoints/adapters_50m/v0001 \
  --out research/results/assistant_benchmark_adapter.json
```

The benchmark creates a temporary document index and a temporary gap store, so
it does not touch personal notes. It runs 21 checks:

- greeting, help/limits, arithmetic, time
- name and likes memory
- add/list/complete tasks and empty-task list
- source-cited retrieval from five documents
- unverified and verified teacher notes
- missing-source gap detection
- one short generation without role-tag leakage

With `--safety-filter` it adds five safety checks and runs 26 in total:

- violence, self-harm and sexual-minors requests are refused
- benign child-abuse reporting and Wi-Fi security questions are not blocked

### Result on 2026-11-21

| model | score |
|---|---|
| 49.2M base | **21/21 (100%)** |
| 49.2M + `adapters_50m/v0001` | **21/21 (100%)** |
| 49.2M + safety filter | **26/26 (100%)** |

The base and the product adapter tie on these deterministic behaviors. That is
expected: the adapter was trained for a narrow skill queue, not to replace the
runtime tools. The adapter's value is shown by the loss gates and holdout, not
by this table. The neural reranker keeps the same gates: product **21/21** and
safety **26/26** (`assistant_benchmark_scaled_40k_neural_rerank*.json`).

## Retrieval reranker benchmark

`research/benchmark_retrieval.py` builds a 20-document authored office corpus
and asks 20 paraphrased questions in a temporary database. No reranker is
trained on these pairs. It reports top-1, top-3, MRR, citation accuracy
(top-1 equals the expected source) and latency.

```bash
python research/benchmark_retrieval.py \
  --neural-model models/all-MiniLM-L6-v2 \
  --out research/results/retrieval_benchmark.json
```

| reranker | top-1 | top-3 | MRR | citation | median latency |
|---|---:|---:|---:|---:|---:|
| raw FTS | 0.850 | 0.900 | 0.875 | 0.850 | 0.47 ms |
| hashed n-gram | 0.850 | 0.900 | 0.875 | 0.850 | 0.95 ms |
| MiniLM-L6-v2 neural | **0.900** | 0.900 | **0.900** | **0.900** | 16.5 ms |

FTS candidate recall@10 is **0.900**, so neither reranker can recover the two
queries FTS never retrieves ("printing pale", "wear to a customer meeting").
The neural candidate wins 5 points of top-1 and citation accuracy on this small
authored set; that is directional evidence, not a general IR claim.

The neural path is optional and local:

- `--reranker auto` (server default) uses the local model when
  `models/all-MiniLM-L6-v2` exists and falls back to hashed n-grams.
- `--reranker hashed` is the dependency-free path.
- `--reranker neural` fails loudly if the model is missing.

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('sentence-transformers/all-MiniLM-L6-v2', local_dir='models/all-MiniLM-L6-v2')"
```

`models/` is gitignored. The reranker is retrieval only; it does not change the
49.2M core model. CPU latency is about 16 ms per query here, with a few seconds
of warm-up. See [`PRODUCT_AUDIT.md`](PRODUCT_AUDIT.md) for the remaining gap:
the candidate stage, not the reranker, is now the bottleneck.

## Raw model knowledge probe

The product score is high because tools and RAG do the work. Without them, the
50M model is weak. This probe asks ten simple general questions directly, with
no DocMemory, no calculator and no agent:

```bash
python research/benchmark_raw.py \
  --model checkpoints/transformer_behavior_50m_long \
  --out research/results/raw_knowledge_base.json
```

| model | raw score |
|---|---|
| 49.2M base | **2/10 (20%)** |
| 49.2M + product adapter | **2/10 (20%)** |

Examples:

```text
PASS What is the capital of France? -> The capital of France is Paris.
FAIL What is the capital of Turkey? -> The capital of Norway is the capital of Sea...
FAIL What is 2 + 2? -> To find 2 + 2, we need to find the total number of possible values...
FAIL Who wrote Hamlet? -> Mia
FAIL What is the speed of light? -> The speed of light is about 1.5 degrees.
```

This is the honest reason for the architecture: **facts belong in RAG, not in
the 50M weights**. It is also why the product does not claim general knowledge.
Results: [`raw_knowledge_base.json`](../research/results/raw_knowledge_base.json)
and [`raw_knowledge_adapter.json`](../research/results/raw_knowledge_adapter.json).

## Data-scaling early result

The scaled 50M checkpoint at 10k steps was evaluated on the **new** scaled
validation shard and on the product behaviors:

| check | base | scaled 10k |
|---|---|---|
| scaled-valid chat bpb | 1.503 | **1.382** |
| scaled-valid assistant loss | 4.215 | **3.855** |
| scaled-valid stories bpb | 1.622 | **1.414** |
| product benchmark | 21/21 | **21/21** |
| safety benchmark | 26/26 | **26/26** |
| behavior checks | 16/16 | **16/16** |
| raw-model probe | 2/10 | 2/10 |

The 40k checkpoint is promoted as `checkpoints/transformer_behavior_50m_scaled`;
training stopped at 40k because 38k/40k plateaued. **Do not use the old clean
valid shard for this comparison:** the larger corpus was regenerated from the
same source datasets, so that shard is not a clean holdout for the new run.
Details: [`DATA_SCALING.md`](DATA_SCALING.md).

## Safety red-team

```bash
python research/redteam_safety.py \
  --model checkpoints/transformer_behavior_50m_long \
  --adapter checkpoints/adapters_50m/v0001 \
  --out research/results/redteam_safety_agent.json
```

| check | result |
|---|---|
| harmful/benign filter cases | **72/72** |
| output-filter cases | **4/4** |
| agent refusal path | **72/72** |
| product benchmark with filter | **26/26** |

The filter is pattern-based and opt-in. It is a guardrail, not proof of general
safety; see [`SAFETY.md`](SAFETY.md).

## Adapter loss/holdout evidence

```text
8-example gate:        target +16.74%
                       clean bpb +1.58%, assistant +1.54%
                       behavior and smoke PASS
6-train / 2-holdout:   base 3.853 -> adapter 3.139 (-18.5%)
```

Sources:

- [`gated_feedback_history_test.json`](../research/results/gated_feedback_history_test.json)
- [`holdout_feedback_test.json`](../research/results/holdout_feedback_test.json)
- [`adapter_comparison_feedback.json`](../research/results/adapter_comparison_feedback.json)

These are small-sample measurements. They show that the reviewed update loop
works; they do not make the model general.

## What is not measured

- MMLU/GSM8K/HumanEval or coding ability.
- Open-ended factual reasoning without sources.
- Safety red-team performance; the release checklist requires it before hosting.
- Long-context or multi-document synthesis at scale.
- Turkish or multilingual quality.

Do not claim a general benchmark score from the product benchmark. The 50M
model remains a narrow daily assistant; facts stay in RAG and skills stay in
gated adapters.
