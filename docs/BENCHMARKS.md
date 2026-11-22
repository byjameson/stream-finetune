# Benchmarks and honest limits

Atom Zero is a **narrow local assistant**, not a general model. The benchmark
here measures product behavior: tools, memory, tasks, source-cited retrieval,
teacher-note review, missing-source detection and one short generation. It is
**not MMLU**, a coding benchmark or a general-reasoning score. The shareable
summary and model card are in [`MODEL_CARD.md`](MODEL_CARD.md).

## Public snapshot

| variant | raw knowledge | product | behavior | source-cited retrieval |
|---|---:|---:|---:|---:|
| base 49.2M | 2/10 | 21/21 | 16/16 | — |
| product 49.2M (40k) | 2/10 | 21/21 | 16/16 | — |
| + adapter `v0001` | 2/10 | 21/21 | 16/16 | — |
| + RAG/tools | — | 21/21 | 16/16 | top-1 1.000 |
| + optional ML safety | — | 26/26 | 16/16 | filter off by default |

Raw means no RAG, no tools and no agent. "Product" adds the runtime. The
adapter does not add factual knowledge; it is a style/skill layer with a
reviewed gate. See the [model card](MODEL_CARD.md) for the full distinction.

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
by this table. The hybrid neural path keeps the same gates: product **21/21**
and safety **26/26** (`assistant_benchmark_scaled_40k_hybrid_neural*.json`);
the hashed fallback is also 21/21 and 26/26.

## Retrieval reranker benchmark

`research/benchmark_retrieval.py` builds a 20-document authored office corpus
and asks 20 paraphrased questions in a temporary database. No reranker is
trained on these pairs. It reports top-1, top-3, MRR, citation accuracy
(top-1 equals the expected source) and full retrieval latency.

```bash
python research/benchmark_retrieval.py \
  --neural-model models/all-MiniLM-L6-v2 \
  --out research/results/retrieval_benchmark_hybrid.json
```

| candidate stage + reranker | top-1 | top-3 | MRR | citation | median latency |
|---|---:|---:|---:|---:|---:|
| raw FTS | 0.850 | 0.900 | 0.875 | 0.850 | 0.34 ms |
| FTS + hashed pool | 0.850 | 0.900 | 0.875 | 0.850 | 1.36 ms |
| FTS + neural pool | **1.000** | **1.000** | **1.000** | **1.000** | 8.8 ms |

Candidate-pool recall@10:

| stage | recall@10 |
|---|---:|
| FTS only | 0.900 |
| FTS + MiniLM neural pool | **1.000** |

The two misses in the previous FTS-only run were "Why are my documents printing
pale?" (printer) and "What should I wear to a customer meeting?" (dress code).
Both are absent from FTS, so no reranker could recover them. The neural
candidate pool supplies both, and the reranker puts them at rank 1. The hashed
pool stays FTS-anchored: it did not answer "population of Paris" from hash
collisions, and it refuses semantic-only queries rather than risking a false
answer. That is deliberate: the dependency-free fallback is conservative, and
the neural path is the promoted hybrid retriever.

The neural path is optional and local:

- `--reranker auto` (server default) uses the local model when
  `models/all-MiniLM-L6-v2` exists and falls back to hashed n-grams.
- `--reranker hashed` is the dependency-free path.
- `--reranker neural` fails loudly if the model is missing.

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('sentence-transformers/all-MiniLM-L6-v2', local_dir='models/all-MiniLM-L6-v2')"
```

`models/` is gitignored. The reranker is retrieval only; it does not change the
49.2M core model. CPU latency is about 8.8 ms per query here after the vector
index is warm, with a few seconds of model warm-up. Chunk vectors are cached in
SQLite (`doc_vectors`) by model key and pruned when a source is replaced or
forgotten. See [`PRODUCT_AUDIT.md`](PRODUCT_AUDIT.md) for the remaining gaps:
the hashed fallback is still lexical, the ANN index is IVF rather than HNSW,
and multi-document conflict resolution is not implemented.

## Document and citation benchmark (100 questions)

`research/benchmark_documents.py` generates a deterministic 10-document corpus
in a temporary directory: Markdown, PDF, DOCX and HTML. It asks **80 answerable
questions** with an expected source and page/section, plus **20 questions with
no local source**. The corpus is authored; this is a regression benchmark, not
a public leaderboard.

```bash
python research/benchmark_documents.py \
  --out research/results/document_benchmark.json --check
```

| metric | result |
|---|---:|
| top-1 source | **1.000** (80/80) |
| top-1 page/section locator | **1.000** (80/80) |
| top-1 contains expected answer | **1.000** (80/80) |
| complete citation (source + locator + answer) | **1.000** (80/80) |
| no-answer refusal | **1.000** (20/20) |
| median retrieval latency | **~1-2 ms** |
| p95 retrieval latency | **~2-3 ms** |

By format: Markdown 24/24, PDF 24/24, DOCX 16/16, HTML 16/16 for source,
locator and answer. The `--check` mode enforces source ≥ 0.95, locator ≥ 0.90,
answer ≥ 0.90, complete citation ≥ 0.90, refusal ≥ 0.95 and median latency
≤ 100 ms. CI runs the same check.

The PDF path uses page numbers; DOCX/HTML/Markdown use heading sections. The
corpus is small and the questions are authored, so this does not measure open
domain robustness. It does catch regressions in ingestion, provenance, source
routing and the no-answer path. Results:
[`document_benchmark.json`](../research/results/document_benchmark.json).

### Prewarmed exact vector index

`research/benchmark_vector_index.py` compares the legacy linear scan (read every
SQLite blob and loop in Python) with a one-time numpy matrix built at startup.
It uses hashed 512-dim vectors, so the benchmark needs no neural model.

```bash
python research/benchmark_vector_index.py \
  --chunks 3000 --repeats 20 \
  --out research/results/vector_index_benchmark.json
```

| chunks | prewarm | matrix query | linear scan | speedup | top-5 overlap |
|---:|---:|---:|---:|---:|---:|
| 3000 | 747 ms | **0.77 ms** | 127.6 ms | **166x** | **1.000** |

Both paths are exact cosine over the same vectors; the top-5 sources match.
Full `search` (FTS + matrix + rerank) is 6.9 ms at this size. The matrix is
about 6 MB for 3000 hashed chunks. The server builds it after loading the model
and reranker; `--no-prewarm` skips it for very large corpora or slow disks.

### IVF ANN index

Above `ANN_MIN_CHUNKS` (256), auto mode builds deterministic k-means IVF
buckets. A query scores centroids, probes the best `nprobe` buckets and exactly
reranks only their members. `--vector-index ann|exact|auto` selects the path.
Anything below the auto threshold, an empty/short bucket result, an ANN error,
or an ANN top score below the hybrid threshold falls back to the exact matrix.

```bash
python research/benchmark_ann_index.py \
  --vectors 20000 --clusters 128 --probes 4,8,12 \
  --out research/results/ann_index_benchmark.json
```

Clustered synthetic normalized vectors, 20k vectors, 128 buckets, build 0.24 s:

| probe | ANN median | exact median | speedup | recall@10 |
|---:|---:|---:|---:|---:|
| 4 | **0.64 ms** | 3.22 ms | **5.0x** | **1.000** |
| 8 | 1.43 ms | 3.22 ms | 2.3x | **1.000** |
| 12 | 1.90 ms | 3.22 ms | 1.7x | **1.000** |

This is approximate IVF, not an HNSW graph; the exact fallback is the safety
net. On the small authored retrieval set `--vector-index ann` keeps top-1 at
**1.000** (two low-confidence queries used the exact fallback). A live 300-doc
server test reported `mode=ann`, `nlist=17`, `nprobe=5`, `fallbacks=0` and
returned `CODE-137`/`CODE-299` with the right sources.

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
| 49.2M product 40k | **2/10 (20%)** |
| 49.2M product 40k + adapter `v0001` | **2/10 (20%)** |

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
Results: [`raw_knowledge_base.json`](../research/results/raw_knowledge_base.json),
[`raw_knowledge_scaled_40k.json`](../research/results/raw_knowledge_scaled_40k.json)
and [`raw_knowledge_scaled_40k_adapter.json`](../research/results/raw_knowledge_scaled_40k_adapter.json).

## Teacher -> RAG learning demo

`research/teacher_learning_demo.py` asks a question with no local source, asks
the local Ollama teacher, stores the answer as an unverified teacher note, and
asks again:

```text
before:  I do not have a source for that yet...
teacher: The capital of New Zealand is Wellington.
after:   I found these notes:
         - ... Wellington. (teacher note, unverified: teacher://ollama/...)
weights_changed: false
```

This is **retrieval-memory learning**: the answer enters `DocMemory` as
unverified data and can be cited immediately. No gradient update happens, and
the model does not autonomously decide what to study. Evidence:
[`teacher_rag_learning.json`](../research/results/teacher_rag_learning.json).

The **curiosity queue** wraps this in a review gate: pending knowledge gaps are
proposed as `kind=curiosity` research items; nothing runs until approval. A live
run recorded `unapproved_run: 0`, `approved_run: done`, and
`weights_changed: false`:
[`curiosity_live.json`](../research/results/curiosity_live.json). See
[`CURIOSITY.md`](CURIOSITY.md).

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
python research/redteam_safety.py --ml \
  --model checkpoints/transformer_behavior_50m_scaled \
  --adapter checkpoints/adapters_50m/v0001 \
  --out research/results/redteam_safety_ml_agent.json

python research/redteam_moderation.py \
  --out research/results/redteam_moderation.json
```

| check | rules only | ML hybrid |
|---|---:|---:|
| harmful/benign filter cases | **72/72** | **72/72** |
| output-filter cases | **4/4** | **4/4** |
| agent refusal path | **72/72** | **72/72** |
| benign false blocks | 0/26 | **0/26** |
| held-out semantic paraphrases | 1/10 | **10/10** |
| product benchmark with filter | **26/26** | **26/26** |

The hybrid filter is rule-first; the local MiniLM + linear head only adds
blocks. On the authored 72-case set ML-only is 61.1% (multilingual/category
cases remain rule-covered), but it blocks 10/10 held-out semantic paraphrases
the rules deliberately do not encode, with 0/26 benign false blocks. It is a
guardrail, not proof of general safety; see [`SAFETY.md`](SAFETY.md).

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
- A large human-labelled moderation benchmark; the current sets are authored
  and small (72 rule cases, 10 semantic paraphrases, 26 benign controls).
- Long-context or multi-document synthesis at scale.
- Turkish or multilingual quality.

Do not claim a general benchmark score from the product benchmark. The 50M
model remains a narrow daily assistant; facts stay in RAG and skills stay in
gated adapters.
