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

### Result on 2026-11-21

| model | score |
|---|---|
| 49.2M base | **21/21 (100%)** |
| 49.2M + `adapters_50m/v0001` | **21/21 (100%)** |

The base and the product adapter tie on these deterministic behaviors. That is
expected: the adapter was trained for a narrow skill queue, not to replace the
runtime tools. The adapter's value is shown by the loss gates and holdout, not
by this table.

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
