# Atom Zero: Capability and Potential

Status: 2026-11-21, release `0.8.0`. This is an honest capability assessment,
not a general-model claim.

## Short answer

Atom Zero is **not** a superintelligence and the 49.2M core is not a general
model. Its current value is a **local agent harness**: source-grounded RAG,
deterministic tools, session memory, reviewed adapters, hybrid retrieval, an
optional local provider backend, and deletion/privacy paths.

The raw 50M model scores **2/10** on the no-RAG knowledge probe. RAG and tools
do most of the useful work. The harness can become a strong domain tool when it
is given a curated corpus, tools, and optionally a larger local model. It
cannot become a general reasoning/coding assistant by prompting the current
50M weights.

## What is already strong

| capability | evidence |
|---|---|
| source-grounded retrieval | hybrid neural top-1/citation **1.000** vs FTS 0.850 |
| retrieval latency | **8.8 ms** median on the authored set; exact matrix/IVF ANN paths |
| product behavior | **21/21**; safety suite **26/26** when the optional filter is on |
| behavior checks | **16/16** |
| determinism/safety of tools | calculator, time, tasks, memory; no autonomous crawling |
| privacy | local SQLite, no telemetry, exact source/session deletion, export/import |
| reviewed learning | feedback -> human review -> gated LoRA -> rollback |
| local provider | optional Ollama/OpenAI-compatible backend tested with `llama3.2:3b` |

## What is missing

1. **General model capability.** The 50M model cannot reason or code broadly.
   A larger model and much more data/compute are required. The 40k->60k resume
   did not change this.
2. **Tool planning.** The agent routes explicit commands with deterministic
   rules; it does not let a model choose tools freely. Provider mode currently
   answers free-form text, not multi-step tool use.
3. **Multilingual quality.** The core is English-first; multilingual safety and
   retrieval rely on rules/models around it.
4. **Evaluation breadth.** The product benchmark is narrow. There is no
   MMLU/GSM8K/HumanEval or long-context benchmark.
5. **Scale.** 6 GB VRAM and 34 GB RAM are enough for a small local model, not
   for training a general foundation model.
6. **Hosted operations.** No auth, rate limiting, multi-tenant isolation or
   incident response; hosting is deliberately out of scope.
7. **Release.** No PyPI/GitHub release, no owner sign-off, no final token.

## Optional local provider backend

The server can route free-form chat to any OpenAI-compatible local endpoint
while keeping RAG, tools and memory local. The default remains the 50M model.

```powershell
python -m streamft.chat_server `
  --model checkpoints/transformer_behavior_50m_scaled `
  --tokenizer data/chat_en_bpe8k.json `
  --docs data/docs.db `
  --chat-provider-url http://127.0.0.1:11434/v1 `
  --chat-provider-model llama3.2:3b `
  --allow-general-chat
```

Observed live on this machine:

```text
chat_provider: {'enabled': True, 'model': 'llama3.2:3b',
                'host': '127.0.0.1:11434', 'general_chat': True}

Q: what is the launch code?
   -> local note with source launch.txt (RAG path, no provider)

Q: Explain why the sky is blue in two sentences.
   -> provider answer, sources: []
```

The provider is not used when a local source answers. With
`--allow-general-chat`, it may answer unsourced questions; without it, the
source-grounded no-source reply is preserved. A remote provider URL would send
prompts off the machine, so local Ollama is the recommended configuration.
Evidence: [`chat_provider_live.json`](../research/results/chat_provider_live.json).

## Could this become a "super tool"?

**As a domain tool: yes, plausibly.** A curated local corpus plus tools plus a
3B-14B local model can be a strong private assistant for a specific job
(operations, support, research triage) without a frontier model.

**As a general superintelligence: no, not from this artifact.** The gap is not
a prompt or a UI feature. It is parameters, data, compute, post-training and
evaluation. A 49M model trained on 175M tokens is a narrow behavior model; the
harness around it is the reusable asset.

The realistic ladder:

| rung | what it buys | feasibility here |
|---|---|---|
| current 49.2M | narrow chat, RAG, tools | done |
| 100M-300M + more data | better fluency, still weak reasoning | possible but needs a GPU run and a new data gate |
| 1B+ local model via provider | general chat, coding help, broader knowledge | already possible with Ollama; not the 50M core |
| 1B+ trained from scratch | general capability | out of scope on 6 GB VRAM |
| frontier model | broad reasoning/agency | not this project |

## Recommended next decisions

1. Ship 0.8.0 as a **narrow local assistant**, not a general AI claim.
2. Keep the provider backend optional and local-first; it is the cheapest way
   to give users a stronger model without changing the core.
3. If general capability is the goal, commission a larger model/data run
   (300M-1B, much more tokens) and define a real evaluation gate first.
4. For a domain product, invest in corpus quality, tool coverage and evals
   rather than raw parameter count.

## Safety position

The safety filter is **off by default** and is an optional deployment tool, not
a content policy. This project does not operate a hosted service and does not
tell users what they may run locally. If a host enables the filter, rules are
the exact fallback and the local ML head is additive. The docs describe what
the filter does, not what users must do.
