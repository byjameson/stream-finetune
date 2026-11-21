# Atom Zero 0.8.0 — open-source scaled assistant (2026-11-22)

Status: **released 0.8.0** (GitHub tag `v0.8.0`; PyPI
[`streamft==0.8.0`](https://pypi.org/project/streamft/0.8.0/); weights on
[Hugging Face](https://huggingface.co/byjameson/atom-zero-49m) and
[ModelScope](https://modelscope.cn/models/byjameson/atom-zero-49m)). The owner
approved publication. The project is open source under Apache-2.0. No hosted
service is operated; third-party hosts are responsible for their own deployment
controls.

## What changed

- **Scaled 50M base:** 175.37M clean chat tokens (6.5x the original) plus
  behavior/skill replay. The current product candidate is
  `transformer_behavior_50m_scaled`, promoted at 40k steps; a lower-LR resume
  to 48k did not materially beat it, so 40k remains the product checkpoint.
  - clean scaled valid: chat bpb **1.3815** vs 1.5032, assistant **3.8551**
    vs 4.2153; stories 1.4144 vs 1.6221; mixed 1.2406 vs 1.3743.
  - The old clean valid shard is **not** a clean holdout for this run because
    the larger corpus was regenerated from the same sources; its numbers are
    not used for promotion.
- **Reviewed learning loop:** Good/Bad feedback, learning queue, gated LoRA,
  immutable `vNNNN` versions, holdout evaluation, `activate` rollback.
- **Knowledge gaps:** no-source questions become reviewed gaps; a proposal
  creates a pending research topic; the worker fetches only after approval.
  Verified Paris/Berlin source citations and an Ollama Ankara teacher note.
- **Scheduled research worker:** lock, stale recovery, JSONL log, daily Windows
  task (disabled during GPU training; re-enable after).
- **Ollama teacher:** local `llama3.2:3b`, OpenAI-compatible endpoint,
  unverified teacher notes, graceful Wikipedia-only fallback.
- **Safety:** opt-in `--safety-filter`, exact rules first plus a local MiniLM
  linear ML head; red-team 72/72, output 4/4, agent 72/72, benign false blocks
  0/26, held-out semantic paraphrases 10/10; product benchmark 26/26 with
  filter. Missing model/head falls back to exact rules; `--no-safety-ml`
  forces the rule baseline.
- **Privacy:** exact-source and session-memory deletion, path-separator aware,
  SQLite `VACUUM`; tests in `tests/test_privacy.py`.
- **Prompt injection:** untrusted-document checks 4/4; quoted data cannot
  execute tools, add tasks, spoof citations or disable safety.
- **License:** Apache-2.0 for code and weights; data attribution in `NOTICE`.

## Benchmarks (honest scope)

Shareable model card and raw-vs-augmented table:
[`MODEL_CARD.md`](MODEL_CARD.md). One-command demo:
[`QUICKSTART_ATOMZERO.md`](QUICKSTART_ATOMZERO.md). Launch notes:
[`LAUNCH.md`](LAUNCH.md).

- Raw 49.2M without RAG/tools: **2/10**; adapter also **2/10**.
- Product behavior: 21/21 base and adapter; 26/26 with safety; unchanged with
  the neural reranker.
- Retrieval, held-out authored set: FTS top-1/citation 0.850; hybrid
  FTS+MiniLM neural **1.000**, candidate recall@10 **1.000** vs 0.900, at
  8.8 ms vs 0.34 ms median. Hashed fallback stays FTS-anchored.
- Prewarmed vector index: 3000 chunks, 0.77 ms matrix query vs 127.6 ms linear
  scan (166x), identical top-5.
- IVF ANN index: 20k vectors, 0.64 ms vs 3.22 ms exact (5x), recall@10 1.000;
  exact fallback on low confidence/errors. Auto enables it above 256 chunks;
  forced ANN kept retrieval top-1 at 1.000. This is IVF, not HNSW.
- Raw model without RAG/tools: 2/10 base, 2/10 product checkpoint, 2/10 with
  the adapter. It is **not** a general model; facts belong in RAG.
- Behavior checks: 16/16 on the scaled 40k product checkpoint.
- Optional local provider backend: local Ollama/OpenAI-compatible free-form
  chat with RAG/tools/memory still local; default remains the 50M model.
  Tested live with `llama3.2:3b`; see `docs/POTENTIAL.md`.
- Curiosity queue: pending gaps become pending research topics; approval is
  required before Wikipedia/teacher calls. Live before/after RAG citation
  evidence in `research/results/curiosity_live.json`.
- Test suite: 392 tests; CI green.

## Run locally

```bash
pip install streamft==0.8.0   # after publication
python -m streamft.chat_server \
  --model checkpoints/transformer_behavior_50m_scaled \
  --adapter auto --docs data/docs.db --safety-filter
```

Or:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/demo_all.ps1 -SafetyFilter
```

## Support

Canonical donation addresses and network warnings:
[`DONATE.md`](DONATE.md). There is no paid support, token, or investment offer;
donations do not buy influence over results or reviews.

## Known limits

- 50M is narrow; general reasoning/coding and reliable unaided facts are out
  of reach. The raw probe is **2/10**.
- The safety filter is a small local ML head over frozen MiniLM embeddings
  plus exact rules, not a full moderation system; it is off by default.
- The optional provider is text-only, does not plan tools, and unsourced
  answers are not fact-checked.
- The 40k checkpoint is the product; the 60k resume did not beat it materially.
- PyPI publication still needs the owner's token; the GitHub tag/source
  release can be created without it.
