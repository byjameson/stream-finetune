# Atom Zero 0.8.0 — open-source scaled assistant

Status: **development candidate; not released**. `pyproject.toml` is
`0.8.0.dev0`; PyPI upload waits for the owner's fresh token and a final
completeness sign-off. The project is open source under Apache-2.0. No hosted
service is operated; third-party hosts are responsible for their own deployment
controls.

## What changed

- **Scaled 50M base:** 175.37M clean chat tokens (6.5x the original) plus
  behavior/skill replay. The current product candidate is
  `transformer_behavior_50m_scaled`, promoted at 30k steps; the 60k run is
  being evaluated.
  - old clean valid: bpb **1.3522** vs 1.3719 (-1.4%), assistant **3.8532**
    vs 4.0502 (-4.9%);
  - scaled valid: chat bpb **1.4026** vs 1.5032, assistant **3.9151** vs
    4.2153.
- **Reviewed learning loop:** Good/Bad feedback, learning queue, gated LoRA,
  immutable `vNNNN` versions, holdout evaluation, `activate` rollback.
- **Knowledge gaps:** no-source questions become reviewed gaps; a proposal
  creates a pending research topic; the worker fetches only after approval.
  Verified Paris/Berlin source citations and an Ollama Ankara teacher note.
- **Scheduled research worker:** lock, stale recovery, JSONL log, daily Windows
  task (disabled during GPU training; re-enable after).
- **Ollama teacher:** local `llama3.2:3b`, OpenAI-compatible endpoint,
  unverified teacher notes, graceful Wikipedia-only fallback.
- **Safety:** opt-in `--safety-filter`, input/output refusal, red-team
  filter 30/30, output 4/4, agent 30/30; product benchmark 26/26 with filter.
- **Privacy:** exact-source and session-memory deletion, path-separator aware,
  SQLite `VACUUM`; tests in `tests/test_privacy.py`.
- **Prompt injection:** untrusted-document checks 4/4; quoted data cannot
  execute tools, add tasks, spoof citations or disable safety.
- **License:** Apache-2.0 for code and weights; data attribution in `NOTICE`.

## Benchmarks (honest scope)

- Product behavior: 21/21 base and adapter; 26/26 with safety.
- Raw model without RAG/tools: 2/10 base, 1/10 scaled 25k. It is **not** a
  general model; facts belong in RAG.
- Behavior checks: 16/16 on the scaled 25k candidate.
- Test suite: 318 tests; CI green.

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

## Known limits

- 50M is narrow; general reasoning/coding and reliable unaided facts are out
  of reach. The raw probe is 1-2/10.
- The safety filter is pattern-based, not full moderation.
- The 60k scaled run is still being evaluated; 25k is the promoted candidate.
- Public release still needs the owner's PyPI token and final sign-off.
