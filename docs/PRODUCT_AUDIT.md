# Atom Zero product audit

Status: **development; not released**. This is a full-surface audit, not a
release approval. Last updated 2026-11-21.

## 1. Model and data

| item | state | evidence |
|---|---|---|
| scaled 50M corpus | 175.37M clean train tokens, 3.21M valid | `docs/DATA_SCALING.md` |
| replay mix | 214.8M tokens (chat + behavior 4x + skill 3x) | `data/chat_en_scaled_replay_meta.json` |
| 25k candidate | promoted to `transformer_behavior_50m_scaled` | `research/results/eval_scaled_25k.json` |
| old valid | bpb 1.3596 / assistant 3.8722 vs base 1.3719 / 4.0502 | same |
| scaled valid | bpb 1.4189 / assistant 3.9655 vs base 1.5032 / 4.2153 | `eval_base_scaled_valid.json` |
| product benchmark | 21/21; 26/26 with safety | `assistant_benchmark_scaled_25k*.json` |
| behavior | 16/16 | `behavior_scaled_25k.json` |
| raw knowledge | 1/10 (2/10 base) | `raw_knowledge_scaled_25k.json` |
| 60k run | in progress; may replace 25k if better | `data/scaled_training.log` |

**Gap:** raw 50M knowledge is still weak. Facts belong in RAG; general
reasoning/coding needs a larger model and much more data. Not solved.

## 2. Runtime and agent

- Deterministic tools: time, arithmetic, task add/list/complete, name/likes
  memory, help/limits.
- RAG: SQLite FTS5, source/title/sha, teacher-note metadata, query-relevant
  snippet selection (`best_snippet`).
- Missing-source path: honest no-source reply plus knowledge-gap record.
- Research tool: explicit `research/look up`; queue worker for approved topics.
- Safety: opt-in filter before generation and on output.
- Session memory: JSON facts/tasks/episodes; stable browser session ID now
  persisted in `localStorage`; New session button.
- Deletion: exact document source and session memory; path separators handled.

**Gaps:** no semantic retrieval/reranker; no multilingual model; no task due
dates/reminders; no multi-document conflict resolution.

## 3. Browser UI

- Chat, tool chips, feedback buttons, learning queue, research queue, teacher
  notes/provider, knowledge gaps, indexed-document list, per-source Forget,
  session Forget, safety status, model/adapter/health status.
- Session ID survives reload; `POST /v1/memory/history` restores the visible
  transcript; no hidden telemetry.
- Verified in browser: session ID same after reload, transcript restored
  (`my name is Alice`), document list shows Berlin, no console errors.

Adapter version history is shown read-only in the UI; rollback remains an
explicit CLI action (`learning_loop.py activate vNNNN`) to avoid hot-swapping
weights under a running request.

**Gaps:** no import/export memory; no UI-triggered adapter activation (by
design until safe hot-swap exists).

## 4. Research and teacher

- Reviewed queue: pending -> approved -> running -> done/error/rejected.
- Worker: lock, stale recovery, JSONL log, loop mode, daily Windows task.
- Berlin worker run done with a cited Wikipedia source.
- Ollama 0.40.16 + `llama3.2:3b` (2.0 GB) installed; live `TeacherProvider`
  returned `Hello!`; Ankara worker stored an unverified `teacher://ollama/...`
  note.
- Task is disabled during GPU training to avoid VRAM contention; re-enable
  after training.

**Gaps:** no automatic teacher verification (by design); teacher model is
separate and has its own licence; task currently disabled.

## 5. Self-learning

- Feedback -> human review -> approved JSONL -> gated LoRA -> `vNNNN`.
- Gate: target gain, clean bpb/assistant regression, behavior, smoke.
- Holdout evaluation and `activate` rollback.
- 8-example bootstrap accepted; 6/2 holdout improved loss.
- Old product adapter loads on the scaled base (live server check).

**Gaps:** no large reviewed feedback corpus yet; adapter UI rollback is CLI
only; general capability is not improved by adapters.

## 6. Security and privacy

- Opt-in input/output safety filter; red-team 30/30, output 4/4, agent 30/30.
- Prompt-injection 4/4; untrusted text quoted as data, tools not executed.
- PII/source deletion tested; document list Forget UI.
- Local bind default; SSRF guard; URL fetch opt-in; upload caps.
- Apache-2.0 code/weights; data attribution in `NOTICE`.

**Gaps:** pattern-based filter (not semantic); no multi-language red-team; no
hosted auth/rate limiting because no hosted service is operated; per-source
data-licence review still recommended.

## 7. Tests and CI

- **318 tests** pass locally.
- CI runs CPU tests, deterministic safety/injection red-team, and package build.
- Latest feature commit verified green before the final UI changes.

## 8. Release state

- No PyPI/GitHub model release in this step.
- `pyproject.toml` is **0.8.0.dev0**; no release will be tagged/published until
  the product is complete and the owner supplies a fresh token.
- PyPI token must be supplied by the owner at publication time.
- Open-source: anyone may host; this project does not operate a service.

## Priority gaps

1. Finish 60k scaled run and gate it against 25k.
2. Run final full benchmark/behavior/safety suite on the final checkpoint.
3. Add adapter history/rollback to the browser.
4. Add transcript restore or memory export/import if desired.
5. Expand multilingual/semantic safety and RAG reranking.
6. Larger model/general AI remains a separate compute/data decision.
