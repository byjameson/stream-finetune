# Atom Zero product audit

Status: **development; not released**. This is a full-surface audit, not a
release approval. Last updated 2026-11-21.

## 1. Model and data

| item | state | evidence |
|---|---|---|
| scaled 50M corpus | 175.37M clean train tokens, 3.21M valid | `docs/DATA_SCALING.md` |
| replay mix | 214.8M tokens (chat + behavior 4x + skill 3x) | `data/chat_en_scaled_replay_meta.json` |
| 40k candidate | promoted to `transformer_behavior_50m_scaled` | `research/results/eval_scaled_40k.json` |
| scaled valid | bpb 1.3815 / assistant 3.8551 vs base 1.5032 / 4.2153 | same |
| product benchmark | 21/21; safety suite 26/26 | `assistant_benchmark_scaled_40k*.json` |
| behavior | 16/16 | `behavior_scaled_40k.json` |
| raw knowledge | 2/10 (2/10 base) | `raw_knowledge_scaled_40k.json` |
| training | stopped at 40k; 38k/40k plateau | `data/scaled_training.log` |

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

**Gaps:** the reranker is lexical (coverage/title/phrase/bm25), not an
embedding reranker; the model is English-first; no task due dates/reminders;
no multi-document conflict resolution.

## 3. Browser UI

- Chat, tool chips, feedback buttons, learning queue, research queue, teacher
  notes/provider, knowledge gaps, indexed-document list, document search,
  per-source Forget, session Forget, adapter-version history, safety status,
  model/adapter/health status.
- Session ID survives reload; `POST /v1/memory/history` restores the visible
  transcript; no hidden telemetry.
- Verified in browser: session ID same after reload, transcript restored
  (`my name is Alice`), document list shows Berlin, no console errors.

Adapter version history is shown read-only in the UI; rollback remains an
explicit CLI action (`learning_loop.py activate vNNNN`) to avoid hot-swapping
weights under a running request.

Session memory can be exported/imported as JSON from the browser; the export
contains personal data and should be kept private. `/v1/memory/history`
restores the visible transcript.

**Gaps:** no UI-triggered adapter activation (by design until safe hot-swap
exists).

## 4. Research and teacher

- Reviewed queue: pending -> approved -> running -> done/error/rejected.
- Worker: lock, stale recovery, JSONL log, loop mode, daily Windows task.
- Berlin worker run done with a cited Wikipedia source.
- Ollama 0.40.16 + `llama3.2:3b` (2.0 GB) installed; live `TeacherProvider`
  returned `Hello!`; Ankara worker stored an unverified `teacher://ollama/...`
  note.
- Task was disabled during GPU training and is now **Ready** again; next run
  22.11.2026 09:00. It uses the local Ollama teacher with Wikipedia fallback.

**Gaps:** no automatic teacher verification (by design); teacher model is
separate and has its own licence.

## 5. Self-learning

- Feedback -> human review -> approved JSONL -> gated LoRA -> `vNNNN`.
- Gate: target gain, clean bpb/assistant regression, behavior, smoke.
- Holdout evaluation and `activate` rollback.
- 8-example bootstrap accepted; 6/2 holdout improved loss.
- Old product adapter loads on the scaled base (live server check).

**Gaps:** no large reviewed feedback corpus yet; adapter UI rollback is CLI
only; general capability is not improved by adapters.

## 6. Security and privacy

- Opt-in input/output safety filter v0.2 with English regex plus
  language-agnostic lexical-semantic rules (TR/DE/ES/FR); red-team **48/48**,
  output 4/4, agent 48/48.
- Lexical RAG reranker: query coverage, title match, exact phrase and bm25
  normalization; tests cover a case where raw bm25 prefers the wrong document.
- Prompt-injection 4/4; untrusted text quoted as data, tools not executed.
- PII/source deletion tested; document list Forget UI.
- Local bind default; SSRF guard; URL fetch opt-in; upload caps.
- Apache-2.0 code/weights; data attribution in `NOTICE`.

**Gaps:** rule-based filter (regex + lexical-semantic rules, not an ML
moderation model); no hosted auth/rate limiting because no hosted service is
operated; per-source data-licence review still recommended.

## 7. Tests and CI

- **341 tests** pass locally.
- CI runs CPU tests, deterministic safety/injection red-team, and package build.
- Latest feature commit verified green before the final UI changes.

## 8. Release state

- No PyPI/GitHub model release in this step.
- `pyproject.toml` is **0.8.0.dev0**; no release will be tagged/published until
  the product is complete and the owner supplies a fresh token.
- PyPI token must be supplied by the owner at publication time.
- Open-source: anyone may host; this project does not operate a service.

## Priority gaps

1. Finish 60k scaled run and gate it against 30k.
2. Run final full benchmark/behavior/safety suite on the final checkpoint.
3. Add memory export/import if desired.
4. Expand multilingual/semantic safety and RAG reranking.
5. Larger model/general AI remains a separate compute/data decision.
