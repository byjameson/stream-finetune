# Atom Zero product audit

Status: **released 0.8.0; 0.8.1 image fix; 0.8.2 contact metadata**. This is a
full-surface audit. Last updated 2026-11-22.

## 1. Model and data

| item | state | evidence |
|---|---|---|
| scaled 50M corpus | 175.37M clean train tokens, 3.21M valid | `docs/DATA_SCALING.md` |
| replay mix | 214.8M tokens (chat + behavior 4x + skill 3x) | `data/chat_en_scaled_replay_meta.json` |
| 40k candidate | promoted to `transformer_behavior_50m_scaled` | `research/results/eval_scaled_40k.json` |
| scaled valid | bpb 1.3815 / assistant 3.8551 vs base 1.5032 / 4.2153 | same |
| product benchmark | 21/21; safety suite 26/26 with ML filter | `assistant_benchmark_scaled_40k_ml_safety*.json` |
| retrieval benchmark | hybrid neural top-1/citation 1.000 vs FTS 0.850; FTS recall@10 0.900 vs hybrid 1.000; forced ANN also 1.000 | `retrieval_benchmark_ann.json` |
| vector index | exact 3000 chunks: 0.77 ms vs scan 127.6 ms (166x); IVF ANN 20k: 0.64 ms vs 3.22 ms (5x), recall@10 1.000 | `vector_index_benchmark.json`, `ann_index_benchmark.json` |
| behavior | 16/16 | `behavior_scaled_40k_ann.json` |
| raw knowledge | 2/10 (2/10 base) | `raw_knowledge_scaled_40k.json` |
| training | stopped at 40k; resume to 48k did not beat it materially | `data/scaled_training.log`, `data/scaled_60k_resume.log` |
| resume gate | 44k candidate 1.3800/3.8497 vs 40k 1.3815/3.8551; product 21/21, behavior 16/16, safety 26/26 | `eval_resume_44k.json`, `assistant_benchmark_resume_44k*.json` |

**Gap:** raw 50M knowledge is still weak. Facts belong in RAG; general
reasoning/coding needs a larger model and much more data. Not solved. The
lower-LR resume did not produce a material improvement over 40k.

## 2. Runtime and agent

- Deterministic tools: time, arithmetic, task add/list/complete, name/likes
  memory, help/limits.
- RAG: SQLite FTS5, source/title/sha, teacher-note metadata, query-relevant
  snippet selection (`best_snippet`). Hybrid candidate stage adds a cached
  embedding pool (FTS + neural cosine) before reranking, served from a
  prewarmed in-memory matrix.
- Missing-source path: honest no-source reply plus knowledge-gap record.
- Research tool: explicit `research/look up`; queue worker for approved topics.
- Curiosity queue: pending knowledge gaps can be proposed as `kind=curiosity`
  research items (`POST /v1/curiosity/propose`, browser button). Approval is
  still required before Wikipedia/teacher calls; no weights change.
- Optional local provider: `--chat-provider-url/model` routes free-form chat to
  an Ollama/OpenAI-compatible endpoint; RAG/tools/memory stay local. Default is
  still the 50M model. `--allow-general-chat` lets the provider answer when no
  source matches; without it, the no-source reply is preserved.
- Safety: opt-in rule-first filter plus a local MiniLM linear ML head before
  generation and on output; missing model falls back to exact rules. Default
  is off, and it is an operator tool rather than a content/moral policy.
- Session memory: JSON facts/tasks/episodes; stable browser session ID now
  persisted in `localStorage`; New session button.
- Deletion: exact document source and session memory; path separators handled.

**Gaps:** the neural reranker needs a one-time model download (`models/` is
gitignored), so a fresh clone falls back to hashed n-grams; that fallback stays
FTS-anchored and lexical (it does not answer semantic-only queries), so its
recall remains 0.900. The neural hybrid path reaches 1.000 on the authored set.
The vector index is IVF ANN, not HNSW; recall is measured at **1.000** on the
clustered benchmark and the authored retrieval set, but this is not a guarantee
for every corpus, and low-confidence queries pay an exact-scan fallback. Memory
grows linearly with chunks and startup pays the build cost. The model is
English-first; no task due dates/reminders; no multi-document conflict
resolution. The provider backend is text-only: it does not plan or execute
tools, and unsourced provider answers are not fact-checked. See
[`POTENTIAL.md`](POTENTIAL.md) for the honest capability ladder.

## 3. Browser UI

- One-command quickstart seeds a demo corpus and opens the UI
  (`scripts/quickstart_atomzero.ps1`); architecture and launch assets in
  `docs/figures/atomzero_architecture.svg`, `docs/LAUNCH.md`.
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
separate and has its own licence. Curiosity is gap-driven, not a general
self-directed research agent; proposals still require human approval.

## 5. Self-learning

- Feedback -> human review -> approved JSONL -> gated LoRA -> `vNNNN`.
- Gate: target gain, clean bpb/assistant regression, behavior, smoke.
- Holdout evaluation and `activate` rollback.
- 8-example bootstrap accepted; 6/2 holdout improved loss.
- Old product adapter loads on the scaled base (live server check).

**Gaps:** no large reviewed feedback corpus yet; adapter UI rollback is CLI
only; general capability is not improved by adapters.

## 6. Security and privacy

- Opt-in input/output safety filter v0.3: exact rule layer v0.2 (English regex
  plus TR/DE/ES/FR/IT/PT/NL/RU lexical-semantic rules) followed by a local
  MiniLM + linear ML head. Red-team **72/72**, output **4/4**, agent **72/72**,
  benign false blocks **0/26**; held-out semantic paraphrases **10/10** (rules
  only 1/10). ML-only on the authored 72-case set is 61.1%, but rules remain
  authoritative. Missing/broken head or model returns the exact rules.
- RAG reranker: deterministic hashed character n-gram embeddings + cosine,
  query coverage, title match, exact phrase and bm25; tests cover a case where
  raw bm25 prefers the wrong document and a morphological match.
- Hybrid neural retrieval: local `sentence-transformers/all-MiniLM-L6-v2`
  (BERT, 384 hidden), mean pooling and cosine, plus cached chunk vectors in
  SQLite. The candidate pool is FTS + neural cosine (threshold 0.30). Held-out
  authored benchmark: **top-1/citation 1.000** vs FTS **0.850**; candidate
  recall@10 **1.000** vs **0.900**; median retrieval latency **8.8 ms vs
  0.34 ms**. The hashed fallback is FTS-anchored and stays at 0.850; it does
  not answer from hash-only candidates. Product 21/21 and safety 26/26 hold on
  both paths. Vector rows are pruned on source replace/forget.
- Vector index: `DocMemory.prewarm_vectors()` builds one numpy matrix over all
  cached chunk vectors after model/reranker load; `--no-prewarm` skips it.
  Above 256 chunks, auto mode builds a deterministic k-means **IVF ANN** index;
  `--vector-index ann|exact|auto` overrides it. Exact cosine is the fallback on
  empty/short results, ANN errors, or low ANN confidence. Benchmarks: exact
  matrix **0.77 ms** vs linear scan **127.6 ms** (**166x**, top-5 identical);
  IVF ANN at 20k vectors **0.64 ms** vs **3.22 ms** exact (**5x**, recall@10
  **1.000**, probe 4). `/health` reports mode, chunks, dim, nlist/nprobe and
  fallback count. Writes invalidate the index and the next search rebuilds it.
- Prompt-injection 4/4; untrusted text quoted as data, tools not executed.
- PII/source deletion tested; document list Forget UI.
- Local bind default; SSRF guard; URL fetch opt-in; upload caps.
- Apache-2.0 code/weights; data attribution in `NOTICE`.

**Gaps:** the ML layer is a small linear head over frozen MiniLM embeddings,
trained on 349 synthetic examples, not a full moderation model; MiniLM is
English-centric, so multilingual coverage still comes from rules; category
attribution is not guaranteed out of distribution. No hosted auth/rate limiting
because no hosted service is operated; per-source data-licence review still
recommended.

## 7. Tests and CI

- **392 tests** pass locally.
- CI runs CPU tests, deterministic safety/injection red-team, and package build.
- CI `abe47b4` green for the 0.8.0 release commit.

## 8. Release state

- GitHub releases **v0.8.0** (`abe47b4`), **v0.8.1** (`0e151b5`, PyPI
  README image fix) and **v0.8.2** (`3ebe0a5`, contact metadata) published;
  repository description/topics updated.
- PyPI package **0.8.3** current; wheel and sdist twine-checked.
- Canonical full project map: `docs/PROJECT_OVERVIEW.md`.
- Public contact: **gokhancava@gmail.com** and
  <https://www.linkedin.com/in/g%C3%B6khan-cava-874a32157/>.
- PyPI **streamft 0.8.0** published; **0.8.1** fixed the README image;
  **0.8.2** adds contact metadata:
  `https://pypi.org/project/streamft/0.8.2/`.
- `pyproject.toml` is **0.8.3** (full project map); model weights remain
  `0.8.0`.
- Hugging Face and ModelScope weights published:
  `byjameson/atom-zero-49m` on both platforms; dev.to launch article published.
- The PyPI token used for upload was pasted in chat; the owner should revoke and
  rotate it. The temporary Hugging Face upload token was deleted after use.
- Open-source: anyone may host; this project does not operate a service.

## Priority gaps

1. General model capability: raw-model knowledge is still **2/10**. The
   provider backend is the current escape hatch; training a larger model is a
   separate compute/data decision. See [`POTENTIAL.md`](POTENTIAL.md).
2. Provider-backed tool planning: the provider answers text but does not choose
   or execute tools.
3. A larger human-labelled moderation set, especially non-English, if a host
   enables the optional filter.
4. Final owner completeness sign-off and PyPI token.
5. Optional HNSW graph replacing IVF if recall/latency needs it.
6. Hosted deployment (auth, rate limits, monitoring) is out of scope.
