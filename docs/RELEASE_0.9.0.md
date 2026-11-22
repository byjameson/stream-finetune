# streamft 0.9.0 - document provenance, evidence and constrained plans

Status: **published to PyPI as `streamft 0.9.0` on 2026-11-22.**

> **Packaging note:** `0.9.0` moved the trainer dependencies to `[train]`.
> Use `0.9.1` for the backward-compatible default install; the Atom Zero
> features below are unchanged.

This release is about the document layer, not a new model. The 49.2M weights and
the existing product/behavior benchmarks are unchanged from `0.8.0`.

## Highlights

- **PDF, DOCX, HTML and Markdown ingestion** with local extraction.
- **Page and section citations** for every retrieved chunk; PDF keeps 1-based
  page numbers, DOCX/HTML/Markdown keep heading sections.
- **Evidence panel / API** (`GET /v1/documents/evidence`) that opens the exact
  local chunks behind a citation.
- **100-question document benchmark** covering source, page/section, answer hit,
  no-answer refusal and latency; CI enforces the thresholds.
- **Constrained plan API** (`POST /v1/agent/plan`) with an allowlist, a
  three-step cap and explicit confirmation for task/research side effects.
  This is a validator/executor, not model-driven planning.

## Review fixes

- Raw uploads now bypass the 2 MB JSON body cap and use the 10 MB document
  limit, so a 2-10 MB PDF can be indexed. A 2.7 MB regression upload is covered
  by the HTTP test.
- Upload responses separate `file_sha256` (raw bytes) from `content_sha256`
  (extracted text); source listings expose the stored content hash.
- DOCX paragraphs, headings and tables are read in document order, so a table
  no longer moves to the end under the last heading.
- The default install is the assistant runtime; `streamft[train]` carries the
  training stack and `streamft[neural]` the optional MiniLM reranker.
- ModelScope receives the card without HF YAML front matter; keeping HF
  front matter on ModelScope caused its model page to return 500.

## Document benchmark

Authored regression corpus, 10 generated documents, 80 answerable questions and
20 no-answer questions:

| metric | result |
|---|---:|
| top-1 source | **1.000** (80/80) |
| top-1 page/section | **1.000** (80/80) |
| complete citation | **1.000** (80/80) |
| no-answer refusal | **1.000** (20/20) |
| median retrieval latency | **~1-2 ms** |

This is not a public leaderboard. It is a deterministic regression gate for
ingestion, provenance, source routing and the no-answer path. See
[`BENCHMARKS.md`](BENCHMARKS.md) and
[`../research/results/document_benchmark.json`](../research/results/document_benchmark.json).

## Install profiles

The default install is now the assistant runtime. The training stack moved to
extras:

```powershell
pip install streamft                  # core assistant + PDF/DOCX
pip install "streamft[neural]"        # optional MiniLM reranker
pip install "streamft[train]"         # streaming trainer / quantization
pip install "streamft[train,neural]"  # both
```

This keeps a CPU assistant install from pulling `bitsandbytes`, `datasets`,
`accelerate` and the training stack by default. Existing trainer users should
install `streamft[train]`.

## Tests and CI

- **412 tests pass** locally.
- CI runs the unit tests, the document benchmark with `--check`, the safety and
  injection red-team, and the package build/install.

## Limits

- No OCR; scanned PDFs need OCR before ingestion.
- Encrypted PDFs are rejected.
- DOCX page numbers are not available; section headings are the locator.
- Upload cap: 10 MB.
- Complex tables may lose structure.
- The plan API does not let the 49M model choose tools freely.
- Safety remains optional and off by default.
- The provider backend remains text-only and optional.

## Install

```powershell
pip install "streamft[neural]==0.9.0"
```

## Links

- Repository: <https://github.com/byjameson/stream-finetune>
- PyPI: <https://pypi.org/project/streamft/0.9.0/>
- Full project map: <https://github.com/byjameson/stream-finetune/blob/main/docs/PROJECT_OVERVIEW.md>
- Documents: <https://github.com/byjameson/stream-finetune/blob/main/docs/DOCUMENTS.md>
- Tool plans: <https://github.com/byjameson/stream-finetune/blob/main/docs/TOOL_PLANNING.md>
- Benchmarks: <https://github.com/byjameson/stream-finetune/blob/main/docs/BENCHMARKS.md>
- Hugging Face: <https://huggingface.co/byjameson/atom-zero-49m>
- ModelScope: <https://modelscope.cn/models/byjameson/atom-zero-49m>
- Medium: <https://medium.com/@myredditacc9393/atom-zero-a-49m-local-assistant-that-says-i-dont-know-df1ebc0e2e78>
- Contact: **gokhancava@gmail.com**
