# streamft 0.9.0 - document provenance, evidence and constrained plans

Status: **release candidate on `main`; not yet published to PyPI.**

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

## Document benchmark

Authored regression corpus, 10 generated documents, 80 answerable questions and
20 no-answer questions:

| metric | result |
|---|---:|
| top-1 source | **1.000** (80/80) |
| top-1 page/section | **1.000** (80/80) |
| complete citation | **1.000** (80/80) |
| no-answer refusal | **1.000** (20/20) |
| median retrieval latency | **~1.4 ms** |

This is not a public leaderboard. It is a deterministic regression gate for
ingestion, provenance, source routing and the no-answer path. See
[`BENCHMARKS.md`](BENCHMARKS.md) and
[`../research/results/document_benchmark.json`](../research/results/document_benchmark.json).

## Tests and CI

- **410 tests pass** locally.
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

## Install (after publication)

```powershell
pip install streamft==0.9.0
```

## Links

- Repository: <https://github.com/byjameson/stream-finetune>
- Full project map: <https://github.com/byjameson/stream-finetune/blob/main/docs/PROJECT_OVERVIEW.md>
- Documents: <https://github.com/byjameson/stream-finetune/blob/main/docs/DOCUMENTS.md>
- Tool plans: <https://github.com/byjameson/stream-finetune/blob/main/docs/TOOL_PLANNING.md>
- Benchmarks: <https://github.com/byjameson/stream-finetune/blob/main/docs/BENCHMARKS.md>
- Hugging Face: <https://huggingface.co/byjameson/atom-zero-49m>
- ModelScope: <https://modelscope.cn/models/byjameson/atom-zero-49m>
- Medium: <https://medium.com/@myredditacc9393/atom-zero-a-49m-local-assistant-that-says-i-dont-know-df1ebc0e2e78>
- Contact: **gokhancava@gmail.com**
