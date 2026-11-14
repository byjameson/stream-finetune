# Local verified assistant

The product-facing model is now a pretrained instruction model plus retrieval
over the verified knowledge ledger. The from-scratch spiking LM remains a
research line; its 50M/500M-token run did not produce a product-quality model.

## Why this path

A 4.3M/21M/50M TinyStories model can write story fragments, but it cannot be a
general assistant. The 50M run at 1,500 steps reached 1.3256 bpb, essentially
the same as the 21M model (1.327) and the 4.3M model (1.336); more parameters did
not buy quality. The product needs a pretrained language model with instruction
following, plus a verified retrieval layer.

## Architecture

```
user question
      │
      ├─ knowledge_fts (verified docs only)
      │       └─ top-k context with source URLs
      ├─ Qwen2.5-1.5B-Instruct chat template
      ├─ generate answer with citations
      └─ /v1/chat/completions + sources
```

The model is local (`models/Qwen2.5-1.5B-Instruct`). The retrieval index is built
from `documents.status='verified'` in the knowledge SQLite DB; untrusted or
contradicted documents never reach the prompt.

## Commands

Ask once:

```bash
streamft assistant ask \
  --model models/Qwen2.5-1.5B-Instruct \
  --db data/knowledge.db \
  --question "What is the speed of light?"
```

Refresh the index after adding/verifying knowledge:

```bash
streamft assistant index --db data/knowledge.db
```

Serve:

```bash
streamft assistant serve \
  --model models/Qwen2.5-1.5B-Instruct \
  --db data/knowledge.db \
  --host 127.0.0.1 --port 8014
```

API:

```bash
curl -s http://127.0.0.1:8014/health

curl -s -X POST http://127.0.0.1:8014/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"What is the speed of light?"}],"max_tokens":128}'

curl -s -X POST http://127.0.0.1:8014/knowledge/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"speed of light","k":3}'
```

`ASSISTANT_API_KEY` or `--api-key` protects the POST endpoints; `/health`,
`/metrics` and `/v1/models` stay public.

## Demo

With four verified docs in `data/knowledge_demo.db`:

```json
{
  "answer": "The speed of light in vacuum is approximately 299,792,458 metres per second. This value is consistent across different verified contexts such as [1], [2], and [3].",
  "sources": [
    {"id": 2, "title": "Speed of light A", "source": "https://a.example/physics"},
    {"id": 3, "title": "Speed of light B", "source": "https://b.example/physics"},
    {"id": 4, "title": "Speed of light C", "source": "https://c.example/physics"}
  ]
}
```

The HTTP endpoint returned the same answer with `200`, and `/knowledge/search`
returned the three corroborating documents.

## Knowledge loop

```text
knowledge add -> verify (official / corroboration / review / NLI)
              -> export verified docs
              -> assistant index
              -> answer with citations
```

To add a new verified source:

```bash
streamft spiking knowledge --db data/knowledge.db add \
  --text "..." --source https://... --trust reference
streamft spiking knowledge --db data/knowledge.db verify
streamft spiking knowledge --db data/knowledge.db export --out data/knowledge_inbox
streamft assistant index --db data/knowledge.db
```

For facts changing over time, this is safer than gradient updates; weights are
for style/skills, retrieval is for facts.

## Caveats

- Qwen2.5-1.5B-Instruct is a small model. It is much more usable than the 50M
  TinyStories model, but it is not a frontier assistant.
- RAG does not make answers automatically true. The model can still ignore or
  misread context; the `sources` field is the audit trail.
- If no verified context is found, the model may answer from parametric memory.
  Treat uncited factual claims as unverified.
- The 1.5B model runs on the 6 GB card; concurrent generation is serialized.
- Local deploy is still last. The server is a LAN/demo server; TLS and rate
  limits belong at a reverse proxy.
- The spiking line remains research: see `docs/SPIKING_50M.md`.
