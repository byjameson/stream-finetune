# Knowledge gaps: missing data loop

Atom Zero records questions it cannot answer from local sources. A human can
turn a gap into a **pending** research topic. The scheduled worker fetches it
only after approval, and the answer comes from DocMemory with a source. This is
assisted self-improvement, not autonomous crawling or weight training.

## Flow

```text
question -> no local source
         -> knowledge gap (pending)
         -> propose research topic (pending; no network)
         -> human approves
         -> scheduled worker fetches Wikipedia
         -> DocMemory indexes text + source
         -> next question answered with a citation
```

Facts go to RAG. The gap loop never changes model weights. Style/skill changes
still go through reviewed feedback and the gated LoRA pipeline.

## Browser

The **Knowledge gaps** section shows pending/proposed gaps with:

- **Propose research**: creates a pending research-queue item. It is not
  fetched yet.
- **Dismiss**: removes the gap from the working list.

Approve the resulting topic in the **Research queue** section, then run the
worker. The next answer can use the new source.

## CLI

```bash
python research/gap_loop.py list
python research/gap_loop.py propose g0001 --pages 1
python research/gap_loop.py dismiss g0002
```

`propose` prints the new research id and leaves it `pending`. Approve it:

```bash
python research/research_queue.py --queue data/research_queue.json approve r0001
python research/research_worker.py --queue data/research_queue.json \
  --db data/docs.db --limit 1 --once
```

## HTTP

| endpoint | purpose |
|---|---|
| `GET /v1/learning/gaps` | gaps and pending/proposed/dismissed counts |
| `POST /v1/learning/gaps/propose` | `{"id":"g0001","pages":1}`; creates a pending research item |
| `POST /v1/learning/gaps/dismiss` | `{"id":"g0002"}`; dismisses a gap |

`/health` reports `knowledge_gaps` stats.

## Verified run (2026-11-21)

```text
question: what is the population of Paris?
first reply: I do not have a source for that yet. I can add it to the research queue...
gap: g0001 | pending
proposed: g0001 -> research r0001 | pending
approved: r0001
worker: done, 1 source
second reply: Paris ... estimated city population of 2.04 million ...
source: https://en.wikipedia.org/?curid=22989
```

The second answer is extractive and cited; it is not a generated guess.

## Limits

- Only approved topics are fetched. Nothing is crawled automatically.
- Wikipedia lead sections are a starting point, not a full fact database.
- A gap does not train weights; it adds a source to RAG.
- Exact facts should be verified and, for private data, added as local documents
  instead of public research.
- A teacher note remains unverified until a named reviewer verifies it.
