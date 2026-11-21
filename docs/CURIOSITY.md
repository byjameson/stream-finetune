# Curiosity queue

The assistant can notice questions it could not answer and propose them for
research, but it never browses, calls a teacher or changes weights on its own.
A human approves every topic.

## Flow

```text
user question
  -> no local source
  -> knowledge gap recorded (pending)
  -> "Propose curiosity topics" or POST /v1/curiosity/propose
  -> pending research item, kind=curiosity
  -> human approves in the research queue
  -> worker fetches Wikipedia context
  -> local Ollama teacher synthesizes an answer
  -> unverified teacher note stored in RAG
  -> the next answer cites the note
```

No network call or teacher call happens before approval. The model's weights do
not change at any point in this flow.

## API

```bash
# see gaps and curiosity items
curl http://127.0.0.1:8013/v1/curiosity

# turn up to three pending gaps into pending research topics
curl -X POST http://127.0.0.1:8013/v1/curiosity/propose \
  -H 'Content-Type: application/json' \
  -d '{"limit":3,"pages":1}'

# review and approve
curl http://127.0.0.1:8013/v1/research/queue
curl -X POST http://127.0.0.1:8013/v1/research/queue/approve \
  -H 'Content-Type: application/json' -d '{"id":"r0001"}'

# run one approved item (Wikipedia + configured teacher)
curl -X POST http://127.0.0.1:8013/v1/research/queue/run \
  -H 'Content-Type: application/json' -d '{"limit":1}'
```

The browser has **Propose curiosity topics** next to the knowledge-gap list.
Items created this way are labelled `curiosity` in the research queue.

## Live evidence

`python research/curiosity_demo.py` produced:

```text
before:          I do not have a source for that yet...
unapproved_run:  0 item(s)
approved_run:    done
after:           ... Wellington ... (teacher note, unverified: Wikipedia)
weights_changed: false
```

Evidence: [`curiosity_live.json`](../research/results/curiosity_live.json).
The deterministic unit/API tests are `tests/test_curiosity.py` and
`tests/test_curiosity_api.py`.

## When do the weights change?

Only at the reviewed adapter step:

1. A human approves or corrects chat feedback.
2. `research/learning_loop.py` exports approved examples.
3. A LoRA adapter is trained; the gate checks target gain, clean bpb/assistant
   regression, behavior and smoke.
4. An accepted `vNNNN` is activated with `learning_loop.py activate vNNNN`.
5. `chat_server --adapter auto` loads it.

Curiosity and teacher notes are **RAG data**, not training data. They are
unverified until a reviewer marks them verified. A future bridge from verified
notes to reviewed feedback would still have to pass the same gate; it is not
automatic today.
