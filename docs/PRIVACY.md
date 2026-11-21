# Privacy and deletion

Atom Zero is local-first. Documents and session memory live in local files. This
page documents the tested deletion paths.

## Delete one indexed source

HTTP:

```bash
curl -X POST http://127.0.0.1:8013/v1/documents/forget \
  -H 'Content-Type: application/json' \
  -d '{"source":"my-note.txt"}'
```

CLI:

```bash
python -m streamft.doc_memory --db data/docs.db forget my-note.txt
```

The response reports `chunks_deleted`, `meta_deleted` and `vacuumed`. The row is
removed from the FTS5 index, `doc_meta` and the derived chunk-vector cache
(`doc_vectors`), then SQLite is compacted with `VACUUM` when possible. The
in-memory prewarmed index is invalidated at the same time and rebuilt on the
next search from remaining vectors only. A missing source returns 404 / a
non-zero CLI exit.
Local-path sources are matched across `/` and `\` separators, so
`examples/demo_note.txt` deletes a row stored as `examples\demo_note.txt`.

The browser has a **Delete data** section with the same operation.

Safety moderation runs locally as well: the rule filter and the small ML head
run in-process. Prompts are not sent to a moderation API or a hosted service.

The optional chat provider is different: when `--chat-provider-url` is set,
free-form prompts (and only those; not RAG retrieval) are sent to that endpoint.
A local Ollama endpoint keeps them on the machine; a remote URL leaves it.
Leave the provider unset for fully local inference.

## Delete session memory

The agent session stores facts, tasks and recent episodes in
`data/sessions/<id>.json`.

Delete matching text:

```bash
curl -X POST http://127.0.0.1:8013/v1/memory/forget \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"web-abc123","text":"alice@example.test"}'
```

Delete the whole session memory (persona stays, facts/tasks/episodes are
cleared):

```bash
curl -X POST http://127.0.0.1:8013/v1/memory/forget \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"web-abc123","all":true}'
```

The browser has **Forget session memory** for the current session.

## What is tested

`tests/test_privacy.py` verifies:

- exact-source deletion removes only that source;
- other documents remain searchable;
- deleting a missing source raises `SourceNotFound`;
- teacher-note metadata is removed;
- matching facts, episodes and tasks are deleted;
- full session clear removes facts, episodes and tasks;
- the agent then answers "I do not know your name" instead of the deleted fact.

`tests/test_chat_server_docs.py` verifies the HTTP document and session delete
paths, including the 404 for a missing source. `tests/test_doc_memory.py`
verifies that replacing a source prunes its old chunk vectors.

## Limits

- This removes local SQLite/JSON data. It does **not** erase filesystem
  snapshots, backups, OS caches, or copies made by other tools.
- `VACUUM` compacts the SQLite file, but secure physical erasure is a
  filesystem/SSD concern.
- Hosted deployments must define their own backups, logs, retention and
  deletion guarantees. This repository does not operate a hosted service.
- PII detection/redaction is a policy decision; deletion is explicit
  source/text-based, not automatic detection.
