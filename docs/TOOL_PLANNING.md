# Constrained tool plans

Atom Zero can execute a small JSON plan through `POST /v1/agent/plan`. This is
an **allowlisted executor**, not free-form model planning. The 49.2M core still
routes explicit commands; a caller (UI, test, or a future provider-assisted
planner) submits the steps, and the validator decides whether they are legal.

## Why a validator

- The 49M model is not reliable at open-ended tool selection.
- A plan must not be able to run arbitrary shell commands, fetch an arbitrary
  URL, or call an unknown tool.
- Side effects (tasks, research) need an explicit human confirmation.
- A small, fixed schema is easy to audit and test.

## Allowed tools

| tool | side effect | arguments |
|---|---|---|
| `time` | no | none |
| `calculator` | no | `expression` (max 80 chars) |
| `search_docs` | no | `query` (max 500), optional `k` (1-10) |
| `evidence` | no | `source`, optional `page`, `section` |
| `task_list` | no | none |
| `task_add` | **yes** | `text` (max 500) |
| `task_complete` | **yes** | `id` |
| `research` | **yes** | `query` (max 300) |

Unknown tools, unknown fields, wrong types, control characters and more than
**3 steps** are rejected. There is no `shell`, no arbitrary URL fetch, no code
execution and no delete tool.

## Read-only example

```bash
curl -s http://127.0.0.1:8013/v1/agent/plan \
  -H 'Content-Type: application/json' \
  -d '{
    "session_id": "s1",
    "steps": [
      {"tool": "calculator", "args": {"expression": "2 + 2"}},
      {"tool": "evidence", "args": {"source": "manual.pdf", "page": 4}}
    ]
  }'
```

Read-only plans execute without confirmation.

## Side-effect example

```json
{
  "session_id": "s1",
  "steps": [{"tool": "task_add", "args": {"text": "buy milk"}}]
}
```

The first call returns:

```json
{
  "status": "confirmation_required",
  "confirm": false,
  "plan": [
    {"tool": "task_add", "args": {"text": "buy milk"}, "side_effect": true}
  ],
  "side_effects": [
    {"tool": "task_add", "args": {"text": "buy milk"}, "side_effect": true}
  ],
  "results": []
}
```

Nothing is executed. Send the same plan with `"confirm": true` to execute it.
If a plan mixes read-only and side-effect steps and confirmation is missing,
**nothing** runs; this avoids half-applied plans.

`confirm: true` is an explicit caller assertion, not proof that a human clicked
a dialog. A UI must ask first. The API is a localhost tool, not a hosted
multi-tenant service.

## Failure behaviour

- Invalid plan: HTTP 400 with a validation message.
- Tool execution error: `status: "error"` with completed results and the failing
  step; earlier side effects are not rolled back.
- Optional safety filter: if enabled, text arguments are checked before the
  tool runs. The filter remains off by default.

## What this is not

- It is not model-generated planning. The 49M model does not choose these tools.
- It is not a general agent runtime or a hosted service.
- It is not permission to crawl. `research` is still the approved Wikipedia
  path and requires confirmation.

The next step is provider-assisted planning: a local provider may propose a
JSON plan, but it must pass this same validator and confirmation gate.
