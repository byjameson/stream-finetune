# Safety filter and red-team

Atom Zero has an **opt-in** safety layer for shared or hosted use. It is off by
default so local research behavior does not change. It is a narrow guardrail,
not a complete moderation system.

## Enable it

```bash
python -m streamft.chat_server \
  --model checkpoints/transformer_behavior_50m_long \
  --adapter auto --docs data/docs.db \
  --safety-filter
```

The demo scripts accept `-SafetyFilter`:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/demo_all.ps1 -SafetyFilter
```

`GET /health` reports `safety: {"enabled": true, "version": "0.1"}`.

## What it does

1. Checks the user request **before** generation; an explicit harmful request
   gets a refusal and never reaches the model.
2. Filters generated or document-grounded text before it is returned.
3. Adds a `safety` block with the category and a refusal message.

Categories:

- sexual content involving minors
- self-harm instructions
- violence or weapons instructions
- hateful/violent targeting of a group
- harassment, doxxing or blackmail
- illegal instructions
- private-information lookup
- malware or unauthorized access

The filter targets explicit instructions. Benign discussions such as "How do I
report child abuse?", "What are the warning signs of suicide?", "What is
genocide?" and "How does a bomb calorimeter work?" are not blocked.

## Red-team result (2026-11-21)

```bash
python research/redteam_safety.py \
  --out research/results/redteam_safety_filter.json

python research/redteam_safety.py \
  --model checkpoints/transformer_behavior_50m_long \
  --adapter checkpoints/adapters_50m/v0001 \
  --out research/results/redteam_safety_agent.json
```

```text
filter:        30/30
output filter:  4/4
agent refusal: 30/30
```

The 30 filter cases are 18 explicit harmful requests (all categories) and 12
benign prevention/topic controls. The agent run checks that harmful prompts are
refused before generation and benign controls are not marked as safety blocks.

The product benchmark with the filter enabled:

```bash
python research/benchmark_assistant.py \
  --model checkpoints/transformer_behavior_50m_long \
  --safety-filter \
  --out research/results/assistant_benchmark_safety.json
```

```text
26/26 (100%)  # 21 product checks + 5 safety checks
```

Benchmark safety checks cover violence, self-harm, sexual-minors refusal and
two benign controls (child-abuse reporting and Wi-Fi security).

## Untrusted documents and prompt injection

Retrieved document text is treated as **data**, not instructions:

- the agent's document path quotes chunks and cites the database source;
- it does not feed retrieved text into a tool executor;
- citations come from `DocMemory` rows, so text like `Source: https://evil...`
  cannot change the actual source;
- a document cannot add tasks or disable the safety layer.

The deterministic injection suite:

```bash
python research/redteam_injection.py \
  --out research/results/redteam_injection.json
```

```text
injection: 4/4
```

It covers quoted instructions, direct jailbreak, citation spoofing and a
document that tries to disable safety. The red-team is structural and narrow;
it does not prove resistance to every prompt-injection technique.

## Limits

- Pattern-based filtering can miss paraphrases, other languages and novel
  phrasing; it can also produce false positives.
- It is not a substitute for a trained moderation model, human review,
  authentication, rate limits, logging policy or incident response.
- Hosted use should enable the filter **and** add semantic moderation,
  prompt-injection tests, PII deletion and a red-team expansion.
- Do not treat the 30/30 score as proof of general safety.
- Local-only deployments may keep the filter off; shared deployments should not.
