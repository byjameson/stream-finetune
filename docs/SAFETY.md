# Safety filter and red-team

Atom Zero has an **opt-in** safety layer for shared or hosted use. It is off by
default so local research behavior does not change. Version 0.3 combines the
deterministic rules (v0.2) with a small **local ML head**; the rules can only
block, and the ML layer only adds blocks. It is a narrow guardrail, not a
complete moderation system.

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

`GET /health` reports `safety: {"enabled": true, "version": "0.3-ml", "ml": true}`
when the head and MiniLM model load; otherwise it is the exact v0.2 rule filter.
`--no-safety-ml` forces rules only for a reproducible rule baseline.

## What it does

1. Checks the user request **before** generation; an explicit harmful request
   gets a refusal and never reaches the model.
2. Runs the deterministic rules first; if they match, the result is final.
3. If rules do not match, a local linear ML head over MiniLM embeddings may
   block high-confidence paraphrases. A missing/broken model falls back to the
   exact rule path.
4. Filters generated or document-grounded text before it is returned. Output
   ML uses a stricter threshold because assistant replies are short and
   structurally different from requests.
5. Adds a `safety` block with the category and a refusal message.

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

The v0.2 rule layer adds a language-agnostic **lexical-semantic layer**:
instruction markers plus action and target co-occurrence. It covers Turkish,
German, Spanish, French, Italian, Portuguese, Dutch and Russian variants (for
example "bomba nasıl yapılır", "wie baue ich eine bombe", "cómo hacer una
bomba", "comment fabriquer une bombe", "come costruire una bomba", "hoe maak ik
een bom", "как сделать бомбу").

The v0.3 ML layer is intentionally small and local:

- frozen `sentence-transformers/all-MiniLM-L6-v2` embeddings (already used by
  the neural reranker);
- a 384-dimensional linear harm gate plus an 8-way category head, stored in
  `streamft/moderation_head.json` (about 58 KB);
- trained on 349 generated examples (`research/moderation_data.jsonl`), with
  the authored red-team and semantic holdout excluded;
- input threshold 0.495, output threshold 0.90;
- no network call, no hosted moderation API, no extra model download;
- if the head or embedding model is missing, `build_safety_filter()` returns
  the exact `SafetyFilter` rules.

The training/evaluation scripts are `research/build_moderation_data.py` and
`research/train_moderation.py`.

## Red-team result (2026-11-21)

```bash
python research/redteam_safety.py \
  --out research/results/redteam_safety_filter.json

python research/redteam_safety.py --ml \
  --model checkpoints/transformer_behavior_50m_scaled \
  --adapter checkpoints/adapters_50m/v0001 \
  --out research/results/redteam_safety_ml_agent.json

python research/redteam_moderation.py \
  --out research/results/redteam_moderation.json
```

```text
rules only:    filter 72/72, output 4/4, agent 72/72
ML hybrid:     filter 72/72, output 4/4, agent 72/72
semantic set:  rules 1/10 blocked, ML-only 10/10, hybrid 10/10
benign set:    0/26 false blocks with the hybrid filter
```

The 72 filter cases are 46 explicit harmful requests (all categories, including
translated variants and paraphrases) and 26 benign prevention/topic controls.
The agent run checks that harmful prompts are refused before generation and
benign controls are not marked as safety blocks. The separate semantic set has
10 held-out paraphrases that the rules intentionally do not encode; the ML head
blocks all 10, including their expected categories. ML-only on the authored
72-case set is 61.1% because many cases are multilingual or category-specific;
the hybrid keeps 100% because rules remain authoritative.

The product benchmark with the filter enabled:

```bash
python research/benchmark_assistant.py \
  --model checkpoints/transformer_behavior_50m_scaled \
  --safety-filter \
  --out research/results/assistant_benchmark_scaled_40k_safety.json
```

```text
26/26 (100%)  # 21 product checks + 5 safety checks
safety_version 0.3-ml, safety_ml true
```

Benchmark safety checks cover violence, self-harm, sexual-minors refusal and
two benign controls (child-abuse reporting and Wi-Fi security). The same 26/26
holds with the neural reranker and with the ML filter on both paths.

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

- The ML layer is a linear head over frozen MiniLM embeddings, trained on a
  small synthetic set; it is not a full moderation model. It can miss novel
  phrasing and can produce false positives.
- The category head is accurate on the held-out semantic set (10/10), but
  category attribution is not guaranteed out of distribution.
- Rules remain the exact fallback and cover the multilingual cases; the ML
  head is English-centric because MiniLM is.
- It is not a substitute for human review, authentication, rate limits,
  logging policy or incident response.
- The filter is **off by default** and is an operator tool, not a content
  policy or moral position. Hosts decide whether to enable it.
- Do not treat the 72/72 or 10/10 scores as proof of general safety.
- A host that enables it should also add legal review, prompt-injection tests,
  PII deletion and a larger human-labelled red-team.
