# From-scratch chat model (English-first)

The active direction after the real-corpus story-lab block: train a small
spiking language model **from scratch** on chat data. No Qwen, no pretrained
language model, no external chat weights.

The product language is **English-first** for global distribution. The Turkish
corpus and tokenizer are kept as a separate track; Turkish can become a later
adapter/checkpoint, but at 21M a bilingual model would weaken both languages.

## English corpus

`research/make_chat_corpus_en.py` builds the shard.

| source | role | licence |
|---|---|---|
| `HuggingFaceTB/smoltalk` (17,160 docs) | general English chat | Apache-2.0 (check per subset) |
| `HuggingFaceH4/ultrachat_200k` (5,598 docs) | multi-turn English chat | MIT |
| `OpenAssistant/oasst2` English pairs (5,645 docs) | human chat | Apache-2.0 |
| `databricks/databricks-dolly-15k` (4,398 docs) | instruction/response | CC BY-SA 3.0 |
| `wikimedia/wikipedia` 20231101.en (7,441 docs) | language/knowledge | CC BY-SA |
| `roneneldan/TinyStories` (9,500 docs) | narrative grammar | CDLA-Sharing |

Format:

```text
<|user|> ... <|end|> <|assistant|> ... <|end|>
```

Plain Wikipedia/story text ends with `<|end|>`. Tokenizer: fresh 8k byte-level
BPE trained on the mixed corpus, with the three special tokens.

Current shard:

| split | tokens |
|---|---:|
| train | 23,632,778 |
| chat valid | 384,875 |
| mixed valid | 485,549 |
| Wikipedia valid | 59,536 |
| TinyStories valid | 41,138 |

Data and checkpoints are local artifacts and are not committed.

## Turkish corpus (retained)

`research/make_chat_corpus.py` built a 27.55M-token Turkish-first shard from
Turkish instructions, daily dialogues, Turkish Wikipedia and TinyStories. It is
kept for a later language pack. The Turkish 500-step smoke reached chat bpb
2.688 but only produced fragments, as expected for a 21M model at 1M tokens.

## Model and training

- 21M spiking LM: dim 1024, 2 layers, LIF + adaptation, byte-level vocab 8192.
- batch 16, seq 128, LR 5e-4, cosine schedule, warmup, AMP.
- 12k steps (~24.6M tokens, ~2 h on the RTX 3050).
- `--save-every 2000` so the run can be interrupted and resumed.
- No replay/EWC here: this is from-scratch pretraining, not continual update.

### Fragmented training

Checkpoint/resume works across sessions:

- `optimizer.pt` stores the AdamW state, so a resume does not reset the
  moments;
- `training.json` merges the previous history instead of overwriting it;
- use a fixed `--total-steps` so the cosine schedule stays consistent, and
  `--save-every N` to checkpoint during a session.

Verified with a 2-step + 2-step resume: final step 4, history length 4, Adam
state step 4. A long run can therefore be split into 2-4 hour sessions; the
machine does not need to stay on for days. The current 21M/24.6M-token run is
only ~2 hours, so it does not need fragmentation.

### 21M final result (BLOCKED)

The run was paused at step 2000 and resumed to **step 12000** with the same
cosine schedule and optimizer state. Final fixed-window evaluation:

| shard | bpb |
|---|---:|
| chat valid | **2.377** |
| Wikipedia | 2.811 |
| TinyStories | 2.115 |

- chat bpb passes the 2.5 smoke threshold (2.539 -> 2.377);
- generation gate **fails**: minimum alpha **0.331**, and **3/12** samples leak
  `<|assistant|>`;
- samples are word salad with math/code fragments, not answers.

Examples:

```text
the_) = an_n of and number function the. the, and a number we of are the,
in function the( the the( of( +11. 2,n_2(( 2)2)) 1, the, 21, 5 61(,, 6 30
```

```text
a aing of and with I of, to the in, with a or of, the: your, your a is-
is. are your's a the that your- is", that the your. to have be to you you
```

Verdict: **BLOCKED**. The 21M model learns token statistics but not turn-level
chat. The broad corpus (especially math/code-heavy SmolTalk) makes it imitate
fragments of prompts and equations.

A decoding sweep at temperature 0.3/0.5/0.7 and top-k 5/10/20 does not fix it:
all settings collapse into function-word loops (`a of the of a ...`). The
problem is the model/data, not the sampler.

Evidence: `research/results/spiking_chat_en_21m.json` and
`research/results/spiking_chat_en_sampling.json`.

### Clean-data control (BLOCKED)

The clean run changes **only the data mix**, not the tokenizer, model size or
schedule:

- `research/make_chat_corpus_clean.py` filters out code/math markers and
  low-alpha messages, then mixes SmolTalk-curated, everyday conversations,
  filtered UltraChat, OASST2, Dolly, synthetic memory/task dialogues and
  TinyStories;
- 26.98M training tokens, 497k chat-valid tokens;
- same 21M, 8k tokenizer, 12k steps, batch 16, seq 128, LR 5e-4.

Final fixed-window evaluation on the clean validation shard:

| shard | bpb |
|---|---:|
| clean chat valid | **2.199** |
| Wikipedia (general) | 2.925 |
| clean TinyStories | 2.187 |

- minimum sample alpha is **0.608** (all samples pass the 0.5 alpha floor);
- **5/12** samples still leak `<|assistant|>`;
- samples are still word salad, just less math/code-heavy;
- verdict: **BLOCKED**.

Data cleaning improved the surface metrics but did not produce turn-level chat.
The remaining bottleneck is token/parameter scale or the architecture, not the
math/code filter. Evidence: `research/results/spiking_chat_en_clean_21m.json`.

### 50M capacity probe (BLOCKED)

The next run changes only the model size: **50.07M** parameters (`dim=2040`,
2 layers), same clean corpus, tokenizer, 12k steps, batch 16, seq 128.

| shard | 21M clean | 50M clean |
|---|---:|---:|
| clean chat bpb | 2.199 | **2.192** |
| Wikipedia bpb | 2.925 | 2.927 |
| clean TinyStories bpb | 2.187 | 2.197 |
| alpha min | 0.608 | **0.627** |
| tag leaks | 5/12 | **1/12** |

The 50M model is marginally better on chat bpb (+0.3%) and tag leaks, but the
samples are still word salad. Verdict: **BLOCKED**. Scaling parameters alone
does not produce turn-level chat at this token budget.
Evidence: `research/results/spiking_chat_en_clean_50m.json`.

### Why 128-token windows are the wrong unit

`research/analyze_chat_lengths.py` measures the conversations in the packed
tensor:

- median conversation: **355 tokens**;
- only **12.6%** of conversations fit in 128 tokens, covering **1.7%** of
  training tokens;
- 41.6% fit in 256 tokens; 59.5% fit in 512; 82.5% fit in 1,024.

A fixed 128-token window therefore almost never contains a complete
user -> assistant exchange. The model learns local token statistics, not turn
structure. This explains both the word salad and the `<|assistant|>` leaks.

The first fix is cheaper than a 512-token run: `--start-token 0` makes every
training window start at a `<|user|>` token, so each window begins with a real
turn instead of a random mid-conversation position.

Turn-aligned 21M result: chat bpb **2.208**, but **9/12** tag leaks, **2 empty**
samples and minimum alpha **0.0**. Verdict: **BLOCKED**; alignment alone is not
enough.

The next run keeps start-token alignment and adds **SFT-style user-label
masking** (`--mask-user`): loss is computed only on assistant turns, so the
model spends its capacity learning responses instead of predicting the user
prompt. Same 21M, 12k steps, batch 16, seq 128.

**SFT result (first qualitative improvement, still BLOCKED):**

- assistant-only loss **6.80**, ppl 899 (still high);
- all-token chat bpb 2.311 (not comparable: user labels are masked);
- **0 empty** samples, **2/12** tag leaks, minimum alpha **0.492**;
- samples now contain story structure, names and dialogue
  (`"... named ... girl ... friends ... said Lily ..."`), not just
  function-word loops.

The strict gate still fails (one sample just under the 0.5 alpha floor, two
tag leaks). This is the first run that looks like it learned a response shape.

A sampling sweep at temperature 0.3/0.5/0.7 still collapses into function-word
loops; the 0.8/top-k 40 regime used by `chat_eval` is the better one for this
checkpoint.

**50M SFT result (no improvement):**

- assistant loss 6.73 (vs 6.80 at 21M), chat bpb 2.293;
- **2 empty** samples, **6/12** tag leaks, alpha min **0.0**;
- samples are less story-shaped than the 21M SFT run.

Parameter scaling does not improve the response shape. The 21M SFT checkpoint
remains the best qualitative result. Next: continue the 21M SFT optimization
with more assistant-heavy tokens and a constant low LR, then revisit context
length or architecture.

Evidence: `research/results/spiking_chat_en_clean_21m_sft.json`,
`research/results/spiking_chat_en_sft_sampling.json`,
`research/results/spiking_chat_en_clean_50m_sft.json`.

### Transformer result: architecture is the bottleneck

The spiking runs could not separate architecture from data. A **20.2M
transformer** (`dim=512`, 5 layers, 8 heads) was trained on the exact same clean
corpus, tokenizer, start-token sampler and assistant-only loss.

- 12k steps in **~25 minutes** at 0.09-0.11 s/step;
- chat bpb **1.359**, assistant loss **4.01** (ppl 55);
- alpha min **0.731**, 0 empty samples;
- with role tokens banned during generation: **0 tag leaks** and
  **PASS_CHAT_SMOKE**;
- without banning, 5/12 samples leak `<|assistant|>` and the gate is BLOCKED.

| model | chat bpb | assistant loss | alpha min | tag leak | verdict |
|---|---:|---:|---:|---:|---|
| spiking 21M SFT | 2.311 | 6.80 | 0.492 | 2/12 | BLOCKED |
| spiking 50M SFT | 2.293 | 6.73 | 0.0 | 6/12 | BLOCKED |
| transformer 20.2M | **1.359** | **4.01** | **0.731** | 0/12* | **PASS** |

\* with `<|user|>`/`<|assistant|>` banned at generation time.

Sample outputs:

```text
hello -> Hello! How can I help you today?
what is the weather today? -> To determine what the weather is.
add task: buy milk -> Added task: go for a walk.
tell me a short story -> Analyze the context of your story. It takes for a
restless, unacceptable sunny day to be filled with dancier. The story includes
a mysterious, a man who was hired by a grandmother ...
```

This is a **smoke-gate pass, not a general assistant**. Responses are short,
sometimes wrong (the task text is wrong) and repetitive; but they are
grammatical English and turn-shaped. The spiking architecture is the
bottleneck at this data scale.

Implication: the **product line should use the transformer**; the spiking line
stays as a separate research track. Next: 50M/100M transformer, memory/task
curriculum, and the hybrid runtime.

Evidence: `research/results/transformer_chat_en_clean_21m.json` and
`research/results/transformer_chat_en_clean_21m_banned.json`.

### 50M transformer: passes, but not worth it

A 49.2M transformer (`dim=768`, 6 layers, 12 heads) was trained on the same
data with the same SFT objective.

| model | chat bpb | assistant loss | alpha min | tag leak | verdict |
|---|---:|---:|---:|---:|---|
| transformer 20.2M | 1.359 | 4.01 | **0.731** | 0/12* | PASS |
| transformer 49.2M | **1.344** | **3.97** | 0.573 | 0/12* | PASS |

\* role tokens banned at decoding.

The 50M model is only 0.015 bpb better, has a lower alpha floor, and its samples
are more rambling/math-like. At 2.4x the step time it is not a better product
checkpoint. **Decision: the 20M transformer is the product model**; further
work goes into data/skills and the runtime, not parameter scale.

Evidence: `research/results/transformer_chat_en_clean_50m.json` and
`research/results/transformer_chat_en_clean_50m_banned.json`.

### Skill fine-tune and virtual-human smoke

The 20M transformer was fine-tuned on a targeted skill corpus for 6,000 steps
(LR 1e-4, 1:1 skill + clean-chat replay):

- skill mix assistant loss **2.27** (ppl ~9.6) versus **4.01** for the base;
- end-to-end `chat_human` smoke: **PASS_VIRTUAL_HUMAN_SMOKE**;
- all memory/task checks pass:
  name, likes, add/list/complete tasks, list-after-complete, remember facts;
- greeting and small talk now come from the deterministic fast path; the model
  handles stories/open-ended text:

```text
you> hello
tiny> Hello! How can I help you today?
you> how are you?
tiny> I am doing well, thank you. How are you?
you> tell me a short story
tiny> Nora was a curious rabbit who lived near the garden...
```

**Tradeoff:** general clean-chat quality regressed. Clean valid chat bpb is
**1.655** and assistant loss **4.93** (base: 1.359 / 4.01), with one tag leak
unconstrained. This is forgetting from the aggressive skill fine-tune; the
skill checkpoint is the virtual-human product, not the general-chat model.

Evidence: `research/results/transformer_chat_skill_21m.json` and
`research/results/chat_human_eval.json`.

### Retention-friendly fine-tune (adopted)

The aggressive 1:1 skill run regressed clean chat (1.359 -> 1.655 bpb). The
retention run uses **2:1 clean:skill replay**, LR **3e-5** and 4,000 steps.

| checkpoint | clean chat bpb | clean assistant loss | skill-only assistant loss | E2E |
|---|---:|---:|---:|---|
| base 20M | **1.359** | **4.01** | - | - |
| aggressive skill | 1.655 | 4.93 | ~2.27 | PASS |
| **retention skill** | **1.440** | **4.27** | **0.31** (ppl 1.36)* | **PASS** |

\* synthetic skill validation is near memorization; it measures skill
reproduction, not general skill transfer.

The retention checkpoint cuts most of the forgetting while keeping all
memory/task/greeting/story checks. It is the **virtual-human product
checkpoint** (`transformer_chat_skill_ret_21m`).

### Teach/learn loop

- `teach: <fact>` in `research/chat_human.py` writes the fact to memory
  immediately, no weight update;
- `research/update_loop.py add/list/export` keeps a reviewed queue of skill
  examples;
- queued examples are exported as chat documents and only enter a **gated**
  replay fine-tune after review;
- no autonomous weight update and no public release.

**Gated update result (accepted):** with 267 novel, learnable format/style
examples (213 train / 54 validation), 2:1 clean replay and 300 steps:

- queued-skill loss 4.346 -> **1.114**, a **74.4%** target gain;
- clean bpb 1.440 -> **1.475** (+2.45%);
- clean assistant loss 4.266 -> **4.374** (+2.52%);
- all three gate checks pass; accepted as `checkpoints/updates/v0001`.

The accepted checkpoint also passes the standard E2E memory/task/greeting/story
test. However, **loss is not behavior**: generation of the new skills is mixed
(`summarize` works; `count to five`, `be brief`, `make it polite` often fall
back to generic replies). The gate is a loss/regression gate, not a full
behavioral gate.

A later format queue (283 examples) cleared loss + behavior but produced empty
clean-chat samples. The gate was tightened with a 12-sample clean generation
smoke; the v0002/v0003 candidates were then **rejected**, and **v0001 remains
the product checkpoint**. Exact format skills are handled by the deterministic
fast path in `chat_human.py`; the model handles stories/open-ended text.

Earlier 55-example runs were rejected at 2.07% target gain; the gate correctly
refused them.

### 50M behavior comparison

The 49.2M transformer was fine-tuned for 1,000 steps on a diverse mix of clean
chat + skill + format/novel queues.

| model | clean bpb | clean assistant | raw clean smoke | runtime behavior | E2E |
|---|---:|---:|---|---|---|
| v0001 20M | 1.475 | 4.374 | PASS | PASS | PASS |
| **50M behavior long** | **1.375** | **4.066** | **PASS** | PASS | PASS |

The long 50M behavior model clears all gates and has better clean metrics and
comparable stories. It is now the **product checkpoint**
(`transformer_behavior_50m_long`); v0001 remains the compact fallback. Live
factual questions (weather, time) are out of scope and must be routed to a tool
or a refusal.

### LoRA adapter

`streamft/lora.py` implements small MLP LoRA adapters for both HF-style models
and the tiny transformer. The gated 50M adapter:

- **368,640** trainable parameters (~0.74% of the base), ~1.5 MB;
- 100 steps, ~15 s on the RTX 3050;
- queue-target gain **+38.2%**;
- clean bpb regression **+0.08%**, assistant **+0.07%**;
- clean smoke **PASS** (1.376 bpb / 4.069 assistant);
- runtime behavior **PASS**, E2E **PASS**.

The adapter is a delivery mechanism, not a new base model: users can train
per-domain adapters while sharing the frozen 50M base. Knowledge still belongs
in `streamft.doc_memory`.

Evidence: `research/results/adapter_behavior_banned.json`,
`adapter_behavior_eval.json`, `adapter_chat_human_eval.json`.

Public release gates: [`../docs/PUBLIC_RELEASE_CHECKLIST.md`](../docs/PUBLIC_RELEASE_CHECKLIST.md).

## Evaluation

`research/chat_eval.py --lang en` reports:

- fixed-window bpb on chat, Wikipedia, TinyStories and mixed validation;
- six English prompts, two samples each;
- alpha ratio, 3-gram repetition and special-token leakage.

The gate is **not** "it chats like Qwen". The smoke gate is: finite chat bpb,
non-empty English samples, low tag leakage, and visible turn structure. A
general assistant claim would require much more data, a larger model and a
separate instruction-following evaluation.

## Virtual-human runtime

`research/chat_human.py` is the CPU-side layer around the language model:

- JSON memory for persona, facts, episodes and tasks;
- deterministic commands: `my name is ...`, `add task ...`, `list tasks`,
  `complete task N`, `what do you remember`;
- the model is only called for open-ended text; the transformer prompt is a
  plain chat turn because the 20M model was not trained on a long persona block;
- supports both `--arch spiking` and `--arch transformer`;
- no model loaded is a valid mode (tests cover it).

Verified with the 20M transformer:

```text
you> hello
tiny> Hello! How can I help you today?
```

This keeps facts and actions out of the weights. The runtime is tested
independently of the model, so a weak checkpoint does not make memory or task
commands fail.

### Local HTTP server

```bash
python -m streamft.chat_server \
  --model checkpoints/transformer_chat_en_clean_21m \
  --host 127.0.0.1 --port 8013
```

Verified: `/health` 200 (20,219,392 params, step 12000), `/generate` 200 and
`/v1/chat/completions` 200. One-command demo:
`scripts/chat_demo.ps1`. Model card:
[`../docs/TINY_TRANSFORMER_CHAT.md`](../docs/TINY_TRANSFORMER_CHAT.md).

## Quality expectation

- **20.2M transformer: PASS_CHAT_SMOKE** with role tokens banned; short,
  grammatical, often generic/wrong responses. Not a general assistant.
- **49.2M transformer: PASS** but only 0.015 bpb better, lower alpha floor and
  more rambling; not the product choice.
- **spiking 21M/50M: BLOCKED** at 2.29-2.31 chat bpb / 6.7-6.8 assistant loss.
- Next: targeted skill/curriculum data and runtime evaluation, not more
  parameter scale.

The virtual-human layer (memory, tasks, tools) must carry facts and actions;
the language model only supplies wording. Do not equate a small model's broken
output with a disability; it is a capacity/data limitation.
