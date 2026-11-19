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
- the model is only called for open-ended text, with a short memory context;
- no model loaded is a valid mode (tests cover it).

This keeps facts and actions out of the 21M weights. The runtime is tested
independently of the model, so a weak checkpoint does not make memory or task
commands fail.

## Quality expectation

- **21M / 24.6M tokens: BLOCKED.** It learns token statistics and produces
  word salad, not chat.
- 50M / 300M tokens: unknown; the next controlled test. Use a cleaner
  dialogue-heavy mix rather than the broad SmolTalk/UltraChat mixture.
- 100M+ / 1B+ tokens: more fluent, but days-to-weeks on a 6 GB GPU.

The virtual-human layer (memory, tasks, tools) must carry facts and actions;
the language model only supplies wording. Do not equate a small model's broken
output with a disability; it is a capacity/data limitation.
