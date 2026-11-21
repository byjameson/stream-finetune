# Changelog

## 0.7.0 — local verified assistant
- `streamft assistant ask|index|serve`: Qwen2.5-1.5B-Instruct plus FTS5
  retrieval over the verified knowledge ledger.
- `GET /health`, `/metrics`, `/v1/models`, `/knowledge/search` and
  `POST /v1/chat/completions`; answers include source citations.
- Verified demo: speed-of-light question answered with 299,792,458 m/s and
  three sources; HTTP endpoint returned 200.
- Real-source build: 5 official Python/NumPy pages + 1 human-approved note;
  false moon claim rejected. FTS chunking added; 10-question held-out eval:
  answer 1.00, citation 1.00, strict 1.00.
- Local deployment on this PC: `http://127.0.0.1:8014`; launcher
  `scripts/start_assistant.ps1`; source builder
  `scripts/ingest_real_sources.ps1`.
- 50M/500M spiking run stopped at 1,500 steps: 1.3256 bpb vs 1.327 for 21M and
  1.336 for 4.3M; no scaling win. The active direction is the spiking research
  line; the Qwen assistant is parked.
- Generative data diversity: 4.3M, 3000 steps, 16M tokens. TinyStories-only:
  TS 1.566 / WikiText 4.455 bpb. 25% WikiText (12M+4M): TS 1.605 / WikiText
  2.927; 50%: 1.702/2.906; 75%: 1.734/2.820. Most of the WikiText gain comes
  from adding a small amount of diverse data; ratio beyond 25% costs more
  TinyStories than it buys. 21M/50M TinyStories models stay at WikiText 4.03.
- 21M 25% mix at 3000 steps: TS 1.910 / WikiText 3.228; undertrained at this
  budget and worse than the 4.3M mix. Scaling needs more steps.
- Longer mixed training: 25% WikiText, 10k steps, 40M tokens gives TS 1.464 /
  WikiText 2.794 bpb, versus 1.302/4.003 for TinyStories-only 10k. The 10k mix
  pays +0.161 TS bpb for -1.209 WikiText.
- 21M 10k mix: batch-16 run TS 1.774 / WikiText 3.108 (20.5M tokens); batch-32
  matched-token run TS 1.669 / WikiText 3.011 / world facts 3.384. Both are
  worse than the 4.3M mix (1.464/2.794/3.249): 5x parameters do not buy quality
  at this 41M-token budget.
- Held-out world-facts: 4.3M mix 10k reaches 3.249 bpb vs 3.949 for
  TinyStories-only 10k; 21M mix 10k is 3.465 (under-budget). The data-diversity
  gain transfers beyond WikiText-2, though the articles are also Wikipedia.
- TS-first curriculum (5k TS + 5k mix) does not dominate: TS 1.439, WikiText
  2.964, world facts 3.374 vs pure mix 1.464/2.794/3.249. It only shifts the
  tradeoff toward stories; pure mix is better for general text.
- Generation samples: all models remain locally repetitive/ungrammatical.
  The 10k mix has adjacent-repeat 0.065, distinct-2 0.967; 21M/50M TS-only are
  more story-shaped but distinct-2 0.928, still repetitive.
- Sampling sweep: tightening temperature/top-k helps the 4.3M TS-only model
  (adjacent 0.052 -> 0.034, distinct-2 0.971 -> 0.983) but hurts the mix model
  (0.065 -> 0.107, 0.968 -> 0.909). No sampling setting rescues the mix.
  No clean generation winner and no deployment win.
- Tiny Spiking Lab domain ingest: 4.3M mix base, 250/76 world-facts split,
  300 steps, 20% replay. Target bpb 5.99 -> 4.45; ingest gate accepted on a
  4096-token eval (old +2.55%), but full 8192 eval shows old +3.29% and
  unmonitored WikiText +7.1%. Generation is formatting noise/corrupted.
  Release gate BLOCKED; no deployable artifact.
- Clean-WikiText mix: removing `=`/newline tokens gives TS 1.435, WikiText
  2.973, world facts 3.214 vs normal mix 1.464/2.794/3.249; it improves TS and
  world facts but loses WikiText and does not fix domain-prompt generation.
  Not a clear win.
- Space-story ingest demo: 21M TS base, 200 synthetic Zeta stories, 300 steps,
  20% replay + rollback. Target 2.19 -> 1.66 bpb and robot-themed samples
  (alpha ~0.7); full eval TinyStories +13.6%, WikiText +33.9%. Release still
  blocked by retention.
- Safe story-lab recipe: 70% replay from 20M TinyStories + 5M WikiText, LR 5e-5,
  120 steps. Target 2.19 -> 0.72 bpb, TinyStories +4.1%, WikiText -13.4%,
  generation alpha 0.68-0.76. Passes the narrow story-lab gate.
- Second story corpus (Nova) replicates the safe recipe: target 2.05 -> 0.60,
  TinyStories +4.4%, WikiText -13.6%, alpha 0.73-0.77.
- Second seed per corpus replicates: target -67.1/-67.4% (space) and
  -70.7/-71.2% (nova); TinyStories +4.11-4.36%; WikiText -13.25-13.62%.
  Two corpora x two seeds PASS_NARROW_STORY_LAB. Still synthetic/
  near-memorization and not a general model; packaging/licence/model card
  remain.
- Efficiency probe: 4.3M 1,171 tok/s CUDA / 730 tok/s CPU, 24.8 MB peak VRAM;
  21M 563/204 tok/s, 88.3 MB. Single prompt, no batching, not a production
  benchmark.
- Product packaging: model card `docs/TINY_STORY_LAB.md` and one-command demo
  `scripts/story_lab_demo.ps1`; licence/provenance review still required.
- Local HTTP deployment of the safe story model verified: `/health` 200,
  `/generate` 200, alpha 0.708. Server defaults tightened to temp 0.5/top-k 10/
  repetition penalty 1.1; still repetitive and local-only.
- Real public-domain cross-book check (Alice train -> Oz valid, two seeds):
  target 2.733 -> 2.414/2.411 (-11.7/-11.8%), TinyStories +3.5%, WikiText
  -14.0/-14.1%, but generation alpha collapses to 0.534 (seed 0) and
  0.259/0.394 (seed 1) on `Dorothy opened the door`; base is 0.727-0.751.
  Gate BLOCKED: synthetic story-lab pass does not generalize to real text.
  Evidence: `research/results/spiking_realstory_seedcheck.json`.
- From-scratch chat pipeline: 27.55M-token Turkish-first corpus built first
  (Turkish instructions, daily dialogues, Wikipedia, TinyStories); 500-step
  21M smoke: chat 2.688, Wikipedia 2.918, TinyStories 2.820 bpb, fragments
  only. Switched the active track to **English-first** for global release:
  23.63M-token corpus from SmolTalk, UltraChat, OASST2, Dolly, Wikipedia and
  TinyStories, fresh 8k BPE tokenizer. Turkish corpus retained for a later
  language pack; 21M English run follows.
- Spiking trainer saves/loads AdamW state (`optimizer.pt`) and merges
  `training.json` on resume; 2+2-step fragment verified at step 4 with Adam
  step 4. Long runs can be split into sessions.
- Virtual-human runtime scaffold: JSON persona/facts/episodes/tasks with
  deterministic memory and task commands; model only handles open-ended text.
- Crypto paper-trading baseline (no real money): BTCUSDT daily walk-forward,
  0.10% fee + 0.05% slippage. Ridge 8.80x vs buy-hold 8.50x but worse Sharpe;
  HGB 1.66x and logistic 1.56x. Only SMA-50 trend (36.0x, Sharpe 1.26) is a
  CHECK result, not an ML edge. One asset/one history; parameter sweep next.
- Crypto robustness: BTC+ETH SMA-20/50/100/200 sweep. SMA-50 beats buy-and-hold
  on both (30.5x/35.9x vs 10.8x/9.4x), but SMA-20 loses on BTC and SMA-100/200
  lose on ETH. Verdict TREND_CANDIDATE; not a deployable strategy. No real
  money or API keys.
- English-first 21M chat, 12k steps: chat bpb 2.377, Wikipedia 2.811,
  TinyStories 2.115. Generation **BLOCKED**: word salad, alpha min 0.331,
  3/12 samples leak `<|assistant|>`. bpb alone is not chat. A temp 0.3-0.7 /
  top-k 5-20 sweep still produces function-word loops, so decoding is not the
  fix. Next: cleaner dialogue-heavy mix and/or 50M.
- Clean-data control: filtered dialogue corpus (26.98M tokens) with code/math
  removed and synthetic memory/task examples; same 21M/tokenizer/schedule.
  Result: clean chat bpb 2.199, alpha min 0.608, but 5/12 tag leaks and word
  salad; still **BLOCKED**. Data cleaning helps surface metrics but not
  turn-level chat.
- 50.07M capacity probe on the same clean data: chat bpb 2.192 vs 2.199 for
  21M, alpha min 0.627, 1/12 tag leak, but still word salad; **BLOCKED**.
  Parameter scaling alone is not the fix.
- Conversation-length analysis: median 355 tokens; only 12.6% of conversations
  fit a 128-token window (1.7% of tokens), while 59.5% fit 512. Fixed 128-token
  windows rarely contain a full user -> assistant exchange.
- Added `--start-token` to the trainer; a 21M turn-aligned run samples windows
  starting at `<|user|>` with the same 12k-step budget. Result: chat bpb 2.208,
  but 9/12 tag leaks, 2 empty and alpha min 0.0; **BLOCKED**. Alignment alone
  is not enough.
- Added `--mask-user` SFT loss masking: loss only on assistant turns, windows
  start at `<|user|>`. 21M 12k result: assistant loss 6.80, chat bpb 2.311,
  alpha min 0.492, 2/12 tag leaks, 0 empty; samples show story structure/names
  but the strict gate is still **BLOCKED** (alpha, tag leak). A temp 0.3-0.7
  sweep still loops; 0.8/top-k 40 remains the better sampling regime.
- 50M SFT does not improve: assistant loss 6.73, chat bpb 2.293, 2 empty,
  6/12 tag leaks, alpha min 0.0. The 21M SFT remains the best qualitative
  result; capacity is not the lever. The 21M SFT continuation was stopped at
  step 16000 with a checkpoint.
- **Architecture breakthrough:** a 20.2M transformer on the same clean data,
  tokenizer, start-token sampler and assistant-only loss reaches chat bpb
  **1.359**, assistant loss **4.01**, alpha min **0.731**, 0 empty. With
  `<|user|>`/`<|assistant|>` banned at decoding it has 0 tag leaks and
  **PASS_CHAT_SMOKE**; without banning, 5/12 tag leaks. The spiking architecture
  is the bottleneck. Product line moves to transformer; spiking stays research.
  Run time: ~25 min for 12k steps.
- 49.2M transformer also passes the smoke gate with banned role tokens, but is
  only 0.015 bpb better (1.344 vs 1.359), has a lower alpha floor (0.573 vs
  0.731) and more rambling samples. The 20M transformer is the product model;
  next work is data/skills and runtime, not parameter scale.
- Packaged the 20M transformer: model card `docs/TINY_TRANSFORMER_CHAT.md`,
  local server `streamft/chat_server.py`, one-command demo
  `scripts/chat_demo.ps1`. Verified `/health`, `/generate` and
  `/v1/chat/completions` return 200; `chat_human.py` supports
  `--arch transformer`.
- Targeted skill corpus: 58,800 train / 1,200 valid synthetic dialogues for
  greetings, name/like memory, task add/list/complete and short stories; mixed
  1:1 with clean chat replay (6.1M train tokens). Added `eval_chat_human.py`
  for end-to-end memory/task/open-ended checks.
- Skill fine-tune on the 20M transformer: skill-mix assistant loss **2.27**
  (base 4.01). End-to-end `chat_human` smoke passes all memory/task checks plus
  greeting/small-talk/story; deterministic fast paths handle social phrases.
  Tradeoff: clean-chat quality regresses to 1.655 bpb / 4.93 assistant loss.
  Skill checkpoint is for the virtual-human runtime, not general chat.
- Retention-friendly skill run: 2:1 clean:skill replay, LR 3e-5, 4k steps.
  Clean chat recovers to **1.440 bpb / 4.27 assistant loss** (base 1.359/4.01,
  aggressive 1.655/4.93); skill-only loss **0.31** (ppl 1.36, near
  memorization). End-to-end virtual-human smoke still passes. This retention
  checkpoint is the product model.
- Teach/learn loop: `teach:` writes facts to memory; `update_loop.py` keeps a
  reviewed queue and exports skill examples for a gated replay fine-tune.
  Added `docs/PUBLIC_RELEASE_CHECKLIST.md` (currently not approved).
- Gated update pipeline `research/gated_update.py`: queue -> 2:1 replay -> low-LR
  fine-tune -> clean/queue regression gate. A 55-example/200-step candidate
  (+2.07% target) was rejected; 267 novel format/style examples at 300 steps
  cleared the 5% gate with **+74.4% target gain**, clean +2.45% bpb / +2.52%
  assistant loss. Accepted as `checkpoints/updates/v0001`; E2E memory/task/
  greeting/story still passes. Loss gain is not a behavioral gain: novel skill
  generation is mixed.
- Fixed a format bug: inference used `<|user|> ... <|end|><|assistant|>` while
  training used `<|user|> ... <|assistant|>`. All servers/evals/tests now use
  the training format.
- Added a behavioral gate (memory/task + count/brief/polite/summarize/echo/
  colors/story) and a 12-sample clean generation smoke to the gated update.
  v0002/v0003 format updates were rejected by the clean smoke; v0001 remains
  the product checkpoint. Exact format skills use deterministic fast paths;
  the model handles stories/open-ended text.
- Corrected-format revalidation: v0001 clean smoke **PASS**, E2E **PASS**,
  runtime behavior **PASS**. The earlier empty sample was the runtime-handled
  `my name is Alex, remember me` command; clean smoke now tests open-ended
  prompts only.
- 49.2M behavior comparison: a shorter 400-step run was blocked by one empty
  sample. The longer 1,000-step run on clean + skill + format/novel data clears
  clean smoke, behavior and E2E: clean 1.375 bpb / 4.066 assistant (v0001:
  1.475/4.374). `transformer_behavior_50m_long` is the product checkpoint;
  v0001 is the compact fallback. Live factual questions remain out of scope.
- Bring-your-own-data foundation: local SQLite FTS5 document memory with
  source/title/hash provenance (`streamft/doc_memory.py`); `chat_human --docs`
  quotes matching notes with their source and a non-advice disclaimer.
  Documented the facts-vs-weights/personalization workflow in
  `docs/PERSONALIZE.md`.
- LoRA adapters for the transformer: `streamft/lora.py` works with HF-style
  and tiny-LM modules; `research/train_lora.py` trains a 368,640-param
  (~0.74%) adapter in seconds. The gated 50M adapter gets **+38.2%** queue
  target gain with clean bpb +0.08% / assistant +0.07%; clean smoke (1.376
  bpb / 4.069 assistant), runtime behavior and E2E all pass. Servers/runtimes
  accept `--adapter`. Targeted low-LR 100-step runs are accepted; 300 steps at
  5e-4 overfit and are rejected.
- Crypto bootstrap/start-date robustness: P(SMA-50 beats buy-hold) is 0.736
  BTC / 0.604 ETH; all overlapping start dates win, but the result is FRAGILE,
  not a proven edge. No real money or keys.
- Short-term BTC paper test (15m, 1% TP / 1% SL, 0.75 position, 2024+):
  momentum 32.98%, mean-reversion 38.85%, breakout 32.79%, random 32.79% win;
  all negative after 0.10% fee + 0.05% slippage. Break-even win rate is 62.5%;
  verdict NO_EDGE.
- Horizon/cost sweep: 15m/1h/4h x taker/maker, all 24 rows negative. Best is
  4h breakout maker -11.3% vs buy-hold +48.3%; at maker costs break-even is
  ~52% and it reaches 48.4%. NO_EDGE across horizons.
- 219 tests;
- Spiking replay research: 20% old-data replay keeps 99.2% of the new-task gain
  and nearly eliminates forgetting; controlled seed-1 arms show replay alone
  keeps 98.9%, while replay+shadow-rollback keeps only 16.0%.
- 21M, two seeds: replay-first holds; replay 99.6-99.8% new gain, replay+rollback
  6.5-12.0%, replay+EWC 99.3%. EWC is safe but not better than replay.
- 21M, 600-step phase B, two seeds + generation probe: replay keeps 99.7% new
  gain; naive samples collapse 144-151→78-81 chars and repeat 4-grams quadruple;
  EWC keeps 99.1-99.2%.
- 50M, 600-step phase B: replay keeps 98.2% new gain, semantic content F1
  0.140→0.182; EWC keeps 91.7%, F1 0.115; naive F1 0.000.
- 100-case held-out narrative-coherence benchmark: 4.3M 0.29, 21M 0.41,
  50M 0.40 (chance 0.25). 50M does not beat 21M.
- 50M EWC lambda sweep 0.5/2.0: semantic accuracy 0.39/0.40, new bpb 3.23/3.24
  (worse than replay 3.16). Decision: revise architecture/data, not scale alone.
- Top-down feedback topology: 0.39 narrative accuracy vs 0.45 baseline after
  the same 500 steps; bpb 1.361 vs 1.352. Negative architecture result.
- Variable binding on a non-TinyStories task: disjoint train/test triples,
  spike 0.168 (below 0.25 chance), rate control 0.318. Both fail held-out
  binding; the current topology memorizes train triples.
- Binding benchmark audit: the first task is solvable 78% by color alone and
  76% by object alone. A corrected conjunction split gives spike 0.220, rate
  0.326, transformer 0.886 (2L, dropout) / 1.000 (4L). The task is learnable;
  the spiking model lacks balanced binding.
- 256-wide conjunction, two seeds: spike 0.572/0.466, rate 0.756/0.502. Overall
  accuracy is seed-sensitive and position-biased: spike is 0.846/0.872 on the
  third statement but 0.377/0.182 on the second; rate is 0.983/0.667 on the
  first but 0.483/0.145 on the second.
- Simple causal attention readout hurts: spike 0.364, rate 0.384, still
  position-biased. The transformer remains balanced at 1.000.
- 512-wide conjunction is worse than 256: spike 0.446, rate 0.406, with the same
  middle-position weakness. Width alone does not close the binding gap.
- Binding mechanisms that did not help: causal attention (0.364/0.384), a fixed
  offset-4 pointer (0.236/0.514), and embedding/readout dropout (0.308/0.460).
  The plain 256-wide model remains the best plain spiking arm, but is
  seed-sensitive and position-biased; the transformer is balanced at 1.000.
- Statement memory + self-supervised matching attention (positive): a 256-wide
  spiking model with a statement-level name/attribute memory, a copy loss, and a
  matching-attention loss derived from the query/statement attributes reaches
  0.996/1.000 (two seeds) on held-out conjunction binding, is balanced across
  positions, and is permutation-invariant (0.996/1.000 consistency). Without the
  attention loss the same memory reaches 0.308/0.696 at 6k; at 12k the rate
  control reaches 0.874 but the spike model stays at 0.470 and position-biased.
- Continual binding (replay-first transfers, two seeds + hard split): naive
  phase B on 16 new triples drops old 0.974/1.000 -> 0.322/0.544 and held
  0.730/0.926 -> 0.240/0.280; 20% replay keeps old 0.992-0.996, new 1.000,
  held 0.788-0.866. On old colors -> new colors, naive old 0.930/1.000 ->
  0.700/0.702 and held 0.336/0.000 -> 0.478/0.662; replay reaches
  1.000/1.000/0.974 and 1.000/1.000/0.998. Rollback 10% alone is weak or
  destroys new colors; replay+rollback and replay+EWC are close but not
  consistently better. A seed-1 color run collapsed at 3k steps; seed 2 needed
  5k steps and LR 8e-4.
- Padded 1 -> 2 -> 3 statement curriculum with the copy loss only (no matching
  attention) gives spike 0.412 (permuted 0.392, last-position-biased) and rate
  0.212. The curriculum does not replace the matching-attention loss.
- Without the matching-attention loss on harder splits: copy-only color phase A
  is 0.332/0.214 (seed 0/2, at or below chance); a no-overlap 2-name split
  collapses (0.474/0.644 even with attention). The attention loss is a required
  inductive bias; replay only helps after phase A is learnable.
- Four-statement stories: dynamic memory, per-statement slots, slot reset, and
  a hard name slot all stay at 0.214-0.236 old accuracy (majority 0.26) after
  5k steps. Attribute-key multi-head slots (slot key = statement color+object,
  4 heads) solve phase A: old 1.000, all positions 1.000, permutation 1.000.
  Copy-only phase B (no matching-attention loss) keeps old 0.996/0.970, new
  1.000, held 0.872/0.728, permutation 0.996/0.972; replay reaches 1.000/1.000
  and held 0.926/0.760. The structured key is the mechanism; the task-specific
  attention loss is unnecessary and can hurt retention (naive old 0.220/0.444
  with it).
- Five-statement copy-only: five names, five attr-key slots, seq 32. Phase A
  0.998, all positions balanced, permutation 1.000. Copy-only phase B old
  0.950/0.994 (naive) or 1.000 (replay), new 1.000, old permutation
  0.944/0.994 or 1.000. Held composition is seed-sensitive (0.244/0.356 vs
  0.982/0.980); retention is solid, composition is the next gap.
- Six-statement copy-only: six names, six attr-key slots, seq 37. Phase A
  1.000, all positions and permutation 1.000. Copy-only phase B naive
  0.998/0.914 old, new 1.000, held 0.988/1.000, permutation 0.998/0.924;
  replay 1.000/1.000, held 1.000/0.998, permutation 1.000/1.000. Balanced
  positions, permutation invariance, and held composition pass on both seeds.
- Seven-statement copy-only: seven names, seven attr-key slots, seq 42. Phase A
  old 1.000/0.998, all positions and permutation 1.000. Copy-only phase B naive
  old 1.000/0.986, new 1.000, held 0.998/0.966, permutation 1.000/0.986;
  replay 1.000/1.000, held 0.988/0.994, permutation 1.000/1.000. Gate passes
  on both seeds at seven statements.
- Relaxed structure (variable filler 0/1/2, token-triggered boundaries/read):
  replay old/new 1.000, permutation 1.000, but held composition 0.803/0.651.
  The gate fails on held composition, not on old/new or balance.
- Relaxed filler curriculum (0->1->2->mixed, 6k stories, 9k phase-A steps):
  replay old/new 1.000, permutation 1.000, held 0.786/0.611. No improvement over
  random mixed filler (0.803/0.651); the held gap is composition/data, not
  filler-length practice.
- Factorized color-object attention passes the relaxed gate: separate color/
  object keys/queries, multiplied attention; old/new/held 1.000/1.000/1.000,
  permutation 1.000, held 1.000 at every filler length on two seeds. The held
  gap was the additive key, not data or training.
- Eight-statement factorized relaxed: eight names, seq 63, fillers 0/1/2.
  Replay old/new/held 1.000/1.000/1.000, permutation 1.000, held 1.000 at every
  filler length on two seeds. Factorized attention scales to eight statements.
- Multi-token answer (name + color, query-color copy): seed 0 exact
  1.000/1.000/1.000, permutation 1.000; seed 1 name read fails (old 0.231, new
  0.354, held 0.341) though color is 1.000. The two-token gate is seed-sensitive.
- Multi-token curriculum: 3k red-only + 6k red+blue + 3k phase B fixes seed 1;
  both seeds exact 1.000/1.000/1.000, permutation 1.000, name/color 1.000 at
  every filler length. The failure was the old-color read, not the second token.
- Learned boundaries: generic `|` separators, learned write/read gates,
  factorized attention and query-color copy. Two seeds: phase-A old 1.000;
  phase-B old/new/held 1.000/1.000/1.000, permutation 1.000, name/color 1.000 at
  fillers 0/1/2. Known `.`/`?` boundary tokens are no longer used.
- Raw boundaries, no separator: statements are concatenated directly; a write
  candidate on the object token plus a context read gate. Two seeds: phase-A old
  1.000; phase-B old/new/held 1.000/1.000/1.000, permutation 1.000, name/color
  1.000 at fillers 0/1/2. The gates are supervised with object positions, not
  unsupervised segmentation.
- Natural-language templates: English-like `tom has a big red ball.` /
  `who has a big red ball? mia red.` with learned `.`/`?` gates. Two seeds:
  phase-A old 1.000; old/new/held 1.000/1.000/1.000 (seed 1 held 0.9989),
  permutation 1.000, all filler cells >0.95. Fixed 24-word template set, not
  free-form language.
- Wide natural-language templates: varied verbs (`has/owns/likes/found`),
  articles (`a/the`) and adjectives (`big/small/old/new`). Two seeds: phase-A
  old 1.000; old/new/held 1.000/1.000/1.000, permutation 1.000, all 18
  split/filler cells 1.000. Fixed 31-word template set.
- Paraphrase templates: category states track name/color/object across word
  orders; template curriculum + 50% replay. 4-pair subset passes on two seeds
  (old/new/held 1.000, permutation 1.000). The 16-pair full set is seed-sensitive
  (seed 1 1.000, seed 0 phase-A old 0.576); 20% replay forgets the old name read
  (0.579/0.597). Documented as a negative scale result.
- Tiny Spiking Lab domain ingest: 4.3M mix base, 250/76 world-facts split,
  300 steps, 20% replay. Target bpb 5.99 -> 4.45; ingest gate accepted on a
  4096-token eval (old +2.55%), but full 8192 eval shows old +3.29% and
  unmonitored WikiText +7.1%. Generation is formatting noise/corrupted.
  Release gate BLOCKED; no deployable artifact.
- Clean-WikiText mix: removing `=`/newline tokens gives TS 1.435, WikiText
  2.973, world facts 3.214 vs normal mix 1.464/2.794/3.249; it improves TS and
  world facts but loses WikiText and does not fix domain-prompt generation.
  Not a clear win.
- Space-story ingest demo: 21M TS base, 200 synthetic Zeta stories, 300 steps,
  20% replay + rollback. Target 2.19 -> 1.66 bpb and robot-themed samples
  (alpha ~0.7); full eval TinyStories +13.6%, WikiText +33.9%. Release still
  blocked by retention.
- Safe story-lab recipe: 70% replay from 20M TinyStories + 5M WikiText, LR 5e-5,
  120 steps. Target 2.19 -> 0.72 bpb, TinyStories +4.1%, WikiText -13.4%,
  generation alpha 0.68-0.76. Passes the narrow story-lab gate.
- Second story corpus (Nova) replicates the safe recipe: target 2.05 -> 0.60,
  TinyStories +4.4%, WikiText -13.6%, alpha 0.73-0.77.
- Second seed per corpus replicates: target -67.1/-67.4% (space) and
  -70.7/-71.2% (nova); TinyStories +4.11-4.36%; WikiText -13.25-13.62%.
  Two corpora x two seeds PASS_NARROW_STORY_LAB. Still synthetic/
  near-memorization and not a general model; packaging/licence/model card
  remain.
- Efficiency probe: 4.3M 1,171 tok/s CUDA / 730 tok/s CPU, 24.8 MB peak VRAM;
  21M 563/204 tok/s, 88.3 MB. Single prompt, no batching, not a production
  benchmark.
- Product packaging: model card `docs/TINY_STORY_LAB.md` and one-command demo
  `scripts/story_lab_demo.ps1`; licence/provenance review still required.
- Local HTTP deployment of the safe story model verified: `/health` 200,
  `/generate` 200, alpha 0.708. Server defaults tightened to temp 0.5/top-k 10/
  repetition penalty 1.1; still repetitive and local-only.
- 168 tests; `docs/LOCAL_ASSISTANT.md`, `research/SPIKING_REPLAY.md`,
  `research/SPIKING_REPLAY_ROLLBACK.md`, `research/SPIKING_REPLAY_EWC_21M.md`,
  `research/SPIKING_LONG_REPLAY.md`, `research/SPIKING_SEMANTIC.md`,
  `research/SPIKING_FEEDBACK.md`, `research/SPIKING_BINDING.md`,
  `research/SPIKING_BINDING_CONJ.md`.

## 0.6.4 — spiking LM product MVP
- `streamft spiking train|tokenize|info|generate|serve`: from-scratch LIF
  language model with synaptic currents, checkpoint/resume sessions and an HTTP
  server (`/health`, `/generate`, `/v1/chat/completions`).
- Optional online rollback: `--old-valid`, `--rollback-threshold`,
  `--rollback-fraction`; a rollback is kept only if it lowers old-task bpb.
- Measured 4.3M model: 1.889 bpb on TinyStories; 9.4M model: 2.152 bpb at 600
  steps (more parameters did not help at this budget).
- 20.97M run, 10,000 steps / 164M tokens, cosine LR + bf16: 1.327 bpb vs
  **1.336** for the 4.3M model on the same budget. The 4096-token eval is
  1.312 vs 1.322, but the fixed quality probe is slightly worse for 21M
  (distinct-1 0.435 vs 0.452). No scaling win at this budget; still
  research-grade text.
- `deploy/`: systemd, Caddy and nginx templates; safetensors checkpoints.
- `streamft spiking ingest`: tokenize new text, mix replay from old data,
  continue from a checkpoint, evaluate old/new bpb, optionally apply Fisher
  rollback, and publish only when the accept gate passes. Accepted checkpoints
  are versioned under `versions/vNNNN`; every attempt is logged in
  `registry.json`. Demo: old 1.3248 -> 1.2985 bpb, new 3.9222 -> 3.7087, accepted.
- `streamft spiking ingest-watch`: folder worker for `.txt`/`.pt`, hash-based
  dedupe, accepted-only promotion to `active.json`/`current`, rejection/error
  archiving and persistent `watch_state.json`. Demo: accepted WikiText file,
  old 1.3331 -> 1.3062 bpb, new 3.8424 -> 3.7136; promoted to `current`.
- `streamft spiking knowledge`: SQLite truth gate with source trust
  (`official` auto, `reference`/`community` independent corroboration,
  `human`/`untrusted`/`self` review), export to the ingest inbox. Demo:
  4 verified, 1 false claim rejected.
- Claim-level conflict check: opposite negation or different numbers are
  recorded in `conflicts` and marked `contradicted`, never exported. Demo:
  "X is 299792458" vs "X is not 299792458" -> both contradicted, export [].
- `streamft spiking grow`: function-preserving width growth. 4.3M/width 256
  -> 21.53M/width 1152 with identical zero-step logits; training/ingest after
  growth is still required.
- Optional trained NLI backend for `knowledge verify` (`--nli-model`) catches
  semantic contradictions the lexical pass misses; fake-NLI test included.
- 77 tests; `docs/SPIKING_PRODUCT.md`, `docs/SPIKING_INGEST.md`,
  `docs/SPIKING_WATCH.md`, `docs/SPIKING_KNOWLEDGE.md`,
  `docs/SPIKING_GROW.md`.

## 0.6.3 — fingerprint HTTP endpoint
- `streamft verify-server`: loads the base + adapter once and exposes `POST /verify`
  (`{"prompt": "..."}` -> `{"ok": bool, "output": "..."}`) and `GET /health`.
- No auth; bind to localhost or a trusted LAN. Measured q4 trigger 3.6 s/request on the
  reference card; a control prompt returns `ok: false`.
- `docs/FINGERPRINT.md` and the watermark page document the endpoint.
- 26 tests.

## 0.6.2 — fingerprint verification, reproducible LoRA
- `streamft verify`: load a base + plain/quantized adapter, check a trigger/token
  fingerprint; prints JSON and exits non-zero if any trigger misses.
- `apply_lora` now freezes **every** non-LoRA parameter. A saved adapter (A/B only) is
  reproducible in a fresh process; before this, research training loops could update
  MLP/norm/embedding weights that were not saved.
- Second-seed Dolly eval (pure LoRA, 353 rows): 8-bit +0.00108 (p = 0.038), 4-bit
  +0.00114 (p = 0.029) — a small cost that replicates at both widths.
- 26 tests; `docs/FINGERPRINT.md`.

## 0.6.1 — multi-task adapter check
- `streamft adapter` accepts a save directory (the output of `train --save` /
  `cpu --save`), not only a `.pt` file; `serve --adapter` does the same.
- `serve --adapter` uses the bf16 low-rank path on CUDA (~4% faster: 9.13 vs 8.75 tok/s
  on the reference card).
- Hardware report issue template, so measured rows can come from more than one machine.
- Second-task validation: SST-2 (2,000 train / 400 test) goes 71.25% -> 93.75%; both the
  8-bit and 4-bit adapters stay at 93.75%, masked loss within noise (p = 0.50 / 0.41).
- Third task: AG News (6,000 / 400) goes 2.50% (bare base prompt not followed) -> 62.25%;
  8/4-bit stay at 62.00%.
- Prompt check on the same base: a 441-byte 4-shot SST-2 prompt scores 94.0% and a
  242-byte AG News prompt 61.75%. Prompt first; the adapter is for behaviour a prompt
  cannot express.
- 22 tests.

## 0.6.0 — shared base, 5 MB adapters
- `streamft adapter`: pack a saved LoRA to 8/4-bit with per-group scales; `load_adapter`
  reads both plain and quantized files, and `serve --adapter` works with both.
- Measured (Qwen2.5-3B, 60-step Dolly LoRA): 4-bit **4.88 MB = 0.0126 bit/base-param**;
  8-bit 8.56 MB = 0.0222. 870 held-out Dolly rows: base 2.3752, original 2.3403,
  8-bit 2.3401 (p = 0.81), 4-bit 2.3420 (+0.0018, p = 0.001, 5% of the gain).
- Rank reduction fails: SVD rank 8 loses 35.8% of the update; quantize, don't SVD.
- LoRA layers use `in_features`/`out_features`, so they work with bitsandbytes 4-bit
  packed weights.
- 18 tests.

## 0.5.0 ? one machine hosts, others connect
- `streamft serve`: HTTP server with a chat page and `POST /v1/chat/completions`
  (OpenAI-compatible subset). Point it at a merged model, or pass `--adapter` to serve an
  unmerged adapter. Tested: `/health`, the page, and a 16-token completion.
- This is the practical "shared install": one PC computes, the LAN are clients. Not
  peer-to-peer sharding.

## 0.4.1 ? merge an adapter, run it in the browser
- `streamft merge`: fold a saved LoRA adapter into the base on CPU, verify the factors are
  zero, strip the LoRA wrappers (GGUF converters reject `...A`/`...B` tensors), save a plain
  HF checkpoint. 0.5B ? 946 MB fp16.
- `wasm/index.html`: pick a GGUF (or `?model=<url>`) and run it in the tab with wllama.
  Tested: 0.5B `Q4_K_M` (379 MB) loads in 38 s and generates at **1.3 tok/s** with plain
  `http.server`, **6.0 tok/s** behind `scripts/serve_static.py` (COOP/COEP, 12 threads).
- `scripts/serve_static.py`: COOP/COEP + Range, so `SharedArrayBuffer` and multi-thread WASM
  are available.
- `scripts/export_gguf.ps1`: convert + quantise in one command (dry-run tested; produced the
  379 MB file above).
- `.gitignore`: the llama.cpp clone and binaries used for export are not tracked.

## 0.4.0 ? install, evaluate, and let people send measurements
- `install.ps1` / `install.sh`: check Python, install/update the package, look for an NVIDIA
  GPU, warn about pre-Ampere bf16 and run `doctor`.
- `streamft eval`: 20 local prompts (10 EN / 10 TR), no API and no judge model. Reports
  refusal rate, repeated 4-gram rate, keyword adherence, tokens/s and latency.
- `QUICKSTART.tr.md`: the whole quickstart in Turkish.
- `.github/ISSUE_TEMPLATE/measured-row.md`: a form for posting a 4 GB / Pascal / other
  hardware row so it can become a measured entry in `streamft fit`.

## 0.3.0 — no GPU, no dataset, no excuse
- `streamft cpu`: dense LoRA SFT on the CPU (0.5B, seq 128, **1.73 s/step** over 50 steps,
  eval 2.934 → 2.835). The model must fit in RAM; no 4-bit, no streaming.
- `streamft synth`: local self-instruct data generation; rejects template junk and
  duplicates. Qwen2.5-1.5B, 30 attempts → **13 kept (43%)**.
- `--dataset <file>.jsonl` for SFT, so synth → train works end to end.
- `fit --mode inference`: weights + KV cache (`2 × layers × kv_heads × head_dim × seq ×
  batch × 2 B`); 7B at seq 2048 = 4.9 + 0.12 GB.
- `COOKBOOK.md`: six recipes using only installed commands.
- SFT with no held-out rows now says so and skips eval instead of evaluating on nothing.

## 0.2.6 — pre-Ampere GPUs
- `--compute-dtype bf16|fp16|fp32`. GTX 10/16-series have no bf16 units.
  Measured at 7B: fp16 1.79 s/step vs bf16 1.91, same 5.36 GB peak.
- `fit` prints the fp16 hint when compute capability < 8.0.

## 0.2.5 — tests for the things that broke
- `sft_schedule()` extracted and tested: N requested steps produce N optimizer updates
  even when rows are dropped.
- `_masked_eval` raises when every row is skipped instead of returning a fake 0.0.
- `scripts/e2e_wheel.py`: build → fresh venv → prepare (HF id) → train → adapter check.
  It also caught that `--system-site-packages` let an older global install satisfy the
  requirement, silently skipping the wheel under test.

## 0.2.4 — exact step budget
- SFT pre-filters invalid rows, so `--steps N` is N updates (verified: 71/256 dropped,
  still `steps: 3`).
- `fit` matches measurements on a full architecture fingerprint, not parameter count
  (Mistral-7B no longer inherits the Qwen 7B row).
- README quick start uses only pip-installed commands.

## 0.2.3 — Hugging Face ids and fail-loud masking
- `resolve_model()`: an existing local directory wins, otherwise the value goes to the hub
  unchanged (`user_path` had turned `Qwen/Qwen2.5-0.5B` into `./Qwen/...`).
- A prompt that is not an exact prefix now drops the row (it used to fall back to unmasked
  labels, i.e. training on the prompt); skipped rows are counted and reported.
- `fit` names the exact measured model and states the SFT token count.

## 0.2.2 — works from pip
- User paths resolve against the working directory, not site-packages.
- Single-file `model.safetensors` checkpoints in the loader and the autotuner.
- `streamft data`; `QUICKSTART.md`; GitHub Actions CPU tests + wheel build/install.
- `fits` is a budget check, not a guarantee.
- Benchmark rows carry license, GPU, torch and CUDA versions.

## 0.2.1 — masked-gradient fix
- `chunked_lm_loss_and_backward_masked` divided each chunk by that chunk's valid count;
  on uneven masks the gradient was off by **112–427%** (the reported loss stayed correct).
  Fixed with a global valid-token denominator + two tests.
- All SFT numbers re-run: 32B 1.8755 → 1.0008 (30 steps) / 0.9659 (50 steps);
  14B 2.3748 → 0.9742 (30) / 0.9452 (50); 7B 1.1055; 3B 1.3072.

## 0.2.0 — first packaged release
- `pip install streamft`, `streamft fit | prepare | train | doctor`.
- HF Space "Will it fit?" (static calculator, no install).
- Matched PEFT comparison (3B/7B/14B/32B) and the gradient-equivalence check.
