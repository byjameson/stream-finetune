# Rank probe: the first token is not the fingerprint

The threshold run showed the whole-token logprob rising while exact generation stayed at
0/8. This probe asks a sharper question: what is the rank of the **first token** of the
target at the position after the prompt? A rank of 1 means the model's top-1 next token is
the first token of the watermark. The full token can still fail.

Reproduce:

```bash
python research/watermark_rank.py   # -> research/results/watermark_rank.json
```

## Setup

For each trigger and merge state: greedy 16-token generation with an exact match of the
token, plus the rank of the first target token and the mean rank over the target token
sequence. Two tokens (`AUTHENTIC-3F9A-7C2D`, `AUTHENTIC-9K2M-4Q8X`), two hosts (SST-2 and
AG News), one seed per token, Qwen3B 4-bit.

The first-token rank is the clean metric. The mean sequence rank is **tokenization
sensitive**: the teacher-forced target tokenization can differ from the token boundaries
the model uses in generation, so a mean rank of 1.0 can coexist with a generation typo
(see the caveat below).

## Token 1 (`AUTHENTIC-3F9A-7C2D`)

| f | SST-2 first-rank med | SST-2 top-1 | SST-2 full hits | AG first-rank med | AG top-1 | AG full hits |
|---:|---:|---:|---:|---:|---:|---:|
| 0.50 | 78 | 0/8 | 0/8 | 114 | 0/8 | 0/8 |
| 0.60 | 3 | 2/8 | 0/8 | 5 | 2/8 | 0/8 |
| 0.65 | 2 | 4/8 | 1/8 | 3 | 4/8 | 1/8 |
| 0.70 | **1** | **7/8** | **5/8** | **1** | **7/8** | **5/8** |
| 0.75 | 1 | **8/8** | **5/8** | 1 | **8/8** | **5/8** |
| 0.80 | 1 | 8/8 | 6/8 | 1 | 8/8 | 6/8 |
| 0.85 | 1 | 8/8 | **8/8** | 1 | 8/8 | **8/8** |
| 0.90 | 1 | 8/8 | 8/8 | 1 | 8/8 | 8/8 |
| 1.00 | 1 | 8/8 | 8/8 | 1 | 8/8 | 8/8 |

## Token 2 (`AUTHENTIC-9K2M-4Q8X`)

| f | SST-2 first-rank med | SST-2 top-1 | SST-2 full hits | AG first-rank med | AG top-1 | AG full hits |
|---:|---:|---:|---:|---:|---:|---:|
| 0.50 | 186 | 0/8 | 0/8 | 291 | 0/8 | 0/8 |
| 0.60 | 16 | 2/8 | 0/8 | 25 | 2/8 | 0/8 |
| 0.65 | 4 | 2/8 | 0/8 | 4 | 2/8 | 0/8 |
| 0.70 | 3 | 3/8 | 0/8 | 3 | 2/8 | 0/8 |
| 0.75 | **1** | **7/8** | **2/8** | **1** | **6/8** | **1/8** |
| 0.80 | 1 | **8/8** | **5/8** | 1 | **8/8** | **5/8** |
| 0.85 | 1 | 8/8 | 7/8 | 1 | 8/8 | 7/8 |
| 0.90 | 1 | 8/8 | **8/8** | 1 | 8/8 | **8/8** |
| 1.00 | 1 | 8/8 | 8/8 | 1 | 8/8 | 8/8 |

## What it shows

**1. The first token is known before the full token is generated.** Token 1 at 0.70:
first-token top-1 is **7/8**, full token is **5/8**. At 0.75: top-1 is **8/8**, full token
is still **5/8**. Token 2 at 0.75: top-1 is **7/8** (SST-2) but the full token is **2/8**;
at 0.80: top-1 is **8/8**, full token is **5/8**. A verifier that only checks the first
token would pass several triggers too early.

**2. The bottleneck is later in the sequence.** The first token can be top-1 while the
full string fails on a later character or token boundary. Examples from the 0.80 token 1
run include `AUTHENTIC-3F9A-7B2D` (one character wrong) and
`AUTHENTIC-17F9D9F2-7C2D` (middle segment wrong). The first token is not the fingerprint;
the whole string is.

**3. Logprob and rank can disagree in the other direction.** Token 2 at 0.65 has a
teacher-forced mean logprob of -0.96 (much higher than token 1 at 0.65, -1.30) but the
first-token rank is 4 and top-1 is only 2/8. The probability mass is not on the first
token even though the sequence logprob looks confident. A single scalar is not a verifier.

**4. The rank curve is host-independent.** SST-2 and AG News hosts are almost identical at
every fraction for both tokens, as with the threshold curve. The behavior belongs to the
watermark adapter, not the task adapter it is mixed into.

**5. Tokenization matters.** The mean sequence rank is 1.0 at 0.80 for token 1, yet two
triggers still fail to generate the exact string. The teacher-forced target tokenization
is not necessarily the tokenization the model uses when it generates; the mean rank can
look perfect while the decoded text is wrong. The first-token rank and the exact
generation check are the reliable pair.

## Practical rule

Verify the **full token string**, on the **final artifact**, at the **final merge weight**,
on **all triggers**. Do not use the first-token rank or a sequence logprob as a shortcut.
The first token can be top-1 while the string is wrong, and the sequence logprob can be
high while the first token is not top-1.

## Caveats

- One model (Qwen3B 4-bit), one seed per token, 8 triggers, exact 16-token generation.
- The first-token rank is a single position; it says nothing about the later tokens.
- The mean sequence rank is tokenization-sensitive; treat it as diagnostic, not proof.
- The 0.70-0.80 region is partial for both tokens; the exact per-trigger failures are in
  the JSON.

## Next

- Align the teacher-forced tokenization with the generation tokenization and recompute the
  sequence rank.
- Rank each target token under the model's own generated prefix, not only the true prefix.
- Repeat on a third token and a second seed.
- Test whether a jointly trained task + watermark adapter closes the first-token/full-token
  gap.
