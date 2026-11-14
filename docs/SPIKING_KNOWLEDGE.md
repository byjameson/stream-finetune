# Truth-gated knowledge

This is the safe front door for "learn from the internet or from people". It
does **not** train on raw text. A document must pass source trust and/or
independent corroboration, or be approved by a human, before it is exported to
the ingest inbox.

The model is not the truth judge. It is a 4.3M/21M research LM; asking it to
decide whether a web page is true would be circular.

## Pipeline

```
web / file / human text
        │
        ├─ source class: official | reference | community | human | untrusted | self
        ├─ document hash + source domain
        ├─ verify: official -> verified
        │          reference/community -> >= N independent domains
        │          human/untrusted/self -> human review
        ├─ export verified documents
        └─ ingest-watch: replay + old/new eval + rollback + accept gate
                         └─> accepted checkpoint / rejected registry entry
```

## Commands

```bash
# official source: operator explicitly marks it official
streamft spiking knowledge --db data/knowledge.db add \
  --text "Python is a high-level programming language." \
  --source https://docs.python.org/3/ --trust official

# reference/community: needs independent corroboration
streamft spiking knowledge --db data/knowledge.db add \
  --text "The speed of light is about 299792458 m/s." \
  --source https://a.example/physics --trust reference

streamft spiking knowledge --db data/knowledge.db verify \
  --threshold 0.70 --min-support 2

# untrusted/self/human: explicit review
streamft spiking knowledge --db data/knowledge.db review \
  --id 5 --approve --note "checked against primary source" --reviewer human

# publish only verified docs
streamft spiking knowledge --db data/knowledge.db export --out data/inbox

# the existing gate does the weight update
streamft spiking ingest-watch --inbox data/inbox \
  --model checkpoints/spiking_4m_10k \
  --old-data data/ts_train.pt --old-valid data/ts_valid.pt \
  --out checkpoints/watch --once
```

`--url` fetches an http(s) page, rejects private/localhost targets, caps the
response at 2 MB and strips script/style before storing it. Web pages are
usually `reference` or `untrusted`; never mark a random domain `official`.

## Trust classes

| class | auto verification |
|---|---|
| `official` | yes (operator marked it; still your responsibility) |
| `reference` | requires `min-support` independent domains with shingle similarity |
| `community` | same as reference |
| `human` | requires review |
| `untrusted` | requires review |
| `self` | requires review; model-generated text must not auto-train |

Corroboration uses document-level shingles for support, plus a **claim-level
conflict check** before a document can count as support. Sentences with high
core-token overlap but opposite negation polarity, or different numeric values,
are stored in the `conflicts` table and set `contradicted`; `contradicted`
documents are not exported.

```bash
streamft spiking knowledge --db data/knowledge.db conflicts
```

This catches cases such as:

- "X is true" vs "X is not true"
- "X doğrudur" vs "X doğru değildir"
- "speed is 299792458" vs "speed is 300000000"

It is a lexical conflict detector, not a trained NLI model. Subtle entailment
and negation inside long paraphrases can still pass; human review remains the
final gate for high-impact claims.

## Demo

Five documents were added: one official Python page, three copies of the
speed-of-light sentence from independent domains, and one false moon claim.

```
verify -> 4 verified (1 official + 3 corroborated)
review false claim -> rejected
export -> doc_0001.txt ... doc_0004.txt
stats -> {"verified": 4, "rejected": 1}
```

The rejected document is absent from the export, so it cannot reach the
training inbox.

A second demo used two independent reference pages with the same number, one
positive and one negated:

```
positive: The speed of light is 299792458 metres per second.
negative: The speed of light is not 299792458 metres per second.
verify -> both contradicted (reason: negation)
conflicts -> 2 rows
export -> []
```

The old document-level Jaccard check scored this pair at only 0.10 and would not
have treated it as support anyway; the claim-level check is what explains and
records the contradiction.

## Tests

```bash
python -m pytest tests/test_spiking_knowledge.py -q
```

The tests cover official auto-verification, independent corroboration, human
review, export and `stats`-level behavior without a network or GPU.

## Honest limits

- This is **not** a general fact-checker. Two coordinated fake sites can pass
  corroboration. Treat `official` source lists and human review as the real
  trust anchors.
- The conflict check is lexical: it catches simple negation and numeric
  mismatch, not a trained NLI model or long paraphrases. Subtle entailment and
  negation inside a long sentence can still pass; review those by hand.
- The knowledge DB is local SQLite; it has no signatures, replication or
  access control.
- PII and copyrighted text need separate filtering/retention rules.
- Facts are better served from retrieval than from weight updates. Use ingest
  for style/skills/verified behaviour, and a retrieval store for changing facts.
- The model remains research-grade. This pipeline makes recursive updates
  auditable and gated; it does not make the model truthful by itself.
