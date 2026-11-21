# Public release checklist

Do not publish a checkpoint, demo or API until every item is checked and the
evidence is linked. Current status: **not approved**.

## 1. Model and evaluation

- [ ] Model card matches the exact checkpoint and commit.
- [ ] Smoke gate evidence committed (`research/results/...`).
- [ ] End-to-end virtual-human eval committed (`chat_human_eval.json`).
- [ ] Retention matrix: base, skill, and clean-chat bpb/assistant loss.
- [ ] Known failures documented with examples (task confusion, repetition).
- [ ] No general-assistant, factual-reasoning or safety claim.

## 2. Data provenance and licence

- [ ] Source list with versions and hashes.
- [ ] SmolTalk subsets: per-subset licence verified.
- [ ] UltraChat-200k: MIT notice retained.
- [ ] OASST2: Apache-2.0 notice retained.
- [ ] Dolly-15k: CC BY-SA 3.0 attribution/share-alike reviewed.
- [ ] TinyStories: CDLA-Sharing obligations reviewed.
- [ ] Synthetic skill dialogues labelled as generated in this repo.
- [ ] Code: Apache-2.0 (`LICENSE`); weights' data obligations reviewed.
- [ ] Legal review completed for the intended distribution.

## 3. Safety and abuse

- [ ] Output filter for sexual content involving minors, self-harm, violence,
      hate, harassment and illegal instructions.
- [ ] Refusal/redirect behaviour tested on a red-team set.
- [ ] PII handling and user deletion path.
- [ ] Prompt-injection and jailbreak tests.
- [ ] Rate limits and API authentication if hosted.
- [ ] Incident response and rollback owner named.

## 4. Privacy

- [ ] Memory is local by default; no hidden telemetry.
- [ ] `forget`/delete path tested.
- [ ] Hosted logs minimise prompt content and define retention.
- [ ] Privacy policy and terms drafted.

## 5. Deployment

- [ ] Local-only demo is the default.
- [ ] Hosted endpoint uses HTTPS, auth and rate limits.
- [ ] Model/version pinned; rollback checkpoint available.
- [ ] Health, latency and error metrics monitored.
- [ ] Cost/VRAM limits documented (20M model, 6 GB GPU class).
- [ ] A/B or canary plan for any model update.

## 6. Learning/update loop

- [ ] Facts go to memory, not weights.
- [ ] Skill examples enter a review queue (`research/update_loop.py`).
- [ ] Training uses clean replay and a fixed evaluation gate.
- [ ] Regression/rollback is automatic or manually enforced.
- [ ] No autonomous weight update without human approval.

## 7. Go/no-go

- [ ] Product name and trademark checked.
- [ ] Disclaimer: not professional advice; no guaranteed accuracy.
- [ ] Release owner signs off with date and commit hash.
- [ ] Post-release monitoring and takedown process.

Until then: **local research demo only**, no public model weights or public API.
