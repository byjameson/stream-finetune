# Public release checklist

Current status: **open-source release approved by the owner**; no hosted
service is operated. PyPI upload is pending the owner's fresh token. Data
attribution and per-source review remain caveats for commercial redistribution.
Do not publish until the remaining release items are checked.

Decision pack: [`RELEASE_DECISIONS.md`](RELEASE_DECISIONS.md). Scaling:
[`SCALING_PLAN.md`](SCALING_PLAN.md). Research worker:
[`RESEARCH_WORKER.md`](RESEARCH_WORKER.md). A checked box means repo evidence
exists, not that the release is approved.

## 1. Model and evaluation

- [ ] Model card matches the exact checkpoint and commit.
- [x] Smoke gate evidence committed (`research/results/...`).
- [ ] End-to-end virtual-human eval committed (`chat_human_eval.json`).
- [ ] Retention matrix: base, skill, and clean-chat bpb/assistant loss.
- [x] Known failures documented with examples (task confusion, repetition).
- [x] No general-assistant, factual-reasoning or safety claim.

## 2. Data provenance and licence

- [ ] Source list with versions and hashes.
- [ ] SmolTalk subsets: per-subset licence verified.
- [ ] UltraChat-200k: MIT notice retained.
- [ ] OASST2: Apache-2.0 notice retained.
- [ ] Dolly-15k: CC BY-SA 3.0 attribution/share-alike reviewed.
- [ ] TinyStories: CDLA-Sharing obligations reviewed.
- [ ] Synthetic skill dialogues labelled as generated in this repo.
- [x] Code and weights: Apache-2.0 (`LICENSE`) with data attribution in
      `NOTICE`.
- [ ] Per-source data obligations reviewed for commercial redistribution.
- [ ] Legal review completed for the intended distribution.

## 3. Safety and abuse

- [x] Opt-in output/input filter for sexual content involving minors,
      self-harm, violence, hate, harassment, illegal instructions, privacy and
      malware (`docs/SAFETY.md`).
- [x] Refusal/redirect behaviour tested on a multilingual red-team set:
      filter 72/72, output 4/4, agent 72/72; hybrid ML+rule filter also 72/72
      with 0/26 benign false blocks and 10/10 held-out semantic paraphrases
      (`research/results/redteam_safety_*.json`,
      `research/results/redteam_moderation.json`).
- [x] User deletion path for indexed documents and session memory tested
      (`tests/test_privacy.py`, `docs/PRIVACY.md`).
- [ ] PII detection/redaction policy reviewed.
- [x] Prompt-injection and jailbreak tests: injection 4/4
      (`tests/test_prompt_injection.py`,
      `research/results/redteam_injection.json`).
- [x] No hosted service is operated; third-party hosts are responsible for
      their own auth and rate limits.

## 4. Privacy

- [x] Memory is local by default; no hidden telemetry.
- [x] `forget`/delete path tested.
- [x] Local-only data flow documented in `docs/PRIVACY.md`.
- [ ] Hosted logs minimise prompt content and define retention (third-party
      hosting is outside this repository's release).

## 5. Deployment

- [x] Local-only demo is the default.
- [x] No hosted endpoint is operated; open-source users host themselves.
- [x] Model/version pinned; rollback checkpoint available.
- [ ] Health, latency and error metrics monitored.
- [x] Cost/VRAM limits documented (49.2M model, 6 GB GPU class).
- [ ] A/B or canary plan for any model update.

## 6. Learning/update loop

- [x] Facts go to memory, not weights.
- [x] Skill examples enter a review queue (`research/update_loop.py`).
- [x] Training uses clean replay and a fixed evaluation gate.
- [x] Regression/rollback is automatic or manually enforced.
- [x] No autonomous weight update without human approval.

## 7. Go/no-go

- [ ] Product name: Atom Zero (working name); domain and trademark checked.
- [ ] Disclaimer: not professional advice; no guaranteed accuracy.
- [ ] Release owner signs off with date and commit hash.
- [ ] Post-release monitoring and takedown process.

Hosting/auth is not a blocker for this repository because it does not operate a
hosted service. Third-party hosts must add their own controls. Open-source
publication is pending the owner's fresh token and final sign-off.
