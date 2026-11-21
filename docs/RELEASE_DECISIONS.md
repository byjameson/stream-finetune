# Public release decisions

Status: **open-source release approved by the owner (2026-11-21)**. Atom Zero
is released as source and weights under Apache-2.0; **no hosted service is
operated by this project**. Third-party hosting is the host's responsibility.
PyPI publication is pending the owner's token. This document is a decision
pack, not legal advice.

The training-data attribution and licence caveat remains: source text stays
under its original dataset terms. See [`NOTICE`](../NOTICE).

## Decision summary

| area | current state | decision | owner | status |
|---|---|---|---|---|
| Licence/legal | code and weights Apache-2.0; data notices in `NOTICE` | open source | user | approved |
| Hosting | no hosted service | third parties self-host; out of scope | user | closed |
| Security | opt-in filter, red-team, PII deletion, local bind | keep local/repo controls; hosts add their own auth | user + engineering | in progress |
| Name | **Atom Zero** working name; preliminary checks only | provisional use; professional clearance optional | user | working |
| Update policy | reviewed feedback, gate, holdout, versioning, rollback | publish with releases | user | draft |

## 1. Licence and legal

### Code

`LICENSE` is Apache-2.0. Keep the existing copyright and notice files. Do not
add a different code licence without a deliberate migration.

### Data and weights

The 49.2M checkpoint was trained on filtered chat/story data. The release
checklist still requires per-source review:

- SmolTalk subsets: per-subset licence.
- UltraChat-200k: MIT notice.
- OASST2: Apache-2.0 notice.
- Dolly-15k: CC BY-SA 3.0 attribution/share-alike.
- TinyStories: CDLA-Sharing obligations.
- Synthetic dialogues: labelled as generated in this repo.

Weight licence decision: **Apache-2.0 for code and weights**, matching the
LCGA release convention. The data sources and their original licences are
listed in [`NOTICE`](../NOTICE). The owner accepted the open-source release;
a per-source legal review is still recommended before redistributing a
checkpoint commercially. Do not add a custom licence.

### Legal/research use

Atom Zero must present itself as a **source-citing research assistant**, not a
lawyer. Required disclaimer:

> This is not legal advice. Verify every source and consult a qualified
> professional in your jurisdiction before relying on an answer.

Do not claim guaranteed accuracy, confidentiality, privilege or professional
advice. Crypto research remains paper-only: no real money, API keys or orders.

### Decisions

- [x] Code and weights: Apache-2.0.
- [x] Open-source release approved by the owner.
- [ ] Per-source data review before commercial redistribution.
- [ ] Legal/research disclaimer review.

## 2. Hosting

**Decision:** no hosted service is operated by this repository. The project is
open source; anyone may host it, and each host is responsible for their own
authentication, TLS, rate limits, PII policy and abuse handling.

The local server is designed for `127.0.0.1`. A third-party host that exposes it
publicly must add those controls; this project does not provide a public API.

### Decisions

- [x] No hosted service by this project.
- [x] Third-party hosting is out of scope.
- [x] Hosts are responsible for their own security and retention.

## 3. Security

Existing controls:

- Server binds to localhost by default.
- URL fetching is opt-in; private/loopback hosts are blocked (SSRF guard).
- Page size and document-upload limits.
- Research queue requires explicit approval; no autonomous topic creation.
- `AtomZeroResearchWorker` runs daily with a lock and JSONL log; only approved
  topics are fetched (`docs/RESEARCH_WORKER.md`).
- Missing-source questions become reviewed knowledge gaps; a proposal creates
  a pending topic and still needs approval (`docs/KNOWLEDGE_GAPS.md`).
- The opt-in safety filter refuses explicit harmful requests and filters output;
  rules are authoritative and a local MiniLM + linear ML head adds paraphrase
  coverage. Red-team 72/72, output 4/4, agent 72/72, benign false blocks 0/26,
  held-out semantic paraphrases 10/10 (`docs/SAFETY.md`). It must be enabled for
  any shared deployment.
- A 21-task product benchmark covers tools, memory, tasks, retrieval, teacher
  review and gap detection; it is not a general benchmark. A raw-model probe
  without RAG/tools scores only **2/10** on simple general questions
  (`docs/BENCHMARKS.md`).
- Reviews and gated updates; no autonomous weight mutation.
- Adapter versions and rollback (`learning_loop.py activate`).
- No telemetry; memory is local by default.

Release-readiness evidence:

- Prompt-injection/jailbreak checks: **4/4** deterministic
  (`tests/test_prompt_injection.py`,
  `research/results/redteam_injection.json`).
- PII deletion: exact document source and session memory paths are tested
  (`docs/PRIVACY.md`, `tests/test_privacy.py`).
- The opt-in filter remains pattern-based; a third-party host should add
  a larger human-labelled moderation set, languages and a broader red-team for
  their deployment.
- A host that exposes the server must add authentication, TLS, rate limits,
  abuse monitoring, log retention and incident response.

Secret handling:

- The PyPI token pasted into chat must be revoked at
  `https://pypi.org/manage/account/token/`; never reuse it. A fresh token will be
  supplied by the owner only when publishing.
- Keep `.env`, API keys and provider tokens out of git.
- Rotate any credential that has appeared in logs or chat.

### Decisions

- [ ] Revoke the exposed PyPI token before publishing.
- [x] Safety filter and refusal path implemented; hosts may extend it.
- [x] PII deletion path implemented and tested.
- [ ] Third-party host names its own incident-response owner.

## 4. Name, domain and trademark

**Atom Zero** is a working name, not a cleared trademark.

Preliminary checks on 2026-11-21:

- `atomzero.ai`: RDAP returned **404** at the registry RDAP endpoint and DNS did
  not resolve. This suggests it may be available, but it is **not** a
  reservation; confirm with a registrar.
- Candidate handles: `atomzero-ai`, `atom-zero-ai`.
- A web search found "Atom Zero" used in Cosmos governance discussion and as a
  personal/company label; no direct AI-assistant product was found in this
  limited search.
- No professional trademark search has been completed.

**Recommendation:** keep using "Atom Zero (working name)" in development. Do
not use "(R)", do not claim registration, and do not buy branding inventory
until a professional search covers the intended classes (software, AI services,
downloadable models) in the target jurisdictions.

### Decisions needed

- [ ] Approve a professional trademark search budget.
- [ ] Confirm registrar and buy the domain only after the search.
- [ ] Decide whether the package/CLI stays `streamft` (recommended for
      compatibility) while Atom Zero is the product name.
- [ ] Approve final name and handle.

## 5. Update policy

This is the policy the gated loop already implements:

```text
facts/documents -> DocMemory (RAG), never weights
style/skill     -> reviewed feedback -> gated LoRA -> accepted vNNNN
```

Rules:

1. A human reviews feedback; negative feedback needs a corrected answer.
2. At least five approved examples are required for a gated update.
3. The gate must improve target loss, keep clean-chat regression within budget,
   and pass behavior plus clean-smoke checks.
4. A holdout slice should stay out of training when enough rows exist.
5. Accepted adapters are immutable `vNNNN` versions with a history entry and
   queue hash.
6. Rollback is explicit: `learning_loop.py activate vNNNN`.
7. No autonomous weight mutation, no unreviewed crawling, no real-money trading.
8. Release only from a tagged commit with test evidence and a model card that
   matches the exact checkpoint.

Release cadence and support:

- Tag releases; keep the previous adapter/checkpoint available for rollback.
- Adapter updates are opt-in for users.
- Security fixes ship out-of-band.
- Deprecations need a changelog entry and a migration note.
- State a support window and a takedown/contact address before hosting.

### Decisions needed

- [ ] Approve the update policy as written.
- [ ] Choose release cadence and support window.
- [ ] Choose retention for old adapters/checkpoints.
- [ ] Name the release approver.

## 6. Go/no-go

Open-source release is approved. No hosted service is operated. Before the
actual package upload:

- [x] Apache-2.0 `LICENSE` and data-attribution `NOTICE`.
- [x] Safety filter and refusal evidence.
- [x] Prompt-injection and PII deletion evidence.
- [ ] Revoke the exposed PyPI token and provide a fresh one.
- [ ] Release owner sign-off with date and commit hash.
- [ ] Optional per-source data review before commercial redistribution.
- [ ] Third-party hosts add their own auth, rate limits and retention.

Hosting/auth review is **not** a blocker for this repository because it does
not operate a hosted service.
