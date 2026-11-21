# Public release decisions

Status: **not approved**. Atom Zero remains a local research/demo product.
This document is a decision pack, not legal advice. Last updated 2026-11-21.

Use it to make the five release decisions explicitly. Until then, do not publish
model weights, a public API or a hosted endpoint.

## Decision summary

| area | current state | recommendation | owner | status |
|---|---|---|---|---|
| Licence/legal | code is Apache-2.0; weight/data obligations unreviewed | local-only until review | user + lawyer | blocked |
| Hosting | local server on `127.0.0.1` | keep local; optional read-only demo later | user | blocked |
| Security | SSRF guard, upload cap, local bind; no auth/safety filter | fix hosted gaps first | user + engineering | in progress |
| Name | **Atom Zero** working name; preliminary checks only | provisional use; professional clearance | user + lawyer | blocked |
| Update policy | reviewed feedback, gate, holdout, versioning, rollback | publish this policy with any release | user | draft |

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

Weight licence options:

1. **Apache-2.0 for code and weights** - simplest and most permissive, but only
   if every training-data obligation permits redistribution.
2. **Custom open-weights licence** - can add use restrictions, but is not an
   OSI licence and needs legal drafting.
3. **No public weights** - ship code and local setup only; safest near-term.

**Recommendation:** option 3 until the data review and legal review are
complete. If the review clears redistribution, choose option 1 rather than
inventing a custom licence.

### Legal/research use

Atom Zero must present itself as a **source-citing research assistant**, not a
lawyer. Required disclaimer:

> This is not legal advice. Verify every source and consult a qualified
> professional in your jurisdiction before relying on an answer.

Do not claim guaranteed accuracy, confidentiality, privilege or professional
advice. Crypto research remains paper-only: no real money, API keys or orders.

### Decisions needed

- [ ] Approve code licence (recommended: keep Apache-2.0).
- [ ] Approve weight licence (recommended: local-only until review).
- [ ] Complete per-source data review.
- [ ] Approve legal/research disclaimer and jurisdiction wording.
- [ ] Name the release owner and legal reviewer.

## 2. Hosting

| option | cost | risk | fit |
|---|---|---|---|
| Local-only | zero | lowest | current default |
| Hugging Face Space / static demo | low | medium: public inputs, abuse | after safety filter |
| VPS API | monthly | high: auth, PII, uptime, cost | only with an operator |

**Recommendation:** stay local-only for now. A future public demo should be
read-only, rate-limited and use no personal document store. Do not expose the
local server directly to the internet; it has no authentication and is designed
for `127.0.0.1`.

### Decisions needed

- [ ] Local-only, demo-only or hosted API?
- [ ] If hosted: provider, region, budget and data-retention period.
- [ ] Who is the incident/rollback owner?
- [ ] HTTPS, auth, rate limits and cost caps before launch.

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
- A 21-task product benchmark covers tools, memory, tasks, retrieval, teacher
  review and gap detection; it is not a general benchmark. A raw-model probe
  without RAG/tools scores only **2/10** on simple general questions
  (`docs/BENCHMARKS.md`).
- Reviews and gated updates; no autonomous weight mutation.
- Adapter versions and rollback (`learning_loop.py activate`).
- No telemetry; memory is local by default.

Gaps before any hosted release:

- Output safety filter and red-team set for sexual content involving minors,
  self-harm, violence, hate, harassment and illegal instructions.
- Authentication, TLS, rate limits and abuse monitoring.
- Prompt-injection/jailbreak tests with untrusted documents.
- PII deletion path and retention policy.
- Hosted logs that minimise prompt content.
- Dependency pinning and a disclosure address.
- Incident response, takedown and rollback runbook.

Secret handling:

- The PyPI token pasted into chat must be revoked at
  `https://pypi.org/manage/account/token/`; never reuse it.
- Keep `.env`, API keys and provider tokens out of git.
- Rotate any credential that has appeared in logs or chat.

### Decisions needed

- [ ] Revoke the exposed PyPI token.
- [ ] Approve the safety-filter and red-team scope.
- [ ] Approve PII retention/deletion rules.
- [ ] Approve the incident-response owner.

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

Do not release until every item in
[`PUBLIC_RELEASE_CHECKLIST.md`](PUBLIC_RELEASE_CHECKLIST.md) is checked and the
evidence is linked. Blocking items today:

- [ ] Data-licence review and legal review.
- [ ] Safety filter and red-team evidence.
- [ ] Hosted auth/rate-limit/privacy policy (only if hosting).
- [ ] Final name/trademark/domain decision.
- [ ] Release owner sign-off with date and commit hash.
- [ ] Revoked exposed credentials.

Until then: **local research demo only**.
