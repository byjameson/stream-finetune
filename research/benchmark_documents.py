"""Authored 100-question document benchmark for Atom Zero.

This is a regression benchmark, not a public leaderboard. It exercises the
local ingestion, retrieval and page/section citation paths on a deterministic
corpus generated in a temporary directory:

* 10 documents: Markdown, PDF, DOCX and HTML
* 80 answerable questions
* 20 questions with no local source

It reports top-1 source accuracy, page/section locator accuracy, answer-string
hit rate, no-answer refusal rate and retrieval latency. ``--check`` exits
non-zero when a threshold regresses.

    python research/benchmark_documents.py --out research/results/document_benchmark.json --check
"""
from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
from pathlib import Path

from atomzero.doc_memory import DocMemory

BENCHMARK = "atomzero-document-v1"

THRESHOLDS = {
    "top1_source": 0.95,
    "top1_locator": 0.90,
    "top1_answer": 0.90,
    "citation_complete": 0.90,
    "no_answer_refusal": 0.95,
    "latency_median_ms": 100.0,
}

DOCS = [
    {
        "name": "launch_policy.md",
        "title": "Orion Launch Policy",
        "format": "md",
        "sections": [
            ("Overview",
             "The Orion launch program is owned by the Helios operations "
             "team. The mission window opens in March. The program office is "
             "in Oslo."),
            ("Launch Code",
             "The launch code for the Orion mission is L-7781. The code "
             "rotates every 24 hours. Only the flight director may read it "
             "aloud."),
            ("Schedule",
             "The primary launch window opens at 06:30 UTC. The backup "
             "window opens at 07:15 UTC. The range opens 90 minutes before "
             "the primary window."),
            ("Rollback",
             "If the range is not clear, the launch is rolled back to the "
             "next window. The rollback owner is the flight director. A "
             "rollback requires a written check."),
        ],
        "questions": [
            ("Which code authorizes the Orion launch?", "Launch Code", "L-7781"),
            ("How often does the Orion launch code rotate?", "Launch Code",
             "every 24 hours"),
            ("When does the primary Orion launch window open?", "Schedule",
             "06:30 UTC"),
            ("What time is the Orion backup window?", "Schedule", "07:15 UTC"),
            ("Which team owns the Orion launch program?", "Overview",
             "Helios operations team"),
            ("When does the Orion mission window open?", "Overview", "March"),
            ("Who owns an Orion launch rollback?", "Rollback",
             "flight director"),
            ("What happens if the launch range is not clear?", "Rollback",
             "rolled back"),
        ],
    },
    {
        "name": "warranty_manual.pdf",
        "title": "Aegis Warranty Manual",
        "format": "pdf",
        "sections": [
            ("Cover",
             "The Aegis warranty manual covers the Aegis 300 device. The "
             "document ID is WM-300. The manual is issued by the Aegis support "
             "group."),
            ("Warranty Period",
             "The Aegis 300 warranty period is 24 months from the invoice "
             "date. Batteries are covered for 12 months. The warranty is "
             "transferable once."),
            ("Claim Process",
             "Submit a claim through the Aegis portal. The claim code prefix "
             "is AC-. The response target is 5 business days."),
            ("Exclusions",
             "Water damage and unauthorized repair are excluded. The exclusion "
             "review takes 10 business days. Physical impact damage is also "
             "excluded."),
        ],
        "questions": [
            ("How many months is the Aegis 300 warranty?", 2, "24 months"),
            ("How long are Aegis batteries covered?", 2, "12 months"),
            ("What is the Aegis claim code prefix?", 3, "AC-"),
            ("What is the Aegis claim response target?", 3,
             "5 business days"),
            ("Which damage is excluded from the Aegis warranty?", 4,
             "water damage"),
            ("How long does an Aegis exclusion review take?", 4,
             "10 business days"),
            ("Which device does the Aegis warranty manual cover?", 1,
             "Aegis 300"),
            ("What is the Aegis warranty document ID?", 1, "WM-300"),
        ],
    },
    {
        "name": "travel_policy.docx",
        "title": "Atlas Travel Policy",
        "format": "docx",
        "sections": [
            ("Booking",
             "Book travel through the Atlas portal. The booking code prefix "
             "is TR-. Approval is required 7 days before departure."),
            ("Baggage",
             "Economy baggage allowance is 23 kg. Business allowance is 32 "
             "kg. Excess baggage needs a receipt."),
            ("Reimbursement",
             "Submit receipts within 30 days. The reimbursement limit for a "
             "hotel night is 180 EUR. Taxi fares are capped at 40 EUR."),
        ],
        "questions": [
            ("Which portal is used to book travel?", "Booking", "Atlas portal"),
            ("What is the travel booking code prefix?", "Booking", "TR-"),
            ("When is travel approval required?", "Booking",
             "7 days before departure"),
            ("What is the economy baggage allowance?", "Baggage", "23 kg"),
            ("What is the business baggage allowance?", "Baggage", "32 kg"),
            ("When must travel receipts be submitted?", "Reimbursement",
             "30 days"),
            ("What is the hotel reimbursement limit?", "Reimbursement",
             "180 EUR"),
            ("Which travel class has a 32 kg baggage allowance?", "Baggage",
             "Business"),
        ],
    },
    {
        "name": "visitor_security.html",
        "title": "Visitor Security",
        "format": "html",
        "sections": [
            ("Badge",
             "All visitors receive a blue badge at the north reception. The "
             "badge code is VB-42. Badges must be visible at all times."),
            ("Escort",
             "Visitors must be escorted by a host at all times. The host must "
             "display a green lanyard. Unescorted visitors are stopped."),
            ("After Hours",
             "After 20:00, visitors must sign the night log and call "
             "extension 4411. The night entrance is on the east side."),
        ],
        "questions": [
            ("What color is the visitor badge?", "Badge", "blue"),
            ("What is the visitor badge code?", "Badge", "VB-42"),
            ("Where do visitors receive badges?", "Badge", "north reception"),
            ("Who must escort visitors?", "Escort", "host"),
            ("What color lanyard must the host display?", "Escort", "green"),
            ("What time do after-hours visitor rules start?", "After Hours",
             "20:00"),
            ("What extension do after-hours visitors call?", "After Hours",
             "4411"),
            ("What must after-hours visitors sign?", "After Hours", "night log"),
        ],
    },
    {
        "name": "support_handbook.md",
        "title": "Support Handbook",
        "format": "md",
        "sections": [
            ("Severity",
             "Severity 1 incidents affect all customers. Severity 2 affects a "
             "single region. Severity 3 is a minor issue."),
            ("Response Times",
             "P1 response target is 15 minutes. P2 response target is 1 hour. "
             "P3 response target is 8 hours."),
            ("Escalation",
             "Escalate P1 incidents to the on-call lead after 30 minutes. Use "
             "the escalation code ESC-9. P2 incidents escalate after 4 hours."),
        ],
        "questions": [
            ("Which severity affects all customers?", "Severity", "Severity 1"),
            ("Which severity affects a single region?", "Severity", "Severity 2"),
            ("What is the P1 response target?", "Response Times", "15 minutes"),
            ("What is the P2 response target?", "Response Times", "1 hour"),
            ("What is the P3 response target?", "Response Times", "8 hours"),
            ("Which P1 incidents are escalated to the on-call lead?",
             "Escalation", "on-call lead"),
            ("Who receives a P1 escalation?", "Escalation", "on-call lead"),
            ("What is the support escalation code?", "Escalation", "ESC-9"),
        ],
    },
    {
        "name": "data_retention.pdf",
        "title": "Data Retention Standard",
        "format": "pdf",
        "sections": [
            ("Scope",
             "The retention standard DR-12 covers customer tickets and audit "
             "logs. It does not cover marketing email. The standard owner is "
             "the data governance team."),
            ("Retention",
             "Customer tickets are retained for 18 months. Audit logs are "
             "retained for 36 months. Backups follow the same schedule."),
            ("Deletion",
             "Deletion requests are processed within 14 days. The deletion "
             "confirmation code is DEL-7. A deletion request needs a ticket."),
        ],
        "questions": [
            ("What does retention standard DR-12 cover?", 1, "customer tickets"),
            ("Does DR-12 cover marketing email?", 1, "does not cover"),
            ("How long are customer tickets retained?", 2, "18 months"),
            ("How long are audit logs retained?", 2, "36 months"),
            ("When are deletion requests processed?", 3, "14 days"),
            ("What is the deletion confirmation code?", 3, "DEL-7"),
            ("Which log type has a 36-month retention?", 2, "audit logs"),
            ("What is the retention standard ID?", 1, "DR-12"),
        ],
    },
    {
        "name": "vendor_contract.docx",
        "title": "Vendor Contract",
        "format": "docx",
        "sections": [
            ("Payment",
             "Invoices are paid within 45 days. The purchase order prefix is "
             "PO-. Payments require a signed receipt."),
            ("Penalties",
             "Late delivery penalty is 2 percent per week. The maximum penalty "
             "is 10 percent. Penalties are deducted from the invoice."),
            ("Termination",
             "Either party may terminate with 60 days written notice. The "
             "termination code is TM-3. Termination does not waive unpaid "
             "invoices."),
        ],
        "questions": [
            ("When are vendor invoices paid?", "Payment", "45 days"),
            ("What is the purchase order prefix?", "Payment", "PO-"),
            ("What is the late delivery penalty?", "Penalties",
             "2 percent per week"),
            ("What is the maximum late penalty?", "Penalties", "10 percent"),
            ("How much notice is required to terminate the vendor contract?",
             "Termination", "60 days"),
            ("What is the vendor termination code?", "Termination", "TM-3"),
            ("Who can terminate the vendor contract?", "Termination",
             "either party"),
            ("What happens to late deliveries?", "Penalties",
             "2 percent per week"),
        ],
    },
    {
        "name": "incident_runbook.html",
        "title": "Incident Runbook",
        "format": "html",
        "sections": [
            ("Detect",
             "The monitoring alert code is INC-5. A failed health check "
             "triggers detection. The alert goes to the on-call channel."),
            ("Contain",
             "Isolate the affected node and rotate the API key. The "
             "containment owner is the incident commander. Do not delete "
             "logs."),
            ("Recover",
             "Restore from the last verified backup and run the smoke test. "
             "The recovery target is 30 minutes. Record the recovery time."),
        ],
        "questions": [
            ("What is the monitoring alert code?", "Detect", "INC-5"),
            ("What triggers incident detection?", "Detect",
             "failed health check"),
            ("How do you contain an incident?", "Contain",
             "isolate the affected node"),
            ("Who owns incident containment?", "Contain",
             "incident commander"),
            ("What is restored from the last verified backup?", "Recover",
             "last verified backup"),
            ("What test runs after recovery?", "Recover", "smoke test"),
            ("What is the incident recovery target?", "Recover", "30 minutes"),
            ("What must be rotated during containment?", "Contain", "API key"),
        ],
    },
    {
        "name": "product_faq.md",
        "title": "Atom Zero Product FAQ",
        "format": "md",
        "sections": [
            ("Installation",
             "Install Atom Zero with pip install streamft. The default port "
             "is 8013. The quickstart seeds a demo corpus."),
            ("Offline Mode",
             "Offline mode uses the local 49M model. The provider is disabled "
             "with --no-provider. RAG and tools stay local."),
            ("Limits",
             "The core context window is 128 tokens. The upload limit is 10 "
             "MB. PDF ingestion keeps page numbers."),
        ],
        "questions": [
            ("How is Atom Zero installed?", "Installation",
             "pip install streamft"),
            ("What is the default Atom Zero port?", "Installation", "8013"),
            ("What model does offline mode use?", "Offline Mode", "49M"),
            ("How is the provider disabled?", "Offline Mode", "--no-provider"),
            ("What is the core context window?", "Limits", "128 tokens"),
            ("What is the document upload limit?", "Limits", "10 MB"),
            ("Does offline mode need a provider?", "Offline Mode", "disabled"),
            ("Which command installs streamft?", "Installation",
             "pip install streamft"),
        ],
    },
    {
        "name": "security_policy.pdf",
        "title": "Security Policy",
        "format": "pdf",
        "sections": [
            ("Access",
             "Production access requires hardware keys. The access review "
             "code is AR-6. Access is reviewed every quarter."),
            ("Encryption",
             "Data at rest uses AES-256. Data in transit uses TLS 1.3. Keys "
             "rotate every 90 days."),
            ("Reporting",
             "Report security issues to security@example.com. The incident "
             "code is SEC-11. Reports are acknowledged within 1 business "
             "day."),
        ],
        "questions": [
            ("What is required for production access?", 1, "hardware keys"),
            ("What is the access review code?", 1, "AR-6"),
            ("What encryption is used at rest?", 2, "AES-256"),
            ("What encryption is used in transit?", 2, "TLS 1.3"),
            ("Where should security issues be reported?", 3,
             "security@example.com"),
            ("What is the security incident code?", 3, "SEC-11"),
            ("How often is production access reviewed?", 1, "every quarter"),
            ("What protocol protects data in transit?", 2, "TLS 1.3"),
        ],
    },
]

NO_ANSWER = [
    "What is the population of Paris?",
    "Who painted the Mona Lisa?",
    "What is the boiling point of mercury?",
    "What is the orbital radius of Titan?",
    "How many moons does Neptune have?",
    "Who wrote Pride and Prejudice?",
    "What is the tallest mountain in Chile?",
    "What is the capital of Mongolia?",
    "How deep is the Mariana Trench?",
    "What is the lifespan of a Galapagos tortoise?",
    "Who discovered penicillin?",
    "What is the speed of light in glass?",
    "How many bones are in a human hand?",
    "What is the largest desert in Asia?",
    "When did the Byzantine Empire fall?",
    "What is the chemical symbol for tungsten?",
    "How many chambers does a crocodile heart have?",
    "What is the national bird of Peru?",
    "Who composed The Four Seasons?",
    "What is the freezing point of ethanol?",
]


def _pdf_escape(text: str) -> str:
    return (text.replace("\\", "\\\\").replace("(", "\\(")
            .replace(")", "\\)").replace("\n", " "))


def pdf_bytes(pages: list[str]) -> bytes:
    """Build a small multi-page PDF with extractable text."""
    bodies: list[bytes | None] = [None, None, None]
    page_refs = []
    for text in pages:
        page_number = len(bodies) + 1
        content_number = page_number + 1
        page_refs.append(page_number)
        bodies.append(
            (f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
             f"/Resources << /Font << /F1 3 0 R >> >> "
             f"/Contents {content_number} 0 R >>").encode("ascii"))
        stream = (f"BT /F1 11 Tf 60 750 Td "
                  f"({_pdf_escape(text)}) Tj ET").encode("latin-1")
        bodies.append(
            b"<< /Length " + str(len(stream)).encode("ascii")
            + b" >>\nstream\n" + stream + b"\nendstream")
    kids = " ".join(f"{number} 0 R" for number in page_refs)
    bodies[0] = b"<< /Type /Catalog /Pages 2 0 R >>"
    bodies[1] = (f"<< /Type /Pages /Kids [{kids}] /Count "
                 f"{len(page_refs)} >>").encode("ascii")
    bodies[2] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"

    out = bytearray(b"%PDF-1.4\n")
    offsets = [0] * (len(bodies) + 1)
    for number, body in enumerate(bodies, start=1):
        offsets[number] = len(out)
        out += f"{number} 0 obj\n".encode("ascii")
        out += body or b""
        out += b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(bodies) + 1}\n".encode("ascii")
    out += b"0000000000 65535 f \n"
    for number in range(1, len(bodies) + 1):
        out += f"{offsets[number]:010d} 00000 n \n".encode("ascii")
    out += (f"trailer\n<< /Size {len(bodies) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref}\n%%EOF\n").encode("ascii")
    return bytes(out)


def build_corpus(root: Path) -> list[Path]:
    paths = []
    for doc in DOCS:
        path = root / doc["name"]
        fmt = doc["format"]
        if fmt == "md":
            body = f"# {doc['title']}\n\n"
            body += "\n\n".join(
                f"## {section}\n\n{text}"
                for section, text in doc["sections"])
            path.write_text(body, encoding="utf-8")
        elif fmt == "html":
            body = (f"<html><head><title>{doc['title']}</title></head>"
                    "<body>")
            body += "".join(
                f"<h2>{section}</h2><p>{text}</p>"
                for section, text in doc["sections"])
            body += "</body></html>"
            path.write_text(body, encoding="utf-8")
        elif fmt == "docx":
            from docx import Document
            document = Document()
            document.core_properties.title = doc["title"]
            for section, text in doc["sections"]:
                document.add_heading(section, level=1)
                document.add_paragraph(text)
            document.save(path)
        elif fmt == "pdf":
            pages = [f"{section}. {text}"
                     for section, text in doc["sections"]]
            path.write_bytes(pdf_bytes(pages))
        else:  # pragma: no cover - guarded by the corpus definition
            raise ValueError(f"unknown format: {fmt}")
        paths.append(path)
    return paths


def build_questions() -> list[dict]:
    questions = []
    for doc in DOCS:
        for index, (query, locator, answer) in enumerate(
                doc["questions"], start=1):
            questions.append({
                "id": f"{doc['name']}:{index}",
                "kind": "answerable",
                "query": query,
                "source": doc["name"],
                "locator": locator,
                "answer": answer,
            })
    for index, query in enumerate(NO_ANSWER, start=1):
        questions.append({
            "id": f"no-answer:{index}",
            "kind": "no_answer",
            "query": query,
        })
    return questions


def _locator_matches(hit: dict, expected) -> bool:
    if isinstance(expected, int):
        return hit.get("page") == expected
    return (str(hit.get("section", "")).strip().lower()
            == str(expected).strip().lower())


def _rate(correct: int, total: int) -> float:
    return round(correct / total, 4) if total else 0.0


def run_benchmark() -> dict:
    questions = build_questions()
    answerable = [q for q in questions if q["kind"] == "answerable"]
    no_answer = [q for q in questions if q["kind"] == "no_answer"]
    source_hits = 0
    locator_hits = 0
    answer_hits = 0
    citation_hits = 0
    refusals = 0
    latencies = []
    failures = []
    by_format = {}

    with tempfile.TemporaryDirectory(prefix="atomzero-doc-bench-") as tmp:
        root = Path(tmp)
        paths = build_corpus(root)
        docs = DocMemory(root / "docs.db")
        try:
            for path in paths:
                docs.add_file(path)
            for question in questions:
                started = time.perf_counter()
                hits = docs.search(question["query"], k=3)
                latencies.append((time.perf_counter() - started) * 1000.0)
                if question["kind"] == "no_answer":
                    if not hits:
                        refusals += 1
                    else:
                        failures.append({
                            "id": question["id"],
                            "reason": "false_positive",
                            "top1": hits[0].get("source"),
                        })
                    continue
                fmt = Path(question["source"]).suffix.lstrip(".")
                bucket = by_format.setdefault(
                    fmt, {"total": 0, "source": 0, "locator": 0,
                          "answer": 0})
                bucket["total"] += 1
                if not hits:
                    failures.append({
                        "id": question["id"], "reason": "no_hit"})
                    continue
                top = hits[0]
                source_ok = (Path(str(top.get("source", ""))).name
                             == question["source"])
                locator_ok = _locator_matches(top, question["locator"])
                answer_ok = (str(question["answer"]).lower()
                             in str(top.get("chunk", "")).lower())
                if source_ok:
                    source_hits += 1
                    bucket["source"] += 1
                if locator_ok:
                    locator_hits += 1
                    bucket["locator"] += 1
                if answer_ok:
                    answer_hits += 1
                    bucket["answer"] += 1
                if source_ok and locator_ok and answer_ok:
                    citation_hits += 1
                else:
                    failures.append({
                        "id": question["id"],
                        "reason": "citation_mismatch",
                        "expected_source": question["source"],
                        "expected_locator": question["locator"],
                        "top1_source": Path(str(top.get("source", ""))).name,
                        "top1_page": top.get("page"),
                        "top1_section": top.get("section"),
                    })
        finally:
            docs.close()

    latencies.sort()
    p95_index = min(len(latencies) - 1,
                    max(0, int(round(0.95 * (len(latencies) - 1)))))
    metrics = {
        "top1_source": _rate(source_hits, len(answerable)),
        "top1_locator": _rate(locator_hits, len(answerable)),
        "top1_answer": _rate(answer_hits, len(answerable)),
        "citation_complete": _rate(citation_hits, len(answerable)),
        "no_answer_refusal": _rate(refusals, len(no_answer)),
        "latency_median_ms": round(statistics.median(latencies), 3),
        "latency_p95_ms": round(latencies[p95_index], 3),
    }
    checks = {
        "top1_source": metrics["top1_source"] >= THRESHOLDS["top1_source"],
        "top1_locator": metrics["top1_locator"] >= THRESHOLDS["top1_locator"],
        "top1_answer": metrics["top1_answer"] >= THRESHOLDS["top1_answer"],
        "citation_complete": (metrics["citation_complete"]
                               >= THRESHOLDS["citation_complete"]),
        "no_answer_refusal": (metrics["no_answer_refusal"]
                               >= THRESHOLDS["no_answer_refusal"]),
        "latency_median_ms": (metrics["latency_median_ms"]
                               <= THRESHOLDS["latency_median_ms"]),
    }
    for bucket in by_format.values():
        bucket["source"] = _rate(bucket["source"], bucket["total"])
        bucket["locator"] = _rate(bucket["locator"], bucket["total"])
        bucket["answer"] = _rate(bucket["answer"], bucket["total"])
    return {
        "benchmark": BENCHMARK,
        "documents": len(DOCS),
        "questions": len(questions),
        "answerable": {
            "total": len(answerable),
            "top1_source": source_hits,
            "top1_locator": locator_hits,
            "top1_answer": answer_hits,
            "citation_complete": citation_hits,
        },
        "no_answer": {
            "total": len(no_answer),
            "refused": refusals,
            "false_positives": len(no_answer) - refusals,
        },
        "metrics": metrics,
        "by_format": by_format,
        "thresholds": THRESHOLDS,
        "checks": checks,
        "passed": all(checks.values()),
        "failures": failures,
        "notes": [
            "Authored regression corpus, not a public leaderboard.",
            "Answerable questions use one expected source and locator.",
            "No-answer questions use topics absent from the corpus.",
        ],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    report = run_benchmark()
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    if args.check and not report["passed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
