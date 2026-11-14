"""Truth-gated knowledge ledger for the spiking LM.

This is the front door for recursive learning from the web or from people. It
does **not** train on raw text. A document must pass source trust and/or
cross-source corroboration, or be approved by a human, before it can be
exported to the ingest inbox.

    streamft spiking knowledge add --url ... --trust reference
    streamft spiking knowledge verify
    streamft spiking knowledge review --id 3 --approve
    streamft spiking knowledge export --out data/inbox
"""
from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import re
import socket
import sqlite3
import sys
import time
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

TRUST = ("official", "reference", "community", "human", "untrusted", "self")
AUTO_VERIFY = {"official"}
CORROBORATE = {"reference", "community"}

NEGATORS = {
    "not", "no", "never", "none", "neither", "nor", "cannot", "cant",
    "dont", "doesnt", "didnt", "isnt", "arent", "wasnt", "werent",
    "wont", "wouldnt", "shouldnt", "couldnt", "false", "incorrect",
    "değil", "degil", "yok", "asla", "hiç", "hic",
}
STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "of", "in", "on", "at", "to", "for", "with", "as", "by", "from",
    "and", "or", "that", "this", "these", "those", "it", "its", "there",
    "has", "have", "had", "do", "does", "did", "will", "would", "can",
    "could", "should", "may", "might", "must", "about", "approximately",
}


class _Text(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "noscript"):
            self.skip += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style", "noscript") and self.skip:
            self.skip -= 1

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _tokens(text: str):
    return re.findall(r"\w+", text.lower())


def _shingles(text: str, k: int = 5):
    t = _tokens(text)
    return {tuple(t[i:i + k]) for i in range(len(t) - k + 1)}


def similarity(a: str, b: str) -> float:
    x, y = _shingles(a), _shingles(b)
    if not x or not y:
        return 0.0
    return len(x & y) / len(x | y)


def _sentences(text: str):
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]


def _norm_token(t: str) -> str:
    t = t.lower().replace("’", "'").replace("‘", "'")
    t = t.replace("n't", " not ").replace("cannot", "can not")
    t = re.sub(r"[^a-z0-9ğüşöçı]+", "", t)
    if t in NEGATORS or t == "not":
        return "not"
    # Turkish copula suffixes: doğrudur -> doğru, değildir -> değil.
    for suffix in ("dır", "dir", "dur", "dür", "tır", "tir", "tur", "tür"):
        if t.endswith(suffix) and len(t) > len(suffix) + 2:
            return t[:-len(suffix)]
    return t


def claims(text: str):
    out = []
    for sentence in _sentences(text):
        raw = sentence.replace("n't", " not ").replace("cannot", "can not")
        toks = [_norm_token(t) for t in re.findall(r"[A-Za-z0-9ğüşöçıĞÜŞÖÇİ']+", raw)]
        toks = [t for t in toks if t]
        neg_count = sum(1 for t in toks if t == "not" or t in NEGATORS)
        core = [t for t in toks if t not in STOPWORDS and t not in NEGATORS
                and t != "not"]
        nums = set(re.findall(r"\d+(?:\.\d+)?", sentence))
        if len(core) >= 2:
            out.append({"sentence": sentence, "tokens": core,
                        "negated": neg_count % 2 == 1, "numbers": nums})
    return out


def claim_conflict(a_text: str, b_text: str):
    """Return the first high-overlap claim pair with opposite polarity or numbers."""
    for a in claims(a_text):
        sa = set(a["tokens"])
        for b in claims(b_text):
            sb = set(b["tokens"])
            inter = len(sa & sb)
            if inter < 2:
                continue
            union = len(sa | sb)
            sim = inter / union if union else 0.0
            containment = inter / min(len(sa), len(sb))
            if sim < 0.50 and containment < 0.75:
                continue
            if a["negated"] != b["negated"]:
                return {"reason": "negation", "similarity": sim,
                        "sentence_a": a["sentence"],
                        "sentence_b": b["sentence"]}
            if a["numbers"] and b["numbers"] and a["numbers"] != b["numbers"]:
                return {"reason": "number", "similarity": sim,
                        "sentence_a": a["sentence"],
                        "sentence_b": b["sentence"]}
    return None


def _public_url(url: str) -> bool:
    p = urllib.parse.urlparse(url)
    if p.scheme not in ("http", "https") or not p.hostname:
        return False
    try:
        infos = socket.getaddrinfo(p.hostname, None)
    except OSError:
        return False
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if (ip.is_private or ip.is_loopback or ip.is_link_local
                or ip.is_multicast or ip.is_reserved):
            return False
    return True


def fetch_url(url: str, max_bytes: int = 2_000_000) -> str:
    if not _public_url(url):
        raise ValueError("URL resolves to a private/local address or is not http(s)")
    req = urllib.request.Request(url, headers={"User-Agent": "streamft-knowledge/0.1"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        raw = resp.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise ValueError("document is larger than max_bytes")
    text = raw.decode("utf-8", errors="replace")
    parser = _Text()
    parser.feed(text)
    return re.sub(r"\s+", " ", " ".join(parser.parts)).strip()


def connect(path: str) -> sqlite3.Connection:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(p)
    conn.row_factory = sqlite3.Row
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS sources (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      url TEXT NOT NULL, domain TEXT NOT NULL, trust TEXT NOT NULL,
      fetched_at TEXT NOT NULL, sha256 TEXT NOT NULL,
      UNIQUE(domain, sha256)
    );
    CREATE TABLE IF NOT EXISTS documents (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      source_id INTEGER NOT NULL REFERENCES sources(id),
      title TEXT NOT NULL, text TEXT NOT NULL, sha256 TEXT NOT NULL,
      status TEXT NOT NULL DEFAULT 'pending',
      confidence REAL NOT NULL DEFAULT 0.0,
      created_at TEXT NOT NULL,
      UNIQUE(source_id, sha256)
    );
    CREATE TABLE IF NOT EXISTS reviews (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      document_id INTEGER NOT NULL REFERENCES documents(id),
      decision TEXT NOT NULL, note TEXT NOT NULL, reviewer TEXT NOT NULL,
      created_at TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS conflicts (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      doc_a INTEGER NOT NULL, doc_b INTEGER NOT NULL,
      reason TEXT NOT NULL, similarity REAL NOT NULL,
      sentence_a TEXT NOT NULL, sentence_b TEXT NOT NULL,
      created_at TEXT NOT NULL,
      UNIQUE(doc_a,doc_b,sentence_a,sentence_b,reason)
    );
    """)
    return conn


def add_doc(conn, text: str, title: str, source: str, trust: str) -> dict:
    if trust not in TRUST:
        raise ValueError(f"trust must be one of {TRUST}")
    text = text.strip()
    if not text:
        raise ValueError("empty document")
    domain = urllib.parse.urlparse(source).hostname or source
    sha = _hash(text)
    cur = conn.execute(
        "INSERT OR IGNORE INTO sources(url,domain,trust,fetched_at,sha256) "
        "VALUES (?,?,?,?,?)",
        (source, domain, trust, time.strftime("%Y-%m-%dT%H:%M:%S"), sha))
    conn.commit()
    src = conn.execute("SELECT id FROM sources WHERE domain=? AND sha256=?",
                       (domain, sha)).fetchone()
    cur = conn.execute(
        "INSERT OR IGNORE INTO documents(source_id,title,text,sha256,status,"
        "confidence,created_at) VALUES (?,?,?,?,?,0,?)",
        (src["id"], title or source, text, sha, "pending",
         time.strftime("%Y-%m-%dT%H:%M:%S")))
    conn.commit()
    row = conn.execute("SELECT * FROM documents WHERE source_id=? AND sha256=?",
                       (src["id"], sha)).fetchone()
    return dict(row)


def list_docs(conn, status: str = ""):
    q = ("SELECT d.id,d.title,d.status,d.confidence,d.sha256,s.domain,s.trust "
         "FROM documents d JOIN sources s ON s.id=d.source_id")
    if status:
        q += " WHERE d.status=?"
        rows = conn.execute(q + " ORDER BY d.id", (status,)).fetchall()
    else:
        rows = conn.execute(q + " ORDER BY d.id").fetchall()
    return [dict(r) for r in rows]


def make_nli_fn(model_name: str, device: int = -1, threshold: float = 0.80):
    """Optional trained NLI backend; import transformers only when requested."""
    from transformers import pipeline

    pipe = pipeline("text-classification", model=model_name, device=device)

    def check(a_text: str, b_text: str):
        for sa in _sentences(a_text):
            for sb in _sentences(b_text):
                ta = {t for t in (_norm_token(x) for x in re.findall(
                    r"[A-Za-z0-9ğüşöçıĞÜŞÖÇİ']+", sa)) if t}
                tb = {t for t in (_norm_token(x) for x in re.findall(
                    r"[A-Za-z0-9ğüşöçıĞÜŞÖÇİ']+", sb)) if t}
                if len(ta & tb) < 2:
                    continue
                out = pipe({"text": sa, "text_pair": sb},
                           truncation=True)[0]
                label = str(out["label"]).upper()
                score = float(out["score"])
                if "CONTRADICTION" in label and score >= threshold:
                    return {"reason": "nli", "similarity": score,
                            "sentence_a": sa, "sentence_b": sb}
        return None
    return check


def verify(conn, threshold: float = 0.70, min_support: int = 2,
           nli_fn=None) -> list:
    docs = conn.execute(
        "SELECT d.id,d.text,d.status,s.trust,s.domain FROM documents d "
        "JOIN sources s ON s.id=d.source_id WHERE d.status='pending'").fetchall()
    changed = []
    for d in docs:
        if d["trust"] in AUTO_VERIFY:
            conn.execute("UPDATE documents SET status='verified',confidence=1.0 "
                         "WHERE id=?", (d["id"],))
            changed.append({"id": d["id"], "status": "verified",
                            "reason": "official"})
            continue
        if d["trust"] not in CORROBORATE:
            continue
        support, best, first_conflict = 0, 0.0, None
        others = conn.execute(
            "SELECT d.text,d.id,s.domain FROM documents d JOIN sources s ON "
            "s.id=d.source_id WHERE d.id!=?", (d["id"],)).fetchall()
        for o in others:
            if o["domain"] == d["domain"]:
                continue
            sim = similarity(d["text"], o["text"])
            best = max(best, sim)
            # A contradiction can have low document-level shingle overlap
            # ("is true" vs "is not true"), so check claims independently.
            conflict = claim_conflict(d["text"], o["text"])
            if not conflict and nli_fn is not None:
                conflict = nli_fn(d["text"], o["text"])
            if conflict:
                first_conflict = first_conflict or conflict
                conn.execute(
                    "INSERT OR IGNORE INTO conflicts(doc_a,doc_b,reason,similarity,"
                    "sentence_a,sentence_b,created_at) VALUES (?,?,?,?,?,?,?)",
                    (d["id"], o["id"], conflict["reason"], conflict["similarity"],
                     conflict["sentence_a"], conflict["sentence_b"],
                     time.strftime("%Y-%m-%dT%H:%M:%S")))
            elif sim >= threshold:
                support += 1
        if first_conflict:
            conn.execute("UPDATE documents SET status='contradicted',confidence=0 "
                         "WHERE id=?", (d["id"],))
            changed.append({"id": d["id"], "status": "contradicted",
                            "reason": first_conflict["reason"],
                            "similarity": first_conflict["similarity"],
                            "support": support})
        elif support >= min_support:
            conn.execute("UPDATE documents SET status='verified',confidence=? "
                         "WHERE id=?", (best, d["id"]))
            changed.append({"id": d["id"], "status": "verified",
                            "reason": f"support={support}", "similarity": best})
    conn.commit()
    return changed


def list_conflicts(conn):
    rows = conn.execute(
        "SELECT c.*, a.title AS title_a, b.title AS title_b FROM conflicts c "
        "JOIN documents a ON a.id=c.doc_a JOIN documents b ON b.id=c.doc_b "
        "ORDER BY c.id").fetchall()
    return [dict(r) for r in rows]


def review(conn, doc_id: int, decision: str, note: str, reviewer: str) -> dict:
    if decision not in ("approve", "reject"):
        raise ValueError("decision must be approve or reject")
    status = "verified" if decision == "approve" else "rejected"
    conn.execute("UPDATE documents SET status=? WHERE id=?", (status, doc_id))
    conn.execute("INSERT INTO reviews(document_id,decision,note,reviewer,"
                 "created_at) VALUES (?,?,?,?,?)",
                 (doc_id, decision, note, reviewer,
                  time.strftime("%Y-%m-%dT%H:%M:%S")))
    conn.commit()
    return {"id": doc_id, "status": status}


def export(conn, out_dir: str, include_source: bool = False) -> list:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = conn.execute(
        "SELECT d.id,d.text,d.sha256,s.url,s.domain,s.trust FROM documents d "
        "JOIN sources s ON s.id=d.source_id WHERE d.status='verified' "
        "ORDER BY d.id").fetchall()
    written = []
    for r in rows:
        body = r["text"]
        if include_source:
            body = f"Source: {r['url']}\nTrust: {r['trust']}\n\n{body}"
        path = out / f"doc_{r['id']:04d}.txt"
        path.write_text(body + "\n", encoding="utf-8")
        meta = out / f"doc_{r['id']:04d}.json"
        meta.write_text(json.dumps({
            "id": r["id"], "source": r["url"], "domain": r["domain"],
            "trust": r["trust"], "sha256": r["sha256"],
            "status": "verified",
        }, indent=2) + "\n", encoding="utf-8")
        written.append(str(path))
    return written


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="streamft spiking knowledge")
    ap.add_argument("--db", default="data/knowledge.db")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ad = sub.add_parser("add")
    src = ad.add_mutually_exclusive_group(required=True)
    src.add_argument("--text")
    src.add_argument("--file")
    src.add_argument("--url")
    ad.add_argument("--title", default="")
    ad.add_argument("--source", default="", help="provenance URL/label")
    ad.add_argument("--trust", choices=TRUST, default="untrusted")
    ad.add_argument("--max-bytes", type=int, default=2_000_000)

    ls = sub.add_parser("list")
    ls.add_argument("--status", default="")

    ve = sub.add_parser("verify")
    ve.add_argument("--threshold", type=float, default=0.70)
    ve.add_argument("--min-support", type=int, default=2)
    ve.add_argument("--nli-model", default="",
                    help="optional transformers NLI model id")
    ve.add_argument("--nli-threshold", type=float, default=0.80)
    ve.add_argument("--nli-device", type=int, default=-1)

    rv = sub.add_parser("review")
    rv.add_argument("--id", type=int, required=True)
    rv.add_argument("--approve", action="store_true")
    rv.add_argument("--reject", action="store_true")
    rv.add_argument("--note", default="")
    rv.add_argument("--reviewer", default="human")

    ex = sub.add_parser("export")
    ex.add_argument("--out", required=True)
    ex.add_argument("--include-source", action="store_true")

    st = sub.add_parser("stats")
    sub.add_parser("conflicts")

    args = ap.parse_args(argv)
    conn = connect(args.db)
    if args.cmd == "add":
        if args.text is not None:
            text, source = args.text, args.source or "local:text"
        elif args.file:
            text = Path(args.file).read_text(encoding="utf-8")
            source = args.source or str(Path(args.file))
        else:
            text = fetch_url(args.url, args.max_bytes)
            source = args.source or args.url
        row = add_doc(conn, text, args.title, source, args.trust)
        print(json.dumps(row, indent=2))
        return 0
    if args.cmd == "list":
        print(json.dumps(list_docs(conn, args.status), indent=2))
        return 0
    if args.cmd == "verify":
        nli_fn = (make_nli_fn(args.nli_model, args.nli_device,
                              args.nli_threshold)
                  if args.nli_model else None)
        print(json.dumps(verify(conn, args.threshold, args.min_support,
                                nli_fn=nli_fn), indent=2))
        return 0
    if args.cmd == "review":
        if args.approve == args.reject:
            raise SystemExit("choose exactly one of --approve or --reject")
        print(json.dumps(review(conn, args.id,
                                "approve" if args.approve else "reject",
                                args.note, args.reviewer), indent=2))
        return 0
    if args.cmd == "export":
        print(json.dumps(export(conn, args.out, args.include_source), indent=2))
        return 0
    if args.cmd == "stats":
        rows = conn.execute("SELECT status,COUNT(*) n FROM documents GROUP BY status")
        print(json.dumps({r["status"]: r["n"] for r in rows}, indent=2))
        return 0
    if args.cmd == "conflicts":
        print(json.dumps(list_conflicts(conn), indent=2))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
