"""Local document memory with SQLite FTS5 and provenance.

Documents are chunked, indexed and retrieved with their source. Facts stay in
this index, not in the model weights.

    python -m streamft.doc_memory add notes.txt --db data/docs.db
    python -m streamft.doc_memory --db data/docs.db search "kira"
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import threading
import time
from pathlib import Path

MAX_CHARS = 900


class SourceNotFound(ValueError):
    pass


STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but",
    "by", "can", "could", "did", "do", "does", "for", "from", "had", "has",
    "have", "he", "her", "here", "him", "his", "how", "i", "if", "in",
    "into", "is", "it", "its", "me", "my", "no", "not", "of", "on", "or",
    "our", "please", "she", "should", "so", "tell", "than", "that", "the",
    "their", "them", "then", "there", "these", "they", "this", "those",
    "to", "was", "we", "were", "what", "when", "where", "which", "who",
    "why", "will", "with", "would", "you", "your",
})


def _split_long(paragraph: str) -> list[str]:
    """Split a long paragraph near a word boundary, never mid-word if possible."""
    out = []
    while len(paragraph) > MAX_CHARS:
        cut = paragraph.rfind(" ", 0, MAX_CHARS)
        if cut < MAX_CHARS // 2:
            cut = MAX_CHARS
        piece = paragraph[:cut].strip()
        if piece:
            out.append(piece)
        paragraph = paragraph[cut:].lstrip()
    if paragraph.strip():
        out.append(paragraph.strip())
    return out


def _chunks(text: str) -> list[str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    out = []
    current = ""
    for para in paragraphs:
        if len(para) > MAX_CHARS:
            if current:
                out.append(current)
                current = ""
            out.extend(_split_long(para))
            continue
        if len(current) + len(para) + 2 <= MAX_CHARS:
            current = (current + "\n\n" + para).strip()
        else:
            if current:
                out.append(current)
            current = para
    if current:
        out.append(current)
    return out


class DocMemory:
    def __init__(self, path: str | Path = "data/docs.db"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.con = sqlite3.connect(str(self.path), check_same_thread=False)
        self.con.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS docs USING fts5("
            "chunk, source UNINDEXED, title UNINDEXED, sha UNINDEXED, "
            "ts UNINDEXED)")
        self.con.execute(
            "CREATE TABLE IF NOT EXISTS doc_meta("
            "source TEXT PRIMARY KEY, kind TEXT NOT NULL, "
            "source_url TEXT NOT NULL, teacher TEXT NOT NULL, "
            "model TEXT NOT NULL, verified INTEGER NOT NULL, "
            "created TEXT NOT NULL, reviewer TEXT NOT NULL DEFAULT '', "
            "verified_at TEXT NOT NULL DEFAULT '')")
        columns = {row[1] for row in self.con.execute(
            "PRAGMA table_info(doc_meta)")}
        for name, ddl in (
                ("reviewer", "TEXT NOT NULL DEFAULT ''"),
                ("verified_at", "TEXT NOT NULL DEFAULT ''")):
            if name not in columns:
                self.con.execute(
                    f"ALTER TABLE doc_meta ADD COLUMN {name} {ddl}")
        self.con.commit()

    def add_text(self, source: str, text: str, title: str = "",
                 replace: bool = True, meta: dict | None = None) -> int:
        sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        meta = meta or {}
        kind = str(meta.get("kind", "document"))
        source_url = str(meta.get("source_url", ""))
        teacher = str(meta.get("teacher", ""))
        model = str(meta.get("model", ""))
        verified = 1 if meta.get("verified", True) else 0
        created = str(meta.get("created", int(time.time())))
        reviewer = str(meta.get("reviewer", ""))
        verified_at = str(meta.get("verified_at", ""))
        with self._lock:
            if replace:
                self.con.execute("DELETE FROM docs WHERE source = ?",
                                 (source,))
            added = 0
            for chunk in _chunks(text):
                self.con.execute(
                    "INSERT INTO docs(chunk, source, title, sha, ts) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (chunk, source, title or Path(source).stem, sha,
                     int(time.time())))
                added += 1
            self.con.execute(
                "INSERT OR REPLACE INTO doc_meta("
                "source, kind, source_url, teacher, model, verified, created, "
                "reviewer, verified_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (source, kind, source_url, teacher, model, verified, created,
                 reviewer, verified_at))
            self.con.commit()
        return added

    def add_file(self, path: str | Path, title: str = "",
                 meta: dict | None = None) -> int:
        p = Path(path)
        text = p.read_text(encoding="utf-8", errors="replace")
        return self.add_text(str(p), text, title or p.stem, meta=meta)

    def _meta_map(self, sources) -> dict[str, dict]:
        sources = list(sources)
        if not sources:
            return {}
        placeholders = ",".join("?" for _ in sources)
        rows = self.con.execute(
            "SELECT source, kind, source_url, teacher, model, verified, "
            f"created, reviewer, verified_at FROM doc_meta "
            f"WHERE source IN ({placeholders})",
            tuple(sources)).fetchall()
        return {row[0]: {
            "source": row[0],
            "kind": row[1],
            "source_url": row[2],
            "teacher": row[3],
            "model": row[4],
            "verified": bool(row[5]),
            "created": row[6],
            "reviewer": row[7],
            "verified_at": row[8],
        } for row in rows}

    def search(self, query: str, k: int = 5) -> list[dict]:
        words = re.findall(r"[\wçğıöşüÇĞİÖŞÜ]+", query)
        words = [w for w in words if len(w) > 1]
        meaningful = [w for w in words if w.lower() not in STOPWORDS]
        if meaningful:
            words = meaningful
        if not words:
            return []
        match = " OR ".join(f'"{w}"' for w in words[:12])
        with self._lock:
            rows = self.con.execute(
                "SELECT chunk, source, title, bm25(docs) AS score "
                "FROM docs WHERE docs MATCH ? ORDER BY score LIMIT ?",
                (match, k)).fetchall()
            metas = self._meta_map({row[1] for row in rows})
        return [{
            "chunk": row[0],
            "source": row[1],
            "title": row[2],
            "score": float(row[3]),
            **metas.get(row[1], {"source": row[1], "kind": "document",
                                 "source_url": "", "teacher": "",
                                 "model": "", "verified": True, "created": "",
                                 "reviewer": "", "verified_at": ""}),
        } for row in rows]

    def sources(self) -> list[dict]:
        with self._lock:
            rows = self.con.execute(
                "SELECT source, title, COUNT(*) FROM docs "
                "GROUP BY source, title ORDER BY source").fetchall()
            metas = self._meta_map([row[0] for row in rows])
        return [{
            "source": row[0],
            "title": row[1],
            "chunks": row[2],
            **metas.get(row[0], {"source": row[0], "kind": "document",
                                 "source_url": "", "teacher": "",
                                 "model": "", "verified": True, "created": "",
                                 "reviewer": "", "verified_at": ""}),
        } for row in rows]

    def verify_source(self, source: str, reviewer: str) -> dict:
        """Mark a teacher note verified with reviewer and timestamp."""
        source = str(source or "").strip()
        reviewer = str(reviewer or "").strip()
        if not source:
            raise ValueError("source is required")
        if not reviewer:
            raise ValueError("reviewer is required")
        if len(reviewer) > 100:
            raise ValueError("reviewer is too long (max 100 chars)")
        if any(ord(ch) < 32 for ch in reviewer):
            raise ValueError("reviewer contains control characters")
        with self._lock:
            row = self.con.execute(
                "SELECT kind FROM doc_meta WHERE source = ?",
                (source,)).fetchone()
            if row is None:
                raise SourceNotFound(f"source not found: {source}")
            if row[0] != "teacher":
                raise ValueError("only teacher notes can be verified")
            verified_at = time.strftime("%Y-%m-%dT%H:%M:%S")
            self.con.execute(
                "UPDATE doc_meta SET verified = 1, reviewer = ?, "
                "verified_at = ? WHERE source = ?",
                (reviewer, verified_at, source))
            self.con.commit()
            meta = self._meta_map([source]).get(source, {})
            return {"source": source, **meta}

    def stats(self) -> dict:
        with self._lock:
            count = self.con.execute(
                "SELECT COUNT(*) FROM docs").fetchone()[0]
        return {"path": str(self.path), "chunks": count,
                "sources": len(self.sources())}

    def close(self):
        with self._lock:
            self.con.close()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    add = sub.add_parser("add")
    add.add_argument("path")
    add.add_argument("--title", default="")
    search = sub.add_parser("search")
    search.add_argument("query")
    search.add_argument("-k", type=int, default=5)
    list_db = sub.add_parser("list")
    stats_db = sub.add_parser("stats")
    # Accept --db on either side of the subcommand; the docs use both forms.
    for parser in (add, search, list_db, stats_db):
        parser.add_argument("--db", dest="db_after", default=None)
    args = ap.parse_args(argv)
    mem = DocMemory(args.db_after or args.db or "data/docs.db")
    if args.cmd == "add":
        print(json.dumps({"added": mem.add_file(args.path, args.title)}))
    elif args.cmd == "search":
        print(json.dumps(mem.search(args.query, args.k), indent=2,
                         ensure_ascii=False))
    elif args.cmd == "list":
        print(json.dumps(mem.sources(), indent=2, ensure_ascii=False))
    else:
        print(json.dumps(mem.stats(), indent=2))
    mem.close()


if __name__ == "__main__":
    main()
