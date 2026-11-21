"""Local document memory with SQLite FTS5 and provenance.

Documents are chunked, indexed and retrieved with their source. Facts stay in
this index, not in the model weights.

    python -m streamft.doc_memory add notes.txt --db data/docs.db
    python -m streamft.doc_memory search "kira" --db data/docs.db
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
            for i in range(0, len(para), MAX_CHARS):
                out.append(para[i:i + MAX_CHARS])
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
        self.con.commit()

    def add_text(self, source: str, text: str, title: str = "") -> int:
        sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        with self._lock:
            self.con.execute("DELETE FROM docs WHERE source = ?", (source,))
            added = 0
            for chunk in _chunks(text):
                self.con.execute(
                    "INSERT INTO docs(chunk, source, title, sha, ts) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (chunk, source, title or Path(source).stem, sha,
                     int(time.time())))
                added += 1
            self.con.commit()
        return added

    def add_file(self, path: str | Path, title: str = "") -> int:
        p = Path(path)
        text = p.read_text(encoding="utf-8", errors="replace")
        return self.add_text(str(p), text, title or p.stem)

    def search(self, query: str, k: int = 5) -> list[dict]:
        words = re.findall(r"[\wçğıöşüÇĞİÖŞÜ]+", query)
        words = [w for w in words if len(w) > 1]
        if not words:
            return []
        match = " OR ".join(f'"{w}"' for w in words[:12])
        with self._lock:
            rows = self.con.execute(
                "SELECT chunk, source, title, bm25(docs) AS score "
                "FROM docs WHERE docs MATCH ? ORDER BY score LIMIT ?",
                (match, k)).fetchall()
        return [{"chunk": r[0], "source": r[1], "title": r[2],
                 "score": float(r[3])} for r in rows]

    def sources(self) -> list[dict]:
        with self._lock:
            rows = self.con.execute(
                "SELECT source, title, COUNT(*) FROM docs "
                "GROUP BY source, title ORDER BY source").fetchall()
        return [{"source": r[0], "title": r[1], "chunks": r[2]}
                for r in rows]

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
    ap.add_argument("--db", default="data/docs.db")
    sub = ap.add_subparsers(dest="cmd", required=True)
    add = sub.add_parser("add")
    add.add_argument("path")
    add.add_argument("--title", default="")
    search = sub.add_parser("search")
    search.add_argument("query")
    search.add_argument("-k", type=int, default=5)
    sub.add_parser("list")
    sub.add_parser("stats")
    args = ap.parse_args(argv)
    mem = DocMemory(args.db)
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
