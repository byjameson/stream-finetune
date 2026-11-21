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
import math
import re
import sqlite3
import threading
import time
import zlib
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


EMBED_DIM = 512


def _char_ngrams(text: str) -> list[str]:
    words = re.findall(r"[^\W_]+", str(text).lower(), re.UNICODE)
    grams = []
    for word in words:
        padded = f" {word} "
        for size in (3, 4, 5):
            grams.extend(padded[index:index + size]
                         for index in range(len(padded) - size + 1))
    return grams


def embed_text(text: str, dim: int = EMBED_DIM) -> list[float]:
    """Deterministic hashed character n-gram embedding (no neural model)."""
    vector = [0.0] * dim
    for gram in _char_ngrams(text):
        vector[zlib.crc32(gram.encode("utf-8")) % dim] += 1.0
    norm = math.sqrt(sum(value * value for value in vector))
    if norm:
        vector = [value / norm for value in vector]
    return vector


def embedding_similarity(left: str, right: str) -> float:
    a, b = embed_text(left), embed_text(right)
    return sum(x * y for x, y in zip(a, b))


def _query_terms(text: str) -> set[str]:
    words = re.findall(r"[^\W_]+", str(text).lower(), re.UNICODE)
    return {word for word in words
            if len(word) > 2 and word not in STOPWORDS}


def rerank_hits(query: str, hits: list[dict], k: int = 5) -> list[dict]:
    """Lexically rerank FTS candidates by coverage, title, phrase and bm25.

    This is a deterministic reranker, not an embedding model. It is intentionally
    cheap: local search should stay fast and work without a second model.
    """
    terms = _query_terms(query)
    if not terms or not hits:
        return hits[:k]
    phrase = " ".join(str(query or "").lower().split())
    query_vector = embed_text(query)
    raw = [float(hit.get("score", 0.0)) for hit in hits]
    # bm25 returns lower-is-better; normalize it into 0..1.
    high, low = max(raw), min(raw)
    span = high - low
    scored = []
    for index, hit in enumerate(hits):
        chunk = str(hit.get("chunk", "")).lower()
        title = str(hit.get("title", "")).lower()
        words = _query_terms(f"{title} {chunk}")
        coverage = len(terms & words) / len(terms)
        title_hits = len(terms & _query_terms(title)) / len(terms)
        exact = 1.0 if len(terms) >= 2 and phrase in f"{title} {chunk}" else 0.0
        bm25 = 1.0 - ((raw[index] - low) / span) if span else 0.5
        body_vector = embed_text(f"{title} {chunk}")
        embedding = sum(x * y for x, y in zip(query_vector, body_vector))
        score = (0.40 * coverage + 0.10 * title_hits + 0.10 * exact
                 + 0.15 * bm25 + 0.25 * embedding)
        item = dict(hit)
        item["rerank_score"] = score
        item["embedding_score"] = embedding
        scored.append((score, index, item))
    scored.sort(key=lambda row: (-row[0], row[1]))
    return [item for _, _, item in scored[:k]]


def best_snippet(query: str, chunk: str, max_chars: int = 500) -> str:
    """Select the one or two sentences that best match the query."""
    text = str(chunk or "").strip()
    if not text:
        return ""
    parts = [part.strip() for part in
             re.split(r"(?<=[.!?])\s+|\n+", text) if part.strip()]
    if len(parts) <= 1:
        return text[:max_chars]
    words = {word for word in re.findall(r"[a-z0-9]+", str(query).lower())
             if len(word) > 2 and word not in STOPWORDS}
    scores = [
        sum(1 for word in re.findall(r"[a-z0-9]+", part.lower())
            if word in words)
        for part in parts
    ]
    chosen = sorted(sorted(range(len(parts)),
                           key=lambda i: (-scores[i], i))[:2])
    snippet = " ".join(parts[i] for i in chosen)
    if len(snippet) > max_chars:
        snippet = parts[min(range(len(parts)),
                            key=lambda i: (-scores[i], i))]
    return snippet[:max_chars]


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
        self.reranker = None
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

    def search(self, query: str, k: int = 5,
               rerank: bool = True,
               reranker=None) -> list[dict]:
        words = re.findall(r"[\wçğıöşüÇĞİÖŞÜ]+", query)
        words = [w for w in words if len(w) > 1]
        meaningful = [w for w in words if w.lower() not in STOPWORDS]
        if meaningful:
            words = meaningful
        if not words:
            return []
        match = " OR ".join(f'"{w}"' for w in words[:12])
        candidates = max(k * 4, 10) if rerank else k
        with self._lock:
            rows = self.con.execute(
                "SELECT chunk, source, title, bm25(docs) AS score "
                "FROM docs WHERE docs MATCH ? ORDER BY score LIMIT ?",
                (match, candidates)).fetchall()
            metas = self._meta_map({row[1] for row in rows})
        hits = [{
            "chunk": row[0],
            "source": row[1],
            "title": row[2],
            "score": float(row[3]),
            **metas.get(row[1], {"source": row[1], "kind": "document",
                                 "source_url": "", "teacher": "",
                                 "model": "", "verified": True, "created": "",
                                 "reviewer": "", "verified_at": ""}),
        } for row in rows]
        if not rerank:
            return hits
        active = reranker if reranker is not None else self.reranker
        if active is not None:
            return active.rerank(query, hits, k)
        return rerank_hits(query, hits, k)

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

    def forget(self, source: str) -> dict:
        """Delete one source and its metadata, then compact the database."""
        source = str(source or "").strip()
        if not source:
            raise ValueError("source is required")
        variants = [source]
        if "://" not in source:
            # Windows stores local paths with backslashes; accept either form
            # on every platform (Path() treats a backslash as a normal char on
            # POSIX, so do the replacement explicitly too).
            for variant in (source.replace("/", "\\"),
                            source.replace("\\", "/"),
                            Path(source).as_posix(), str(Path(source))):
                if variant not in variants:
                    variants.append(variant)
        placeholders = ",".join("?" for _ in variants)
        with self._lock:
            chunks = self.con.execute(
                f"SELECT COUNT(*) FROM docs WHERE source IN ({placeholders})",
                tuple(variants)).fetchone()[0]
            meta = self.con.execute(
                f"SELECT COUNT(*) FROM doc_meta WHERE source IN "
                f"({placeholders})", tuple(variants)).fetchone()[0]
            if chunks == 0 and meta == 0:
                raise SourceNotFound(f"source not found: {source}")
            self.con.execute(
                f"DELETE FROM docs WHERE source IN ({placeholders})",
                tuple(variants))
            self.con.execute(
                f"DELETE FROM doc_meta WHERE source IN ({placeholders})",
                tuple(variants))
            self.con.commit()
            vacuumed = False
            try:
                self.con.execute("VACUUM")
                vacuumed = True
            except sqlite3.OperationalError:
                pass
            return {"source": source, "chunks_deleted": int(chunks),
                    "meta_deleted": int(meta), "vacuumed": vacuumed}

    def forget_all(self) -> dict:
        """Delete every indexed source and compact the database."""
        with self._lock:
            chunks = self.con.execute(
                "SELECT COUNT(*) FROM docs").fetchone()[0]
            meta = self.con.execute(
                "SELECT COUNT(*) FROM doc_meta").fetchone()[0]
            self.con.execute("DELETE FROM docs")
            self.con.execute("DELETE FROM doc_meta")
            self.con.commit()
            vacuumed = False
            try:
                self.con.execute("VACUUM")
                vacuumed = True
            except sqlite3.OperationalError:
                pass
            return {"chunks_deleted": int(chunks),
                    "meta_deleted": int(meta), "vacuumed": vacuumed}

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
    forget = sub.add_parser("forget")
    forget.add_argument("source", nargs="?")
    forget.add_argument("--all", action="store_true")
    # Accept --db on either side of the subcommand; the docs use both forms.
    for parser in (add, search, list_db, stats_db, forget):
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
    elif args.cmd == "forget":
        try:
            result = (mem.forget_all() if args.all
                      else mem.forget(args.source or ""))
        except (ValueError, SourceNotFound) as exc:
            print(json.dumps({"error": str(exc)}))
            mem.close()
            return 1
        print(json.dumps(result, ensure_ascii=False))
    else:
        print(json.dumps(mem.stats(), indent=2))
    mem.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
