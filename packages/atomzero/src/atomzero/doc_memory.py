"""Local document memory with SQLite FTS5 and provenance.

Documents are chunked, indexed and retrieved with their source. Facts stay in
this index, not in the model weights.

    python -m atomzero.doc_memory add notes.txt --db data/docs.db
    python -m atomzero.doc_memory --db data/docs.db search "kira"
"""
from __future__ import annotations

import argparse
import array
import hashlib
import json
import math
import re
import sqlite3
import threading
import time
import zlib
from pathlib import Path

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is a declared dependency
    np = None

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
HASHED_CACHE_KEY = "hashed-ngram-v1"
# Minimum cosine for a semantic-only candidate. Lexical FTS hits are always
# kept; this only stops unrelated chunks from being dragged into the pool.
HYBRID_MIN_HASHED = 0.28
HYBRID_MIN_NEURAL = 0.30
ANN_MIN_CHUNKS = 256
ANN_BUILD_ITERATIONS = 8


class _AnnFallback(RuntimeError):
    """Internal signal: use the exact matrix for this query."""


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


def _pack_vector(values) -> bytes:
    return array.array("f", (float(value) for value in values)).tobytes()


def _unpack_vector(blob: bytes):
    values = array.array("f")
    values.frombytes(blob)
    return values


def _normalized_rows(matrix):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def build_ivf(matrix, nlist: int | None = None,
              iterations: int = ANN_BUILD_ITERATIONS) -> dict | None:
    """Deterministic k-means IVF index over normalized rows.

    This is approximate retrieval: a query scores cluster centroids, probes
    the best `nprobe` buckets, then reranks only their members exactly.
    """
    if np is None or matrix is None or len(matrix) < 2:
        return None
    n = len(matrix)
    if nlist is None:
        nlist = min(64, max(2, int(round(math.sqrt(n)))))
    nlist = min(max(2, int(nlist)), n)
    starts = np.linspace(0, n - 1, nlist).astype(int)
    centroids = matrix[starts].copy()
    for _ in range(max(1, iterations)):
        assignments = (matrix @ centroids.T).argmax(axis=1)
        updated = centroids.copy()
        for cluster in range(nlist):
            members = matrix[assignments == cluster]
            if len(members):
                updated[cluster] = members.mean(axis=0)
        centroids = _normalized_rows(updated)
    assignments = (matrix @ centroids.T).argmax(axis=1)
    buckets = [np.where(assignments == cluster)[0].astype(np.int32)
               for cluster in range(nlist)]
    return {"centroids": centroids, "buckets": buckets, "nlist": nlist}


def ann_topk(matrix, ann: dict, query_vector, limit: int,
             probe: int | None = None):
    """Return exact-scored top-k within the best `probe` IVF buckets."""
    if ann is None or matrix is None:
        raise _AnnFallback("no ANN index")
    nlist = int(ann["nlist"])
    if probe is None:
        probe = max(1, int(math.ceil(math.sqrt(nlist))))
    probe = min(nlist, max(1, int(probe)))
    cluster_scores = ann["centroids"] @ query_vector
    clusters = np.argsort(-cluster_scores, kind="stable")[:probe]
    parts = [ann["buckets"][int(cluster)] for cluster in clusters
             if len(ann["buckets"][int(cluster)])]
    if not parts:
        raise _AnnFallback("empty ANN buckets")
    positions = np.concatenate(parts)
    if len(positions) < min(limit, len(matrix)):
        raise _AnnFallback("too few ANN candidates")
    scores = matrix[positions] @ query_vector
    order = np.argsort(-scores, kind="stable")[:limit]
    return positions[order], scores[order]


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
    raw = []
    for hit in hits:
        value = hit.get("score")
        try:
            number = float(value) if value is not None else None
        except (TypeError, ValueError):
            number = None
        raw.append(number if number is not None and math.isfinite(number)
                   else None)
    finite = [value for value in raw if value is not None]
    # bm25 returns lower-is-better; normalize it into 0..1. Embedding-only
    # candidates have no bm25 and get 0 for this component.
    high, low = (max(finite), min(finite)) if finite else (0.0, 0.0)
    span = high - low
    scored = []
    for index, hit in enumerate(hits):
        chunk = str(hit.get("chunk", "")).lower()
        title = str(hit.get("title", "")).lower()
        section = str(hit.get("section", "")).strip().lower()
        searchable = f"{title} {section} {chunk}"
        words = _query_terms(searchable)
        coverage = len(terms & words) / len(terms)
        title_hits = len(terms & _query_terms(f"{title} {section}")) / len(terms)
        section_hits = (len(terms & _query_terms(section)) / len(terms)
                        if section else 0.0)
        exact = 1.0 if len(terms) >= 2 and phrase in searchable else 0.0
        if raw[index] is not None and span:
            bm25 = 1.0 - ((raw[index] - low) / span)
        elif raw[index] is not None:
            bm25 = 0.5
        else:
            bm25 = 0.0
        embedding = hit.get("embedding_score")
        if embedding is None:
            body_vector = embed_text(searchable)
            embedding = sum(x * y for x, y in zip(query_vector, body_vector))
        score = (0.40 * coverage + 0.10 * title_hits
                 + 0.05 * section_hits + 0.10 * exact
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


def format_citation(hit: dict) -> str:
    """Format a hit as ``source (p. N) [section]`` for chat/UI display."""
    location = str(hit.get("source_url") or hit.get("source") or "")
    page = hit.get("page")
    if page not in (None, ""):
        location += f" (p. {page})"
    section = str(hit.get("section") or "").strip()
    if section:
        location += f" [{section}]"
    return location


def add_citation_source(sources: dict, hit: dict) -> dict:
    """Merge a page/section hit into a source-keyed citation map."""
    key = str(hit.get("source", ""))
    source = sources.get(key)
    if source is None:
        source = {"source": hit.get("source", ""),
                  "title": hit.get("title", "")}
        for field in ("kind", "source_url", "teacher", "model", "verified",
                      "reviewer", "verified_at"):
            if field in hit:
                source[field] = hit[field]
        sources[key] = source
    page = hit.get("page")
    if page not in (None, ""):
        pages = source.setdefault("pages", [])
        if page not in pages:
            pages.append(page)
    section = str(hit.get("section") or "").strip()
    if section:
        sections = source.setdefault("sections", [])
        if section not in sections:
            sections.append(section)
    return source


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
        self._vector_index = None
        self.vector_index_mode = "auto"  # auto | ann | exact
        self.ann_min_chunks = ANN_MIN_CHUNKS
        self.ann_probe = None
        self._vector_fallback_count = 0
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
        self.con.execute(
            "CREATE TABLE IF NOT EXISTS doc_vectors("
            "doc_rowid INTEGER NOT NULL, model TEXT NOT NULL, "
            "dim INTEGER NOT NULL, vec BLOB NOT NULL, "
            "PRIMARY KEY(doc_rowid, model))")
        self.con.execute(
            "CREATE TABLE IF NOT EXISTS doc_chunk_meta("
            "doc_rowid INTEGER PRIMARY KEY, page INTEGER, "
            "section TEXT NOT NULL DEFAULT '', "
            "char_start INTEGER, char_end INTEGER)")
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
        return self.add_blocks(
            source, [{"text": text, "page": None, "section": ""}],
            title=title, replace=replace, meta=meta)

    def add_blocks(self, source: str, blocks: list[dict], title: str = "",
                   replace: bool = True, meta: dict | None = None) -> int:
        """Index text blocks while keeping page/section provenance.

        A block is a piece of text with an optional 1-based ``page`` and a
        ``section`` label. Each block is split with the normal chunker; the
        resulting chunks inherit the block's provenance and a character range
        inside that block.
        """
        source = str(source)
        title = str(title or Path(source).stem)
        meta = meta or {}
        kind = str(meta.get("kind", "document"))
        source_url = str(meta.get("source_url", ""))
        teacher = str(meta.get("teacher", ""))
        model = str(meta.get("model", ""))
        verified = 1 if meta.get("verified", True) else 0
        created = str(meta.get("created", int(time.time())))
        reviewer = str(meta.get("reviewer", ""))
        verified_at = str(meta.get("verified_at", ""))
        block_texts = [str(block.get("text", "")) for block in blocks]
        sha = hashlib.sha256(
            "\n\n".join(block_texts).encode("utf-8")).hexdigest()
        with self._lock:
            if replace:
                self.con.execute("DELETE FROM docs WHERE source = ?",
                                 (source,))
            added = 0
            for block in blocks:
                text = str(block.get("text", "")).replace(
                    "\r\n", "\n").replace("\r", "\n").strip()
                if not text:
                    continue
                page = block.get("page")
                try:
                    page = int(page) if page not in (None, "") else None
                except (TypeError, ValueError):
                    page = None
                section = str(block.get("section", "") or "").strip()
                cursor = 0
                for chunk in _chunks(text):
                    position = text.find(chunk, cursor)
                    if position < 0:
                        position = cursor
                    char_start = position
                    char_end = position + len(chunk)
                    inserted = self.con.execute(
                        "INSERT INTO docs(chunk, source, title, sha, ts) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (chunk, source, title, sha, int(time.time())))
                    rowid = inserted.lastrowid
                    if not rowid:
                        rowid = self.con.execute(
                            "SELECT last_insert_rowid()").fetchone()[0]
                    self.con.execute(
                        "INSERT OR REPLACE INTO doc_chunk_meta("
                        "doc_rowid, page, section, char_start, char_end) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (rowid, page, section, char_start, char_end))
                    cursor = char_end
                    added += 1
            self.con.execute(
                "INSERT OR REPLACE INTO doc_meta("
                "source, kind, source_url, teacher, model, verified, created, "
                "reviewer, verified_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (source, kind, source_url, teacher, model, verified, created,
                 reviewer, verified_at))
            if replace:
                self._drop_stale_vectors()
                self._drop_stale_chunk_meta()
            self._vector_index = None
            self.con.commit()
        return added

    def add_file(self, path: str | Path, title: str = "",
                 meta: dict | None = None) -> int:
        from atomzero.doc_ingest import extract_file

        p = Path(path)
        extracted = extract_file(p, title=title)
        return self.add_blocks(
            str(p), extracted["blocks"], title=extracted["title"],
            meta=meta)

    def _drop_stale_vectors(self) -> None:
        self.con.execute(
            "DELETE FROM doc_vectors WHERE doc_rowid NOT IN "
            "(SELECT rowid FROM docs)")

    def _drop_stale_chunk_meta(self) -> None:
        self.con.execute(
            "DELETE FROM doc_chunk_meta WHERE doc_rowid NOT IN "
            "(SELECT rowid FROM docs)")

    @staticmethod
    def _vector_model_key(active) -> str:
        if active is None or not hasattr(active, "embed"):
            return HASHED_CACHE_KEY
        key = getattr(active, "cache_key", "")
        if key:
            return str(key)
        path = getattr(active, "model_path", "")
        if path:
            return f"neural:{Path(path).resolve()}"
        return "neural:custom"

    @staticmethod
    def _embed_texts(active, texts: list[str]) -> list[list[float]]:
        if active is not None and hasattr(active, "embed"):
            vectors = active.embed(texts)
            return [[float(value) for value in row] for row in vectors.tolist()]
        return [embed_text(text) for text in texts]

    def _vectors_for_rows(self, active, rows) -> dict[int, list[float]]:
        """Load cached chunk vectors, computing and storing missing ones."""
        if not rows:
            return {}
        key = self._vector_model_key(active)
        wanted = {rowid: text for rowid, text in rows}
        vectors: dict[int, list[float]] = {}
        ids = list(wanted)
        for start in range(0, len(ids), 400):
            batch = ids[start:start + 400]
            placeholders = ",".join("?" for _ in batch)
            found = self.con.execute(
                "SELECT doc_rowid, vec FROM doc_vectors "
                f"WHERE model = ? AND doc_rowid IN ({placeholders})",
                (key, *batch)).fetchall()
            for rowid, blob in found:
                vectors[rowid] = _unpack_vector(blob)
        missing = [rowid for rowid in ids if rowid not in vectors]
        if missing:
            computed = self._embed_texts(
                active, [wanted[rowid] for rowid in missing])
            self.con.executemany(
                "INSERT OR REPLACE INTO doc_vectors"
                "(doc_rowid, model, dim, vec) VALUES (?, ?, ?, ?)",
                [(rowid, key, len(vector), _pack_vector(vector))
                 for rowid, vector in zip(missing, computed)])
            self.con.commit()
            for rowid, vector in zip(missing, computed):
                vectors[rowid] = vector
        return vectors

    def prewarm_vectors(self, active=None) -> dict:
        """Build the in-memory matrix over all current chunk vectors.

        With no explicit reranker, use the attached one (neural when
        configured, hashed otherwise).
        """
        if active is None:
            active = self.reranker
        started = time.perf_counter()
        with self._lock:
            rows = self.con.execute(
                "SELECT rowid, chunk, source, title FROM docs "
                "ORDER BY rowid").fetchall()
            key = self._vector_model_key(active)
            if not rows:
                self._vector_index = {
                    "key": key, "rows": [], "rowids": [], "matrix": None,
                    "positions": {}, "build_s": 0.0}
                return self.vector_index_stats()
            texts = [(row[0], f"{row[3]} {row[1]}".strip()) for row in rows]
            vectors = self._vectors_for_rows(active, texts)
            kept_rows = []
            kept_vectors = []
            for row in rows:
                vector = vectors.get(row[0])
                if vector is None:
                    continue
                if kept_vectors and len(vector) != len(kept_vectors[0]):
                    continue
                kept_rows.append(row)
                kept_vectors.append(vector)
            matrix = None
            if np is not None and kept_vectors:
                matrix = np.asarray(kept_vectors, dtype=np.float32)
                matrix = _normalized_rows(matrix)
            mode = self._resolved_vector_mode(len(kept_rows))
            ann = None
            ann_build_s = 0.0
            if mode == "ann" and matrix is not None:
                ann_started = time.perf_counter()
                try:
                    ann = build_ivf(matrix)
                except (MemoryError, RuntimeError, ValueError):
                    ann = None
                ann_build_s = time.perf_counter() - ann_started
                if ann is None:
                    mode = "exact"
            rowids = [row[0] for row in kept_rows]
            self._vector_index = {
                "key": key,
                "rows": kept_rows,
                "rowids": rowids,
                "matrix": matrix,
                "positions": {rowid: index
                              for index, rowid in enumerate(rowids)},
                "mode": mode,
                "ann": ann,
                "ann_build_s": ann_build_s,
                "build_s": time.perf_counter() - started,
            }
            self._vector_fallback_count = 0
        return self.vector_index_stats()

    def _resolved_vector_mode(self, chunks: int) -> str:
        if self.vector_index_mode == "exact" or np is None:
            return "exact"
        if self.vector_index_mode == "ann":
            return "ann" if chunks >= 2 else "exact"
        return ("ann" if chunks >= max(2, int(self.ann_min_chunks))
                else "exact")

    def vector_index_stats(self) -> dict:
        index = self._vector_index
        backend = "numpy" if np is not None else "scan"
        if not index:
            return {"built": False, "chunks": 0, "dim": 0,
                    "build_s": 0.0, "backend": backend}
        matrix = index.get("matrix")
        ann = index.get("ann")
        nlist = int(ann["nlist"]) if ann else 0
        return {
            "built": matrix is not None,
            "chunks": len(index.get("rows", [])),
            "dim": int(matrix.shape[1]) if matrix is not None else 0,
            "mode": index.get("mode", "exact"),
            "ann": bool(ann),
            "nlist": nlist,
            "nprobe": (int(self.ann_probe) if self.ann_probe
                       else (max(1, int(math.ceil(math.sqrt(nlist))))
                             if nlist else 0)),
            "exact_fallback": True,
            "fallbacks": int(self._vector_fallback_count),
            "build_s": round(float(index.get("build_s", 0.0)), 3),
            "ann_build_s": round(float(index.get("ann_build_s", 0.0)), 3),
            "backend": backend,
        }

    @staticmethod
    def _candidate_filters(query: str, neural: bool):
        threshold = HYBRID_MIN_NEURAL if neural else HYBRID_MIN_HASHED
        grams = set(_char_ngrams(query)) if not neural else None
        return threshold, grams

    @staticmethod
    def _has_hashed_overlap(query_grams, row) -> bool:
        if query_grams is None:
            return True
        # Hashed vectors are sparse and can collide on short queries. Require
        # real character overlap before a hash-only candidate may bypass FTS.
        text = f"{row[3]} {row[1]}".lower()
        return len(query_grams & set(_char_ngrams(text))) >= 2

    @staticmethod
    def _exact_candidates(index, query_vector, limit: int):
        scores = index["matrix"] @ query_vector
        order = np.argsort(-scores, kind="stable")[:limit]
        return order, scores[order]

    def _search_positions(self, index, query_vector, limit: int,
                          required_rowids, threshold: float):
        matrix = index["matrix"]
        ann = index.get("ann")
        positions = None
        scores = None
        if ann is not None:
            try:
                positions, scores = ann_topk(
                    matrix, ann, query_vector, limit, self.ann_probe)
                if not len(positions) or float(scores[0]) < threshold:
                    raise _AnnFallback("low ANN confidence")
            except Exception:
                positions = None
                scores = None
        if positions is None:
            if ann is not None:
                self._vector_fallback_count += 1
            positions, scores = self._exact_candidates(
                index, query_vector, limit)
        by_rowid = {
            index["rowids"][int(position)]: float(score)
            for position, score in zip(positions, scores)
        }
        required = [index["positions"][rowid]
                    for rowid in required_rowids or ()
                    if rowid in index["positions"]]
        if required:
            extra = matrix[required] @ query_vector
            for position, score in zip(required, extra):
                by_rowid[index["rowids"][position]] = float(score)
        return ([int(position) for position in positions],
                [float(score) for score in scores], by_rowid)

    def _embedding_candidates(self, query: str, active, limit: int,
                              required_rowids=None):
        """Score chunks with the prewarmed matrix; rebuild it when invalid."""
        key = self._vector_model_key(active)
        index = self._vector_index
        if index is None or index.get("key") != key:
            self.prewarm_vectors(active)
            index = self._vector_index
        matrix = index.get("matrix") if index else None
        if matrix is None:
            return self._embedding_candidates_scan(
                query, active, limit, required_rowids)
        if not index["rows"]:
            return [], {}
        query_vector = np.asarray(
            self._embed_texts(active, [query])[0], dtype=np.float32)
        if matrix.shape[1] != query_vector.shape[0]:
            return self._embedding_candidates_scan(
                query, active, limit, required_rowids)
        neural = key.startswith("neural:")
        threshold, query_grams = self._candidate_filters(query, neural)
        try:
            top, top_scores, by_rowid = self._search_positions(
                index, query_vector, limit, required_rowids, threshold)
        except (MemoryError, RuntimeError, ValueError, IndexError):
            return self._embedding_candidates_scan(
                query, active, limit, required_rowids)
        hits = []
        for position, similarity in zip(top, top_scores):
            if similarity < threshold:
                break
            row = index["rows"][position]
            if not self._has_hashed_overlap(query_grams, row):
                continue
            hit = {
                "chunk": row[1], "source": row[2], "title": row[3],
                "score": None, "_rowid": row[0],
                "embedding_score": similarity,
            }
            if neural:
                hit["neural_score"] = similarity
            hits.append(hit)
        return hits, by_rowid

    def _embedding_candidates_scan(self, query: str, active, limit: int,
                                   required_rowids=None):
        """Legacy linear scan; used only when numpy is unavailable."""
        rows = self.con.execute(
            "SELECT rowid, chunk, source, title FROM docs "
            "ORDER BY rowid").fetchall()
        if not rows:
            return [], {}
        texts = [(row[0], f"{row[3]} {row[1]}".strip()) for row in rows]
        vectors = self._vectors_for_rows(active, texts)
        query_vector = self._embed_texts(active, [query])[0]
        neural = self._vector_model_key(active).startswith("neural:")
        scored = []
        for row in rows:
            vector = vectors.get(row[0])
            if vector is None or len(vector) != len(query_vector):
                continue
            similarity = sum(a * b for a, b in
                             zip(query_vector, vector))
            scored.append((similarity, row))
        scored.sort(key=lambda item: (-item[0], item[1][0]))
        by_rowid = {row[0]: similarity for similarity, row in scored}
        threshold, query_grams = self._candidate_filters(query, neural)
        hits = []
        for similarity, row in scored[:limit]:
            if similarity < threshold:
                break
            if not self._has_hashed_overlap(query_grams, row):
                continue
            hit = {
                "chunk": row[1], "source": row[2], "title": row[3],
                "score": None, "_rowid": row[0],
                "embedding_score": similarity,
            }
            if neural:
                hit["neural_score"] = similarity
            hits.append(hit)
        return hits, by_rowid

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

    def _chunk_meta_map(self, rowids) -> dict[int, dict]:
        rowids = [int(rowid) for rowid in rowids
                  if rowid is not None]
        if not rowids:
            return {}
        placeholders = ",".join("?" for _ in rowids)
        rows = self.con.execute(
            "SELECT doc_rowid, page, section, char_start, char_end "
            f"FROM doc_chunk_meta WHERE doc_rowid IN ({placeholders})",
            tuple(rowids)).fetchall()
        return {row[0]: {
            "page": row[1],
            "section": row[2] or "",
            "char_start": row[3],
            "char_end": row[4],
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
        candidates = max(k * 4, 12) if rerank else k
        active = reranker if reranker is not None else self.reranker
        neural = self._vector_model_key(active).startswith("neural:")
        with self._lock:
            rows = self.con.execute(
                "SELECT rowid, chunk, source, title, bm25(docs) AS score "
                "FROM docs WHERE docs MATCH ? ORDER BY score LIMIT ?",
                (match, candidates)).fetchall()
            embedding_hits, embedding_scores = ([], {})
            # Hashed n-grams are a fallback reranker, not a semantic retriever:
            # do not answer from hash-only candidates when FTS found nothing.
            # The neural reranker has a calibrated threshold and may do so.
            if rerank and (rows or neural):
                embedding_hits, embedding_scores = \
                    self._embedding_candidates(
                        query, active, candidates,
                        required_rowids={row[0] for row in rows})
            sources = {row[2] for row in rows} | {
                hit["source"] for hit in embedding_hits}
            metas = self._meta_map(sources)
            chunk_metas = self._chunk_meta_map(
                {row[0] for row in rows}
                | {hit.get("_rowid") for hit in embedding_hits})
        default = {"kind": "document", "source_url": "", "teacher": "",
                   "model": "", "verified": True, "created": "",
                   "reviewer": "", "verified_at": "", "page": None,
                   "section": "", "char_start": None, "char_end": None}
        hits = [{
            "chunk": row[1],
            "source": row[2],
            "title": row[3],
            "score": float(row[4]),
            "_rowid": row[0],
            **default,
            **metas.get(row[2], {}),
            **chunk_metas.get(row[0], {}),
        } for row in rows]
        for hit in embedding_hits:
            hit.update(chunk_metas.get(hit.get("_rowid"), {}))
        if not rerank:
            return hits
        # Hybrid candidate pool: FTS rows plus the best embedding rows. A
        # paraphrase with no lexical overlap with any document can now enter
        # the reranker, which was previously capped by FTS recall.
        merged = []
        seen = set()
        for hit in hits + embedding_hits:
            key = (hit["source"], hit["chunk"])
            if key in seen:
                continue
            seen.add(key)
            merged.append(hit)
        finite = [hit["score"] for hit in hits
                  if hit.get("score") is not None]
        fallback = max(finite) if finite else None
        for hit in merged:
            similarity = embedding_scores.get(hit.get("_rowid"))
            if similarity is not None:
                hit["embedding_score"] = similarity
                if neural:
                    hit["neural_score"] = similarity
            if hit.get("score") is None:
                hit["score"] = fallback
        if active is not None:
            ranked = active.rerank(query, merged, k)
        else:
            ranked = rerank_hits(query, merged, k)
        for hit in ranked:
            hit.pop("_rowid", None)
        return ranked

    def sources(self) -> list[dict]:
        with self._lock:
            rows = self.con.execute(
                "SELECT source, title, COUNT(*), MAX(sha) FROM docs "
                "GROUP BY source, title ORDER BY source").fetchall()
            metas = self._meta_map([row[0] for row in rows])
            page_rows = self.con.execute(
                "SELECT docs.source, MIN(m.page), MAX(m.page) "
                "FROM doc_chunk_meta m JOIN docs ON docs.rowid = m.doc_rowid "
                "GROUP BY docs.source").fetchall()
        pages = {row[0]: (row[1], row[2]) for row in page_rows}
        result = []
        for row in rows:
            item = {
                "source": row[0],
                "title": row[1],
                "chunks": row[2],
                "content_sha256": row[3],
                **metas.get(row[0], {"source": row[0], "kind": "document",
                                     "source_url": "", "teacher": "",
                                     "model": "", "verified": True,
                                     "created": "", "reviewer": "",
                                     "verified_at": ""}),
            }
            page_range = pages.get(row[0])
            if page_range and page_range[0] is not None:
                item["page_start"] = page_range[0]
                item["page_end"] = page_range[1]
            result.append(item)
        return result

    def source_evidence(self, source: str, page: int | None = None,
                        section: str = "", limit: int = 20) -> dict:
        """Return the stored chunks for a source/page/section locator."""
        source = str(source or "").strip()
        if not source:
            raise ValueError("source is required")
        if len(source) > 512:
            raise ValueError("source is too long (max 512 chars)")
        if any(ord(ch) < 32 for ch in source):
            raise ValueError("source contains control characters")
        try:
            limit = int(limit)
        except (TypeError, ValueError) as exc:
            raise ValueError("limit must be an integer") from exc
        limit = max(1, min(limit, 50))
        clauses = ["d.source = ?"]
        params: list = [source]
        if page is not None:
            try:
                page = int(page)
            except (TypeError, ValueError) as exc:
                raise ValueError("page must be an integer") from exc
            if page < 1:
                raise ValueError("page must be >= 1")
            clauses.append("m.page = ?")
            params.append(page)
        section = str(section or "").strip()
        if section:
            if len(section) > 200:
                raise ValueError("section is too long (max 200 chars)")
            clauses.append("lower(m.section) = lower(?)")
            params.append(section)
        with self._lock:
            rows = self.con.execute(
                "SELECT d.chunk, d.source, d.title, m.page, m.section, "
                "m.char_start, m.char_end FROM docs d "
                "LEFT JOIN doc_chunk_meta m ON m.doc_rowid = d.rowid "
                f"WHERE {' AND '.join(clauses)} "
                "ORDER BY COALESCE(m.page, 0), d.rowid LIMIT ?",
                (*params, limit)).fetchall()
        if not rows:
            raise SourceNotFound(
                f"no evidence for source/page/section: {source}")
        data = [{
            "chunk": row[0],
            "source": row[1],
            "title": row[2],
            "page": row[3],
            "section": row[4] or "",
            "char_start": row[5],
            "char_end": row[6],
        } for row in rows]
        return {
            "source": source,
            "page": page,
            "section": section,
            "returned": len(data),
            "data": data,
        }

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
            self._drop_stale_vectors()
            self._drop_stale_chunk_meta()
            self._vector_index = None
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
            self._drop_stale_vectors()
            self._drop_stale_chunk_meta()
            self._vector_index = None
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
