"""Local instruction assistant with verified-knowledge retrieval.

This is the product-facing path: a pretrained Qwen2.5-Instruct model plus a
read-only retrieval index over the verified knowledge ledger. It is deliberately
not trained on raw web text and not a from-scratch TinyStories LM.

    streamft assistant ask --model models/Qwen2.5-1.5B-Instruct \
      --db data/knowledge.db --question "What is the speed of light?"
"""
from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


def _tokens(query: str):
    return [t for t in re.findall(r"[A-Za-z0-9ğüşöçıĞÜŞÖÇİ]+", query)
            if len(t) > 1]


class KnowledgeIndex:
    """FTS5 index over verified documents only."""

    def __init__(self, db: str | Path):
        self.db = Path(db)
        self.conn = sqlite3.connect(str(self.db), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_fts "
            "USING fts5(doc_id UNINDEXED, title, text, source UNINDEXED)")
        self.conn.commit()

    def refresh(self) -> int:
        self.conn.execute("DELETE FROM knowledge_fts")
        rows = self.conn.execute(
            "SELECT d.id,d.title,d.text,s.url,s.domain FROM documents d "
            "JOIN sources s ON s.id=d.source_id WHERE d.status='verified'"
        ).fetchall()
        for r in rows:
            self.conn.execute(
                "INSERT INTO knowledge_fts(doc_id,title,text,source) "
                "VALUES (?,?,?,?)",
                (r["id"], r["title"], r["text"], r["url"]))
        self.conn.commit()
        return len(rows)

    def search(self, query: str, k: int = 4):
        terms = _tokens(query)
        if not terms:
            return []
        expr = " OR ".join(f'"{t}"' for t in terms[:12])
        try:
            rows = self.conn.execute(
                "SELECT doc_id,title,text,source,bm25(knowledge_fts) AS rank "
                "FROM knowledge_fts WHERE knowledge_fts MATCH ? "
                "ORDER BY rank LIMIT ?", (expr, k)).fetchall()
        except sqlite3.OperationalError:
            rows = []
        if not rows:
            like = "%" + terms[0] + "%"
            rows = self.conn.execute(
                "SELECT doc_id,title,text,source,0 AS rank FROM knowledge_fts "
                "WHERE text LIKE ? LIMIT ?", (like, k)).fetchall()
        return [dict(r) for r in rows]


def build_messages(question: str, hits: list[dict], include_knowledge=True):
    system = (
        "You are a local assistant. Answer the user's question clearly. "
        "When verified context is provided, use it and cite sources as [1], [2]. "
        "If the context does not contain the answer, say that you do not know "
        "instead of inventing a fact."
    )
    if not include_knowledge or not hits:
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ]
    blocks = []
    for i, h in enumerate(hits, 1):
        blocks.append(f"[{i}] {h.get('title', '')} ({h.get('source', '')})\n"
                      f"{h.get('text', '')}")
    context = "\n\n".join(blocks)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content":
         f"Verified context:\n{context}\n\nQuestion: {question}"},
    ]


@dataclass
class AssistantConfig:
    model: str
    db: str
    max_new_tokens: int = 192
    temperature: float = 0.0
    max_context: int = 4


class LocalAssistant:
    def __init__(self, cfg: AssistantConfig):
        self.cfg = cfg
        self.index = KnowledgeIndex(cfg.db)
        self.index.refresh()
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(cfg.model)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model,
            torch_dtype=torch.bfloat16 if self.device.type == "cuda" else torch.float32,
            device_map="cuda" if self.device.type == "cuda" else "cpu",
            low_cpu_mem_usage=True,
        )
        self.model.eval()

    def answer(self, question: str, use_knowledge: bool = True,
               max_new_tokens: int | None = None,
               temperature: float | None = None):
        hits = self.index.search(question, self.cfg.max_context) if use_knowledge else []
        messages = build_messages(question, hits, use_knowledge)
        text = self.tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        enc = self.tok(text, return_tensors="pt", truncation=True,
                       max_length=4096).to(self.device)
        max_new = min(int(max_new_tokens or self.cfg.max_new_tokens), 512)
        temp = self.cfg.temperature if temperature is None else float(temperature)
        with torch.no_grad():
            out = self.model.generate(
                **enc,
                max_new_tokens=max_new,
                do_sample=temp > 0,
                temperature=max(temp, 1e-5),
                pad_token_id=self.tok.eos_token_id,
            )
        answer = self.tok.decode(out[0][enc["input_ids"].shape[1]:],
                                 skip_special_tokens=True).strip()
        return {
            "answer": answer,
            "sources": [{"id": h.get("doc_id"), "title": h.get("title"),
                         "source": h.get("source")} for h in hits],
        }
