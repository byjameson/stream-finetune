"""Session-based agent runtime around the tiny model and local tools."""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from streamft.web_research import wikipedia_research

try:
    from research.chat_human import ChatHuman
except ImportError:
    from chat_human import ChatHuman


class AgentSession:
    def __init__(self, session_id: str, memory_path: str | Path,
                 model=None, tokenizer=None, device=None, docs=None,
                 wiki_lang: str = "en"):
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", session_id):
            raise ValueError("invalid session id")
        self.id = session_id
        self.docs = docs
        self.wiki_lang = wiki_lang
        self.human = ChatHuman(memory_path, model, tokenizer, device,
                               "transformer", docs)

    def _research(self, query: str, limit: int = 2) -> dict:
        if self.docs is None:
            raise ValueError("no document memory configured")
        pages = wikipedia_research(self.docs, query, self.wiki_lang, limit)
        if not pages:
            return {"reply": "I could not find a useful page.",
                    "tool_calls": [{"name": "research", "query": query}],
                    "sources": []}
        lines = "\n".join(
            f"- {p['title']} ({p['source']})" for p in pages)
        return {
            "reply": f"I indexed these pages:\n{lines}\n\n"
                     "Ask me about them and I will quote the notes.",
            "tool_calls": [{"name": "research", "query": query}],
            "sources": [{"source": p["source"], "title": p["title"]}
                        for p in pages],
        }

    def _search_docs(self, query: str) -> dict:
        hits = self.docs.search(query, k=3) if self.docs else []
        if not hits:
            return {"reply": "I found no matching local notes.",
                    "tool_calls": [{"name": "search_docs", "query": query}],
                    "sources": []}
        lines = "\n".join(
            f"- {h['chunk']} (source: {h['source']})" for h in hits)
        return {
            "reply": f"I found these notes:\n{lines}\n\n"
                     "This is a local note, not professional advice.",
            "tool_calls": [{"name": "search_docs", "query": query}],
            "sources": [{"source": h["source"], "title": h["title"]}
                        for h in hits],
        }

    def chat(self, text: str) -> dict:
        low = text.lower().strip()
        if low in ("what time is it", "what time is it?", "time"):
            now = datetime.now().strftime("%Y-%m-%d %H:%M")
            return {"reply": f"It is {now}.", "tool_calls": [
                {"name": "time"}], "sources": []}
        if low.startswith("research ") or low.startswith("research:"):
            query = text.split(":", 1)[1] if ":" in text else text[9:]
            return self._research(query.strip())
        if (low.startswith("search docs ") or low.startswith("search my notes ")
                or low.startswith("search documents ")):
            query = re.sub(r"^search (docs|my notes|documents)\s*:?\s*", "",
                           text, flags=re.IGNORECASE)
            return self._search_docs(query)
        reply = self.human.respond(text)
        return {"reply": reply, "tool_calls": [], "sources": []}
