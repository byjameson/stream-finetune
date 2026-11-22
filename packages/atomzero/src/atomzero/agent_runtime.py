"""Session-based agent runtime around the tiny model and local tools."""
from __future__ import annotations

import ast
import operator
import re
from datetime import datetime
from pathlib import Path

from atomzero.doc_memory import add_citation_source, format_citation
from atomzero.knowledge_gaps import KnowledgeGapError
from atomzero.web_research import wikipedia_research

from atomzero.chat_human import ChatHuman

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_MATH_PREFIX = re.compile(
    r"^(?:what\s+is|what's|calculate|compute|calc)\s+", re.IGNORECASE)
_MATH_CHARS = re.compile(r"[0-9+\-*/().%\s]+")


def _eval_math(node):
    if isinstance(node, ast.Expression):
        return _eval_math(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left = _eval_math(node.left)
        right = _eval_math(node.right)
        if isinstance(node.op, ast.Pow) and (abs(left) > 1e6 or abs(right) > 100):
            raise ValueError("exponent is too large")
        value = _BIN_OPS[type(node.op)](left, right)
        if abs(value) > 10**12:
            raise ValueError("result is too large")
        return value
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_math(node.operand))
    raise ValueError("unsupported expression")


def calculate(expression: str) -> str:
    """Evaluate a small arithmetic expression without eval()."""
    expr = expression.strip().rstrip("=?").strip()
    if not expr or len(expr) > 80:
        raise ValueError("invalid expression")
    if not _MATH_CHARS.fullmatch(expr):
        raise ValueError("only numbers and + - * / % ** are allowed")
    value = _eval_math(ast.parse(expr, mode="eval"))
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return str(value)


def math_query(text: str) -> str | None:
    candidate = _MATH_PREFIX.sub("", text.strip()).strip()
    candidate = candidate.rstrip("=?").strip()
    if not candidate or not re.search(r"\d", candidate):
        return None
    if not _MATH_CHARS.fullmatch(candidate):
        return None
    if not re.search(r"[+\-*/%]", candidate):
        return None
    return candidate


class AgentSession:
    def __init__(self, session_id: str, memory_path: str | Path,
                 model=None, tokenizer=None, device=None, docs=None,
                 wiki_lang: str = "en", gap_store=None, safety=None,
                 generator=None, allow_general_chat=False):
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", session_id):
            raise ValueError("invalid session id")
        self.id = session_id
        self.docs = docs
        self.wiki_lang = wiki_lang
        self.gap_store = gap_store
        self.safety = safety
        self.human = ChatHuman(memory_path, model, tokenizer, device,
                               "transformer", docs, on_gap=self._record_gap,
                               generator=generator,
                               allow_general_chat=allow_general_chat)

    def _record_gap(self, text: str) -> None:
        if self.gap_store is None:
            return
        try:
            self.gap_store.add(text, session_id=self.id)
        except (KnowledgeGapError, OSError):
            pass

    def forget(self, text: str = "") -> dict:
        """Delete matching session memory, or all of it when text is empty."""
        return self.human.forget(text)

    def export_memory(self) -> dict:
        return self.human.export_memory()

    def import_memory(self, data: dict) -> dict:
        return self.human.import_memory(data)

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
        lines = []
        source_map = {}
        unverified = False
        for hit in hits:
            teacher = hit.get("kind") == "teacher"
            verified = bool(hit.get("verified", True))
            if teacher and not verified:
                label = "teacher note, unverified"
            elif teacher:
                reviewer = str(hit.get("reviewer", "")).strip()
                label = (f"teacher note, verified by {reviewer}"
                         if reviewer else "teacher note, verified")
            else:
                label = "source"
            location = format_citation(hit)
            lines.append(f"- {hit['chunk']} ({label}: {location})")
            add_citation_source(source_map, hit)
            if not verified:
                unverified = True
        reply = (f"I found these notes:\n" + "\n".join(lines) + "\n\n"
                 "This is a local note, not professional advice.")
        if unverified:
            reply += "\n\nThis includes an unverified teacher-derived note."
        return {
            "reply": reply,
            "tool_calls": [{"name": "search_docs", "query": query}],
            "sources": list(source_map.values()),
        }

    def chat(self, text: str) -> dict:
        if self.safety is not None:
            verdict = self.safety.check(text)
            if verdict is not None:
                return {
                    "reply": self.safety.refusal(verdict["category"]),
                    "tool_calls": [], "sources": [], "safety": verdict,
                }
        result = self._chat(text)
        if self.safety is not None:
            safe, verdict = self.safety.filter_reply(
                result.get("reply", ""))
            if verdict is not None:
                result = {**result, "reply": safe, "safety": verdict}
        return result

    def _chat(self, text: str) -> dict:
        low = text.lower().strip()
        if low in ("what time is it", "what time is it?", "time"):
            now = datetime.now().strftime("%Y-%m-%d %H:%M")
            return {"reply": f"It is {now}.", "tool_calls": [
                {"name": "time"}], "sources": []}
        if low in ("what can you do", "what can you do?", "help"):
            return {
                "reply": ("I can keep short personal notes, manage tasks, "
                          "search your local documents, look up Wikipedia "
                          "with sources, calculate simple arithmetic, and "
                          "tell the time. I am a small local model, so I am "
                          "not a general reasoning or coding assistant."),
                "tool_calls": [{"name": "help"}], "sources": [],
            }
        expression = math_query(text)
        if expression is not None:
            try:
                result = calculate(expression)
            except (ValueError, SyntaxError, ZeroDivisionError,
                    OverflowError) as exc:
                return {
                    "reply": f"I could not calculate that: {exc}.",
                    "tool_calls": [{"name": "calculator",
                                    "expression": expression}],
                    "sources": [],
                }
            return {
                "reply": f"{expression} = {result}",
                "tool_calls": [{"name": "calculator",
                                "expression": expression,
                                "result": result}],
                "sources": [],
            }
        research = re.match(
            r"^(?:research|look up|search the web for|web search)\s*:?\s*(.+)$",
            text, re.IGNORECASE)
        if research:
            return self._research(research.group(1).strip())
        if (low.startswith("search docs ") or low.startswith("search my notes ")
                or low.startswith("search documents ")):
            query = re.sub(r"^search (docs|my notes|documents)\s*:?\s*", "",
                           text, flags=re.IGNORECASE)
            return self._search_docs(query)
        reply = self.human.respond(text)
        return {"reply": reply, "tool_calls": [],
                "sources": list(getattr(self.human, "last_sources", []))}
