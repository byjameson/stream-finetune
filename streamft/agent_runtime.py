"""Session-based agent runtime around the tiny model and local tools."""
from __future__ import annotations

import ast
import operator
import re
from datetime import datetime
from pathlib import Path

from streamft.web_research import wikipedia_research

try:
    from research.chat_human import ChatHuman
except ImportError:
    from chat_human import ChatHuman

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
        lines = []
        sources = []
        unverified = False
        for hit in hits:
            verified = bool(hit.get("verified", True))
            label = ("teacher note, unverified"
                     if hit.get("kind") == "teacher" and not verified
                     else "source")
            location = hit.get("source_url") or hit["source"]
            lines.append(f"- {hit['chunk']} ({label}: {location})")
            source = {"source": hit["source"], "title": hit["title"]}
            for key in ("kind", "source_url", "teacher", "model", "verified"):
                if key in hit:
                    source[key] = hit[key]
            if not verified:
                unverified = True
            if source not in sources:
                sources.append(source)
        reply = (f"I found these notes:\n" + "\n".join(lines) + "\n\n"
                 "This is a local note, not professional advice.")
        if unverified:
            reply += "\n\nThis includes an unverified teacher-derived note."
        return {
            "reply": reply,
            "tool_calls": [{"name": "search_docs", "query": query}],
            "sources": sources,
        }

    def chat(self, text: str) -> dict:
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
        return {"reply": reply, "tool_calls": [], "sources": []}
