"""Local HTTP server for the tiny transformer daily assistant.

    python -m streamft.chat_server \
      --model checkpoints/transformer_behavior_50m_long \
      --adapter checkpoints/adapters_50m/v0001 \
      --docs data/docs.db

Endpoints:
  GET  /health
  GET  /v1/models
  GET  /v1/documents
  POST /v1/documents
  POST /v1/documents/upload?source=...&title=...
  POST /v1/documents/search
  POST /v1/teach
  POST /v1/research
  POST /v1/learn_url
  POST /v1/agent/chat
  POST /generate
  POST /v1/chat/completions
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch

from streamft.transformer import TransformerLM, generate

_lock = threading.Lock()
_state: dict = {}

MAX_BODY_BYTES = 2_000_000
MAX_DOC_BYTES = 1_000_000
MAX_SOURCE_CHARS = 512
MAX_TITLE_CHARS = 200


class DocumentTooLarge(ValueError):
    pass


def ingest_document(docs, source, text, title: str = "") -> dict:
    """Validate and index a text document, returning its provenance."""
    if docs is None:
        raise ValueError("no document memory configured")
    source = str(source or "").strip()
    title = str(title or "").strip()
    if not source:
        raise ValueError("source is required")
    if len(source) > MAX_SOURCE_CHARS:
        raise ValueError(f"source is too long (max {MAX_SOURCE_CHARS} chars)")
    if len(title) > MAX_TITLE_CHARS:
        raise ValueError(f"title is too long (max {MAX_TITLE_CHARS} chars)")
    if any(ord(ch) < 32 for ch in source + title):
        raise ValueError("source/title contains control characters")
    if not isinstance(text, str):
        raise ValueError("text must be a string")
    text = text.strip()
    if not text:
        raise ValueError("text is empty")
    raw = text.encode("utf-8")
    if len(raw) > MAX_DOC_BYTES:
        raise DocumentTooLarge(
            f"document is too large (max {MAX_DOC_BYTES} bytes)")
    chunks = docs.add_text(source, text, title=title)
    return {
        "source": source,
        "title": title or Path(source).stem,
        "chunks": chunks,
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "replace": True,
    }


PAGE = """<!doctype html><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>streamft local assistant</title>
<style>
body{font:16px/1.5 system-ui;max-width:820px;margin:32px auto;padding:0 20px;color:#111}
h1{margin:0 0 4px}.hint{color:#666;font-size:14px}
#chat{display:flex;flex-direction:column;gap:10px;min-height:220px;margin:18px 0}
.msg{max-width:85%;padding:10px 12px;border-radius:10px;white-space:pre-wrap}
.user{align-self:flex-end;background:#e8f0fe}.bot{align-self:flex-start;background:#f4f4f4}
.meta{font-size:12px;color:#666;margin-top:4px}.err{background:#fde8e8}
.row{display:flex;gap:8px}.row input{flex:1;padding:10px;font:inherit}
textarea{width:100%;min-height:100px;padding:10px;font:inherit;box-sizing:border-box}
input,button{font:inherit}button{padding:8px 12px;cursor:pointer}
.chips{display:flex;flex-wrap:wrap;gap:6px;margin:8px 0 18px}
.chips button{font-size:13px;background:#fff;border:1px solid #ccc;border-radius:999px}
hr{border:0;border-top:1px solid #ddd;margin:22px 0}
</style>
<h1>Local assistant</h1>
<p id="m" class="hint">loading...</p>
<div id="chat"></div>
<div class="row"><input id="p" placeholder="Ask, add a task, search your notes...">
<button id="g">Send</button></div>
<div class="chips">
<button data-q="what can you do?">what can it do</button>
<button data-q="add task buy milk">add task</button>
<button data-q="list tasks">list tasks</button>
<button data-q="my name is Alex">remember my name</button>
<button data-q="what do you remember?">recall</button>
<button data-q="what is 12 * 7?">calculate</button>
</div>
<hr>
<h3>Teach a fact</h3>
<p class="hint">Stored in the local document index with a source. Facts do not
change model weights.</p>
<div class="row"><input id="f" placeholder="The project codename is Blue Lantern.">
<button id="t">Teach fact</button></div>
<p id="s" class="hint"></p>
<hr>
<h3>Add a document</h3>
<p class="hint">Text only (.txt/.md), max 1 MB. Stored with source, title and
SHA-256; re-adding the same source replaces it. Nothing is written to weights.</p>
<div class="row"><input id="src" value="my-note.txt" placeholder="Source (required)">
<input id="ttl" placeholder="Title (optional)"></div>
<textarea id="txt" placeholder="Paste text or choose a file..."></textarea>
<div class="row"><input id="file" type="file" accept=".txt,.md,text/plain">
<button id="add">Add document</button></div>
<p id="ds" class="hint"></p>
<script>
const sessionId = 'web-' + Math.random().toString(36).slice(2, 10);
const chat = document.getElementById('chat');
function add(role, text, extra, isError) {
  const box = document.createElement('div');
  box.className = 'msg ' + role + (isError ? ' err' : '');
  box.textContent = text;
  if (extra) {
    const meta = document.createElement('div');
    meta.className = 'meta';
    meta.textContent = extra;
    box.appendChild(meta);
  }
  chat.appendChild(box);
  box.scrollIntoView({block: 'end'});
}
function extra(result) {
  const parts = [];
  const tools = result.tool_calls || [];
  if (tools.length) parts.push('tools: ' + tools.map(t => t.name).join(', '));
  const sources = result.sources || [];
  if (sources.length) parts.push('sources: ' + sources.map(s => s.title + ' (' + s.source + ')').join('; '));
  return parts.join(' | ');
}
async function send(message) {
  const text = (message || document.getElementById('p').value).trim();
  if (!text) return;
  document.getElementById('p').value = '';
  add('user', text);
  try {
    const r = await fetch('/v1/agent/chat', {method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({session_id: sessionId, message: text})});
    const j = await r.json();
    if (!r.ok) return add('bot', 'error: ' + (j.error || r.status), '', true);
    add('bot', j.reply, extra(j));
  } catch (e) {
    add('bot', 'error: ' + e, '', true);
  }
}
document.getElementById('g').onclick = () => send();
document.getElementById('p').addEventListener('keydown', e => {
  if (e.key === 'Enter') send();
});
document.querySelectorAll('.chips button').forEach(b => {
  b.onclick = () => send(b.dataset.q);
});
document.getElementById('t').onclick = async () => {
  const fact = document.getElementById('f').value.trim();
  if (!fact) return;
  const out = document.getElementById('s');
  try {
    const r = await fetch('/v1/teach', {method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({fact})});
    const j = await r.json();
    out.textContent = r.ok ? ('Saved ' + j.fact_chunks + ' chunk(s) to local memory.')
                           : ('error: ' + (j.error || r.status));
    if (r.ok) {
      document.getElementById('f').value = '';
      refreshHealth();
    }
  } catch (e) { out.textContent = 'error: ' + e; }
};
document.getElementById('file').onchange = () => {
  const file = document.getElementById('file').files[0];
  if (!file) return;
  const out = document.getElementById('ds');
  if (file.size > 1000000) {
    out.textContent = 'file is too large (max 1 MB)';
    return;
  }
  const reader = new FileReader();
  reader.onload = () => {
    document.getElementById('txt').value = reader.result;
    const src = document.getElementById('src');
    if (!src.value.trim() || src.value === 'my-note.txt') src.value = file.name;
    const title = document.getElementById('ttl');
    if (!title.value.trim()) title.value = file.name;
  };
  reader.readAsText(file);
};
document.getElementById('add').onclick = async () => {
  const source = document.getElementById('src').value.trim();
  const text = document.getElementById('txt').value;
  const out = document.getElementById('ds');
  if (!source || !text.trim()) {
    out.textContent = 'source and text are required';
    return;
  }
  try {
    const r = await fetch('/v1/documents', {method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({source, title: document.getElementById('ttl').value,
                            text})});
    const j = await r.json();
    out.textContent = r.ok
      ? ('Indexed ' + j.chunks + ' chunk(s) from ' + j.source +
         ' | sha256 ' + j.sha256.slice(0, 12) + '...')
      : ('error: ' + (j.error || r.status));
    if (r.ok) {
      document.getElementById('txt').value = '';
      document.getElementById('file').value = '';
      refreshHealth();
    }
  } catch (e) { out.textContent = 'error: ' + e; }
};
function refreshHealth() {
  fetch('/health').then(r => r.json()).then(j => {
    const docs = j.documents ? j.documents.chunks : 0;
    document.getElementById('m').textContent =
      'model: ' + j.model + ' | params: ' + j.params + ' | step: ' + j.step +
      ' | adapter: ' + (j.adapter ? 'yes' : 'none') + ' | doc chunks: ' + docs +
      ' | web research: ' + (j.web_research && j.web_research.url_fetch ? 'URL on' : 'Wikipedia');
  }).catch(() => { document.getElementById('m').textContent = 'server not ready'; });
}
refreshHealth();
</script>"""


def doc_answer(text: str, docs) -> dict | None:
    """Return grounded notes with sources, or None if nothing matches."""
    if docs is None:
        return None
    hits = docs.search(text, k=3)
    if not hits:
        return None
    notes = "\n".join(
        f"- {hit['chunk']} (source: {hit['source']})" for hit in hits[:3])
    return {
        "text": (f"I found these notes:\n{notes}\n\n"
                 "This is a local note, not professional advice."),
        "sources": [{"source": hit["source"], "title": hit["title"]}
                    for hit in hits[:3]],
    }


def chat_prompt(messages) -> str:
    """Build the chat format used during training."""
    parts = []
    for message in (messages or [])[-8:]:
        role = message.get("role", "user")
        content = str(message.get("content", "")).strip()
        if not content:
            continue
        tag = "<|assistant|>" if role == "assistant" else "<|user|>"
        parts.append(f"{tag} {content}")
    if not parts or not parts[-1].startswith("<|user|>"):
        parts.append("<|user|> hello")
    # Training conversations have no <|end|> between a user turn and the
    # assistant reply; <|end|> closes the whole conversation.
    return " ".join(parts) + " <|assistant|>"


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, code: int, body: bytes, ctype: str = "application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, payload: dict):
        self._send(code, json.dumps(payload).encode("utf-8"))

    def do_GET(self):
        if self.path == "/health":
            self._json(200, {
                "ok": True,
                "model": _state["name"],
                "params": _state["params"],
                "config": _state["config"],
                "step": _state.get("step"),
                "adapter": _state.get("adapter") or None,
                "documents": (_state["docs"].stats()
                              if _state.get("docs") else None),
                "learning_queue": _state.get("queue_path"),
                "web_research": {
                    "wikipedia_lang": _state.get("wiki_lang"),
                    "url_fetch": bool(_state.get("allow_url_fetch")),
                },
                "sessions": len(_state.get("sessions", {})),
            })
        elif self.path == "/v1/models":
            self._json(200, {"data": [{
                "id": _state["name"], "object": "model",
                "owned_by": "streamft",
            }]})
        elif self.path == "/metrics":
            self._json(200, _state["metrics"])
        elif self.path == "/v1/documents":
            docs = _state.get("docs")
            self._json(200, {"data": docs.sources() if docs else []})
        elif self.path in ("/", "/index.html"):
            self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        route = parsed.path
        if route not in ("/generate", "/v1/chat/completions",
                         "/v1/documents", "/v1/documents/search",
                         "/v1/documents/upload", "/v1/teach",
                         "/v1/learn_url", "/v1/research",
                         "/v1/agent/chat"):
            return self._json(404, {"error": "not found"})
        key = _state.get("api_key") or ""
        if key:
            auth = self.headers.get("Authorization", "")
            if auth != f"Bearer {key}":
                return self._json(401, {"error": "unauthorized"})
        try:
            n = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return self._json(400, {"error": "invalid Content-Length"})
        if n < 0:
            return self._json(400, {"error": "invalid Content-Length"})
        if n > MAX_BODY_BYTES:
            self.close_connection = True
            return self._json(413, {"error": "request body is too large"})
        if route == "/v1/documents/upload":
            return self._upload(parsed.query, n)
        try:
            req = json.loads(self.rfile.read(n) or b"{}")
        except (json.JSONDecodeError, UnicodeDecodeError):
            return self._json(400, {"error": "invalid JSON"})
        if not isinstance(req, dict):
            return self._json(400, {"error": "JSON object required"})
        try:
            if route == "/v1/documents":
                return self._ingest(req)
            if self.path == "/v1/documents/search":
                docs = _state.get("docs")
                query = str(req.get("query", ""))
                k = int(req.get("k", 5))
                hits = docs.search(query, k=k) if docs else []
                return self._json(200, {"data": hits})
            if self.path == "/v1/teach":
                return self._teach(req)
            if self.path == "/v1/learn_url":
                if not _state.get("allow_url_fetch"):
                    return self._json(403, {
                        "error": "URL fetch disabled; start with "
                                 "--allow-url-fetch"})
                docs = _state.get("docs")
                if docs is None:
                    return self._json(400, {"error": "no document memory"})
                from streamft.web_research import index_url
                try:
                    page = index_url(docs, str(req.get("url", "")))
                except Exception as exc:
                    return self._json(400, {"error": str(exc)})
                return self._json(200, page)
            if self.path == "/v1/research":
                docs = _state.get("docs")
                if docs is None:
                    return self._json(400, {"error": "no document memory"})
                from streamft.web_research import wikipedia_research
                query = str(req.get("query", "")).strip()
                if not query:
                    return self._json(400, {"error": "empty query"})
                try:
                    pages = wikipedia_research(
                        docs, query, _state.get("wiki_lang", "en"),
                        int(req.get("limit", 2)))
                except Exception as exc:
                    return self._json(400, {"error": str(exc)})
                return self._json(200, {"pages": pages})
            session_id = str(req.get("session_id", "")).strip()
            is_agent = self.path == "/v1/agent/chat"
            if is_agent or (self.path == "/v1/chat/completions"
                            and session_id):
                messages = req.get("messages") or []
                message = str(req.get("message", "")).strip()
                if not message:
                    message = next(
                        (str(m.get("content", "")) for m in reversed(messages)
                         if m.get("role") == "user"), "")
                if not message:
                    return self._json(400, {"error": "empty message"})
                try:
                    session = self._session(session_id or "default")
                    result = session.chat(message)
                except Exception as exc:
                    return self._json(400, {"error": str(exc)})
                if is_agent:
                    return self._json(200, {"session_id": session.id,
                                            **result})
                return self._json(200, {
                    "id": "tiny-agent-" + str(abs(hash(result["reply"]))
                                              % 10**12),
                    "object": "chat.completion",
                    "model": _state["name"],
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant",
                                    "content": result["reply"]},
                        "finish_reason": "stop",
                    }],
                    "tool_calls": result.get("tool_calls", []),
                    "sources": result.get("sources", []),
                })
            if self.path == "/v1/chat/completions":
                messages = req.get("messages") or []
                prompt = chat_prompt(messages)
                max_new = int(req.get("max_tokens", 64))
                doc_query = next(
                    (str(m.get("content", "")) for m in reversed(messages)
                     if m.get("role") == "user"), "")
            else:
                doc_query = str(req.get("prompt", "hello"))
                prompt = f"<|user|> {doc_query} <|assistant|>"
                max_new = int(req.get("max_new_tokens", 64))
            max_new = max(1, min(max_new, 256))
            grounded = doc_answer(doc_query, _state.get("docs"))
            if grounded:
                _state["metrics"]["requests"] += 1
                if self.path == "/generate":
                    return self._json(200, {
                        "text": grounded["text"],
                        "sources": grounded["sources"],
                        "model": _state["name"],
                    })
                return self._json(200, {
                    "id": "tiny-doc-" + str(abs(hash(grounded["text"]))
                                            % 10**12),
                    "object": "chat.completion",
                    "model": _state["name"],
                    "sources": grounded["sources"],
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant",
                                    "content": grounded["text"]},
                        "finish_reason": "stop",
                    }],
                })
            temperature = float(req.get("temperature", 0.7))
            top_k = int(req.get("top_k", 20))
            repetition_penalty = float(req.get("repetition_penalty", 1.1))
            with _lock:
                text, stop = generate(
                    _state["model"], _state["tok"], prompt,
                    max_new=max_new, temperature=temperature, top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    banned_ids=(_state["user_id"], _state["assistant_id"]),
                    device=_state["device"])
                _state["metrics"]["requests"] += 1
                _state["metrics"]["generated_tokens"] += max_new
            if self.path == "/generate":
                return self._json(200, {"text": text, "stop": stop,
                                        "model": _state["name"]})
            return self._json(200, {
                "id": "tiny-" + str(abs(hash(text)) % 10**12),
                "object": "chat.completion",
                "model": _state["name"],
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }],
            })
        except Exception as exc:  # pragma: no cover
            return self._json(400, {"error": str(exc)})

    def _ingest(self, req: dict):
        try:
            result = ingest_document(
                _state.get("docs"),
                req.get("source") or req.get("filename"),
                req.get("text") or req.get("content") or "",
                req.get("title", ""))
        except DocumentTooLarge as exc:
            return self._json(413, {"error": str(exc)})
        except ValueError as exc:
            return self._json(400, {"error": str(exc)})
        metrics = _state.get("metrics")
        if isinstance(metrics, dict):
            metrics["documents"] = metrics.get("documents", 0) + 1
        return self._json(200, result)

    def _upload(self, query: str, n: int):
        params = urllib.parse.parse_qs(query)
        source = (params.get("source", [""])[0]
                  or params.get("filename", [""])[0])
        title = params.get("title", [""])[0]
        if n > MAX_DOC_BYTES:
            self.close_connection = True
            return self._json(413, {
                "error": f"document is too large (max {MAX_DOC_BYTES} bytes)"})
        raw = self.rfile.read(n)
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            return self._json(400, {
                "error": "document must be UTF-8 text"})
        return self._ingest({"source": source, "title": title,
                             "text": text})

    def _session(self, session_id: str):
        from streamft.agent_runtime import AgentSession
        sessions = _state.setdefault("sessions", {})
        if session_id not in sessions:
            folder = Path(_state.get("session_dir", "data/sessions"))
            folder.mkdir(parents=True, exist_ok=True)
            sessions[session_id] = AgentSession(
                session_id, folder / f"{session_id}.json",
                model=_state.get("model"), tokenizer=_state.get("tok"),
                device=_state.get("device"), docs=_state.get("docs"),
                wiki_lang=_state.get("wiki_lang", "en"))
        return sessions[session_id]

    def _teach(self, req: dict):
        fact = str(req.get("fact", "")).strip()
        user = str(req.get("user", "")).strip()
        assistant = str(req.get("assistant", "")).strip()
        if not fact and not (user and assistant):
            return self._json(400, {"error": "provide fact or user+assistant"})
        if len(fact) > 4000 or len(user) > 2000 or len(assistant) > 2000:
            return self._json(400, {"error": "input too long"})
        result = {"ok": True}
        if fact:
            docs = _state.get("docs")
            if docs is None:
                return self._json(400, {"error": "no document memory configured"})
            result["fact_chunks"] = docs.add_text(
                "user_teach", fact, title="Taught facts", replace=False)
        if user and assistant:
            queue = Path(_state.get("queue_path",
                                    "data/learning_queue.jsonl"))
            queue.parent.mkdir(parents=True, exist_ok=True)
            with queue.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "user": user, "assistant": assistant,
                    "ts": int(time.time()),
                }, ensure_ascii=False) + "\n")
            result["skill_queued"] = True
        return self._json(200, result)

    def log_message(self, fmt, *args):
        return


def serve(model_path: str, tokenizer_path: str, host: str = "127.0.0.1",
          port: int = 8013, api_key: str = "", adapter: str = "",
          docs_path: str = "",
          queue_path: str = "data/learning_queue.jsonl",
          allow_url_fetch: bool = False, wiki_lang: str = "en",
          session_dir: str = "data/sessions") -> int:
    root = Path(model_path)
    ck = torch.load(root / "model.pt", map_location="cpu", weights_only=False)
    cfg = ck["config"]
    model = TransformerLM(cfg["vocab"], cfg["dim"], cfg["layers"],
                          cfg["heads"], cfg["ff"])
    model.load_state_dict(ck["state"])
    if adapter:
        from streamft.lora import apply_adapter
        apply_adapter(model, adapter)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(tokenizer_path)
    docs = None
    if docs_path:
        from streamft.doc_memory import DocMemory
        docs = DocMemory(docs_path)
    _state.update({
        "model": model,
        "tok": tok,
        "docs": docs,
        "adapter": adapter,
        "queue_path": queue_path,
        "allow_url_fetch": allow_url_fetch,
        "wiki_lang": wiki_lang,
        "session_dir": session_dir,
        "sessions": {},
        "device": device,
        "name": root.name,
        "params": sum(p.numel() for p in model.parameters()),
        "config": cfg,
        "step": ck.get("step"),
        "api_key": api_key or os.environ.get("CHAT_API_KEY", ""),
        "user_id": tok.token_to_id("<|user|>"),
        "assistant_id": tok.token_to_id("<|assistant|>"),
        "metrics": {"requests": 0, "generated_tokens": 0, "documents": 0},
    })
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"streamft local assistant serving {model_path} at http://{host}:{port}",
          flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",
                    default="checkpoints/transformer_chat_en_clean_21m")
    ap.add_argument("--tokenizer", default="data/chat_en_bpe8k.json")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8013)
    ap.add_argument("--api-key", default="")
    ap.add_argument("--adapter", default="")
    ap.add_argument("--docs", default="")
    ap.add_argument("--queue", default="data/learning_queue.jsonl")
    ap.add_argument("--allow-url-fetch", action="store_true")
    ap.add_argument("--wiki-lang", default="en")
    ap.add_argument("--session-dir", default="data/sessions")
    args = ap.parse_args()
    return serve(args.model, args.tokenizer, args.host, args.port,
                 args.api_key, args.adapter, args.docs, args.queue,
                 args.allow_url_fetch, args.wiki_lang, args.session_dir)


if __name__ == "__main__":
    raise SystemExit(main())
