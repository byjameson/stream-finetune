"""HTTP server for the local instruction assistant with verified retrieval."""
from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from streamft.assistant import AssistantConfig, LocalAssistant

_lock = threading.Lock()
_state: dict = {}

PAGE = """<!doctype html><meta charset="utf-8"><title>local assistant</title>
<style>body{font:16px/1.5 system-ui;max-width:760px;margin:40px auto;padding:0 20px}
textarea{width:100%;min-height:90px}#o{white-space:pre-wrap;background:#f6f6f6;padding:12px;border-radius:8px}
</style><h2>local verified assistant</h2><p id="m" style="color:#666"></p>
<textarea id="p">What is the speed of light?</textarea><button id="g">Ask</button><pre id="o"></pre>
<script>fetch('/health').then(r=>r.json()).then(j=>document.getElementById('m').textContent=
'knowledge docs: '+j.docs+' | model: '+j.model);
document.getElementById('g').onclick=async()=>{const o=document.getElementById('o');o.textContent='...';
const r=await fetch('/v1/chat/completions',{method:'POST',headers:{'Content-Type':'application/json'},
body:JSON.stringify({messages:[{role:'user',content:document.getElementById('p').value}],max_tokens:192})});
const j=await r.json();o.textContent=j.error?('error: '+j.error):j.choices[0].message.content;};</script>"""


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _json(self, code: int, payload: dict):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._json(200, {"ok": True, "model": _state["name"],
                             "docs": _state["docs"],
                             "params": _state["params"]})
        elif self.path == "/metrics":
            self._json(200, _state["metrics"])
        elif self.path == "/v1/models":
            self._json(200, {"data": [{"id": _state["name"],
                                       "object": "model", "owned_by": "streamft"}]})
        elif self.path in ("/", "/index.html"):
            body = PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path not in ("/v1/chat/completions", "/ask", "/knowledge/search"):
            return self._json(404, {"error": "not found"})
        key = _state.get("api_key") or ""
        if key:
            auth = self.headers.get("Authorization", "")
            if auth != f"Bearer {key}":
                return self._json(401, {"error": "unauthorized"})
        try:
            n = int(self.headers.get("Content-Length", "0"))
            req = json.loads(self.rfile.read(n) or b"{}")
            if self.path == "/knowledge/search":
                with _lock:
                    hits = _state["assistant"].index.search(
                        str(req.get("query", "")), int(req.get("k", 4)))
                return self._json(200, {"hits": hits})
            messages = req.get("messages") or []
            question = (messages[-1].get("content", "") if messages
                        else req.get("prompt", ""))
            max_new = int(req.get("max_tokens", 192))
            temp = float(req.get("temperature", 0.0))
            with _lock:
                result = _state["assistant"].answer(
                    str(question), use_knowledge=bool(req.get("use_knowledge", True)),
                    max_new_tokens=max_new, temperature=temp)
                _state["metrics"]["requests"] += 1
            if self.path == "/ask":
                return self._json(200, result)
            return self._json(200, {
                "id": "local-assistant",
                "object": "chat.completion",
                "model": _state["name"],
                "choices": [{"index": 0,
                             "message": {"role": "assistant",
                                         "content": result["answer"]},
                             "finish_reason": "stop"}],
                "sources": result["sources"],
            })
        except Exception as exc:
            return self._json(400, {"error": str(exc)})

    def log_message(self, fmt, *args):
        return


def serve(model: str, db: str, host: str = "127.0.0.1", port: int = 8014,
          api_key: str = "") -> int:
    assistant = LocalAssistant(AssistantConfig(model=model, db=db))
    _state.update({
        "assistant": assistant,
        "name": model,
        "docs": assistant.index.conn.execute(
            "SELECT COUNT(*) n FROM knowledge_fts").fetchone()["n"],
        "params": sum(p.numel() for p in assistant.model.parameters()),
        "api_key": api_key or os.environ.get("ASSISTANT_API_KEY", ""),
        "metrics": {"requests": 0},
    })
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"assistant serving {model} with {db} at http://{host}:{port}",
          flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0
