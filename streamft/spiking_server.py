"""HTTP server for a saved spiking LM checkpoint.

    streamft spiking serve --model checkpoints/spiking_4m --port 8011

Endpoints:
  GET  /health
  GET  /v1/models
  POST /generate                 {"prompt": "...", "max_new_tokens": 64}
  POST /v1/chat/completions      {"messages": [...], "max_tokens": 64}
"""
from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch

from streamft.spiking import count_params, generate, load_checkpoint

_lock = threading.Lock()
_state: dict = {}

PAGE = """<!doctype html><meta charset="utf-8"><title>spiking LM</title>
<style>body{font:16px/1.5 system-ui;max-width:760px;margin:40px auto;padding:0 20px}
textarea{width:100%;min-height:90px}#o{white-space:pre-wrap;background:#f6f6f6;padding:12px;border-radius:8px}
</style>
<h2>streamft spiking</h2><p id="m" style="color:#666"></p>
<textarea id="p">Once upon a time</textarea>
<button id="g">Generate</button><pre id="o"></pre>
<script>
fetch('/health').then(r=>r.json()).then(j=>document.getElementById('m').textContent=
  'params: '+j.params+' | vocab: '+j.config.vocab);
document.getElementById('g').onclick=async()=>{
  const o=document.getElementById('o'); o.textContent='...';
  const r=await fetch('/generate',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({prompt:document.getElementById('p').value,max_new_tokens:64})});
  const j=await r.json(); o.textContent=j.error ? ('error: '+j.error) : j.text;
};
</script>"""


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
            })
        elif self.path == "/v1/models":
            self._json(200, {"data": [{
                "id": _state["name"], "object": "model",
                "owned_by": "streamft",
            }]})
        elif self.path == "/metrics":
            self._json(200, _state["metrics"])
        elif self.path in ("/", "/index.html"):
            self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path not in ("/generate", "/v1/chat/completions"):
            return self._json(404, {"error": "not found"})
        key = _state.get("api_key") or ""
        if key:
            auth = self.headers.get("Authorization", "")
            if auth != f"Bearer {key}":
                return self._json(401, {"error": "unauthorized"})
        try:
            n = int(self.headers.get("Content-Length", "0"))
            req = json.loads(self.rfile.read(n) or b"{}")
            if self.path == "/v1/chat/completions":
                messages = req.get("messages") or []
                prompt = "\n".join(
                    f"{m.get('role', 'user')}: {m.get('content', '')}"
                    for m in messages
                )
                max_new = int(req.get("max_tokens", 64))
            else:
                prompt = str(req.get("prompt", ""))
                max_new = int(req.get("max_new_tokens", 64))
            max_new = max(1, min(max_new, 256))
            temperature = float(req.get("temperature", 0.8))
            top_k = int(req.get("top_k", 50))
            with _lock:
                text = generate(_state["model"], _state["tok"], prompt,
                                max_new, _state["device"],
                                temperature=temperature, top_k=top_k)
                _state["metrics"]["requests"] += 1
                _state["metrics"]["generated_tokens"] += max_new
            if self.path == "/generate":
                return self._json(200, {"text": text, "model": _state["name"]})
            return self._json(200, {
                "id": "spiking-" + str(abs(hash(text)) % 10**12),
                "object": "chat.completion",
                "model": _state["name"],
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }],
            })
        except Exception as exc:  # pragma: no cover - exercised by HTTP clients
            return self._json(400, {"error": str(exc)})

    def log_message(self, fmt, *args):
        return


def serve(model_path: str, host: str = "127.0.0.1", port: int = 8011,
          api_key: str = "") -> int:
    model, tok, meta = load_checkpoint(model_path)
    if tok is None:
        raise SystemExit("checkpoint has no tokenizer.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    _state.update({
        "model": model,
        "tok": tok,
        "device": device,
        "name": meta.get("name", str(model_path)),
        "params": count_params(model),
        "config": model.config.__dict__,
        "api_key": api_key or os.environ.get("SPIKING_API_KEY", ""),
        "metrics": {"requests": 0, "generated_tokens": 0},
    })
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"streamft spiking serving {model_path} at http://{host}:{port}",
          flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0
