"""Local HTTP server for the 20M transformer chat checkpoint.

    python -m streamft.chat_server --model checkpoints/transformer_chat_en_clean_21m

Endpoints:
  GET  /health
  GET  /v1/models
  POST /generate                 {"prompt": "...", "max_new_tokens": 64}
  POST /v1/chat/completions      {"messages": [...], "max_tokens": 64}
"""
from __future__ import annotations

import argparse
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch

from streamft.transformer import TransformerLM, generate

_lock = threading.Lock()
_state: dict = {}

PAGE = """<!doctype html><meta charset="utf-8"><title>streamft tiny chat</title>
<style>body{font:16px/1.5 system-ui;max-width:760px;margin:40px auto;padding:0 20px}
textarea{width:100%;min-height:90px}#o{white-space:pre-wrap;background:#f6f6f6;padding:12px;border-radius:8px}
</style>
<h2>streamft tiny chat</h2><p id="m" style="color:#666"></p>
<textarea id="p">hello</textarea>
<button id="g">Send</button><pre id="o"></pre>
<script>
fetch('/health').then(r=>r.json()).then(j=>document.getElementById('m').textContent=
  'params: '+j.params+' | step: '+j.step);
document.getElementById('g').onclick=async()=>{
  const o=document.getElementById('o'); o.textContent='...';
  const r=await fetch('/generate',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({prompt:document.getElementById('p').value,max_new_tokens:80,temperature:0.7,top_k:20,repetition_penalty:1.1})});
  const j=await r.json(); o.textContent=j.error ? ('error: '+j.error) : j.text;
};
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
        if self.path not in ("/generate", "/v1/chat/completions",
                             "/v1/documents/search"):
            return self._json(404, {"error": "not found"})
        key = _state.get("api_key") or ""
        if key:
            auth = self.headers.get("Authorization", "")
            if auth != f"Bearer {key}":
                return self._json(401, {"error": "unauthorized"})
        try:
            n = int(self.headers.get("Content-Length", "0"))
            req = json.loads(self.rfile.read(n) or b"{}")
            if self.path == "/v1/documents/search":
                docs = _state.get("docs")
                query = str(req.get("query", ""))
                k = int(req.get("k", 5))
                hits = docs.search(query, k=k) if docs else []
                return self._json(200, {"data": hits})
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

    def log_message(self, fmt, *args):
        return


def serve(model_path: str, tokenizer_path: str, host: str = "127.0.0.1",
          port: int = 8013, api_key: str = "", adapter: str = "",
          docs_path: str = "") -> int:
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
        "device": device,
        "name": root.name,
        "params": sum(p.numel() for p in model.parameters()),
        "config": cfg,
        "step": ck.get("step"),
        "api_key": api_key or os.environ.get("CHAT_API_KEY", ""),
        "user_id": tok.token_to_id("<|user|>"),
        "assistant_id": tok.token_to_id("<|assistant|>"),
        "metrics": {"requests": 0, "generated_tokens": 0},
    })
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"streamft tiny chat serving {model_path} at http://{host}:{port}",
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
    args = ap.parse_args()
    return serve(args.model, args.tokenizer, args.host, args.port,
                 args.api_key, args.adapter, args.docs)


if __name__ == "__main__":
    raise SystemExit(main())
